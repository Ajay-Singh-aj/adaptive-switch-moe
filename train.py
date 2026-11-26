"""Train a Switch Transformer character LM on Tiny Shakespeare.

Examples
--------
    python train.py --moe baseline --seed 0
    python train.py --moe adaptive --seed 0
    python train.py --moe adaptive --simbal-coef 0 --run-name adaptive_no_simbal
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from moe.data import CharCorpus
from moe.model import ModelConfig, SwitchTransformerLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--moe", choices=["baseline", "adaptive"], default="baseline")
    p.add_argument("--run-name", default=None, help="defaults to '<moe>_s<seed>'")
    p.add_argument("--out-dir", default="results/runs")
    p.add_argument("--data", default="data/tiny_shakespeare.txt")
    p.add_argument("--seed", type=int, default=0)
    # optimisation
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=50)
    # model
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-experts", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--capacity-factor", type=float, default=1.0)
    p.add_argument("--load-balance-coef", type=float, default=1e-2)
    # adaptive-only
    p.add_argument("--simbal-coef", type=float, default=1e-3)
    p.add_argument("--adaptive-k", type=float, default=0.2)
    p.add_argument("--no-importance-priority", action="store_true")
    p.add_argument("--no-reassign", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, corpus, args, device) -> dict:
    model.eval()
    gen = torch.Generator().manual_seed(1234)          # identical eval batches for every run
    loss_sum, dropped, tokens = 0.0, 0, 0
    for _ in range(args.eval_batches):
        x, y = corpus.batch("val", args.batch_size, args.block_size, device, gen)
        _, loss, stats = model(x, y)
        loss_sum += loss.item()
        dropped += stats["n_dropped"]
        tokens += stats["n_tokens"]
    model.train()
    return {"val_loss": loss_sum / args.eval_batches, "val_drop_rate": dropped / tokens}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = CharCorpus(Path(args.data))
    cfg = ModelConfig(
        vocab_size=corpus.vocab_size, block_size=args.block_size, d_model=args.d_model,
        n_heads=args.n_heads, d_ff=args.d_ff, n_layers=args.n_layers, n_experts=args.n_experts,
        dropout=args.dropout, capacity_factor=args.capacity_factor, moe=args.moe,
        load_balance_coef=args.load_balance_coef, simbal_coef=args.simbal_coef,
        adaptive_k=args.adaptive_k, importance_priority=not args.no_importance_priority,
        reassign=not args.no_reassign,
    )
    model = SwitchTransformerLM(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / args.steps))
    )

    run_name = args.run_name or f"{args.moe}_s{args.seed}"
    out = Path(args.out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{run_name}] device={device} params={n_params / 1e6:.2f}M")

    gen = torch.Generator().manual_seed(args.seed)
    rows = []
    window = {"loss": 0.0, "dropped": 0, "reassigned": 0, "tokens": 0, "n": 0}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = corpus.batch("train", args.batch_size, args.block_size, device, gen)
        _, lm_loss, stats = model(x, y)
        loss = lm_loss + stats["aux_loss"]
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()

        window["loss"] += lm_loss.item()
        window["dropped"] += stats["n_dropped"]
        window["reassigned"] += stats["n_reassigned"]
        window["tokens"] += stats["n_tokens"]
        window["n"] += 1

        if step % args.eval_every == 0 or step == args.steps:
            row = {
                "step": step,
                "train_loss": window["loss"] / window["n"],
                "train_drop_rate": window["dropped"] / window["tokens"],
                "train_reassign_rate": window["reassigned"] / window["tokens"],
                **evaluate(model, corpus, args, device),
                "elapsed_s": round(time.time() - t0, 1),
            }
            rows.append(row)
            window = {k: 0 for k in window}
            print(f"[{run_name}] step {step:5d} | train {row['train_loss']:.4f} | val {row['val_loss']:.4f} "
                  f"| drop {row['train_drop_rate']:.4f} | reassign {row['train_reassign_rate']:.4f} "
                  f"| {row['elapsed_s']}s", flush=True)

    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "run_name": run_name, "seed": args.seed, "params": n_params, "config": asdict(cfg),
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
        "final": rows[-1], "expert_counts_last_batch": stats["expert_counts"].tolist(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{run_name}] saved to {out}")


if __name__ == "__main__":
    main()
