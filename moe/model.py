"""Switch Transformer language model with a baseline and an adaptive MoE layer.

Two interchangeable feed-forward layers are provided:

* ``SwitchFFN``: the Top-1 Switch layer from Fedus et al. (2021). Each expert
  has a fixed capacity ``C = capacity_factor * T / E``; tokens over capacity are
  dropped at random and pass through on the residual path.
* ``AdaptiveSwitchFFN``: the same router with three changes:
    1. Adaptive capacity: overloaded experts get extra slack
       ``delta_i = min(k * max(0, n_i - T/E), C/2)``.
    2. Importance-priority trimming: when an expert still overflows, it keeps
       the tokens with the largest L2 norm instead of a random subset.
    3. No-Token-Left-Behind (NTLB) reassignment: overflow tokens are greedily
       moved to the expert with spare capacity that maximises
       ``p(t, j) / (1 + load_j)``. Tokens are dropped only when every expert
       is full.
  plus a SimBal regulariser that penalises off-diagonal entries of ``W W^T`` for
  the (row-normalised) router weights, discouraging experts from collapsing
  onto each other.

Both layers return ``(output, stats)`` where ``stats["aux_loss"]`` is a
differentiable tensor that the training loop adds to the language-model loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 4
    n_experts: int = 4
    dropout: float = 0.1
    capacity_factor: float = 1.0
    moe: str = "baseline"              # "baseline" or "adaptive"
    load_balance_coef: float = 1e-2
    # adaptive-only settings
    simbal_coef: float = 1e-3
    adaptive_k: float = 0.2
    importance_priority: bool = True
    reassign: bool = True


class FeedForward(nn.Module):
    """Two-layer MLP used as a single expert."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwitchFFN(nn.Module):
    """Top-1 Switch Transformer feed-forward layer with fixed expert capacity."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.capacity_factor = cfg.capacity_factor
        self.load_balance_coef = cfg.load_balance_coef
        self.experts = nn.ModuleList(
            FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_experts)
        )
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)

    # --- routing steps (overridden by the adaptive layer) -------------------
    def _capacities(self, counts: torch.Tensor, n_tokens: int) -> List[int]:
        cap = int(self.capacity_factor * n_tokens / self.n_experts)
        return [cap] * self.n_experts

    def _trim_order(self, idx: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Order in which an overflowing expert keeps its tokens (random here)."""
        return idx[torch.randperm(idx.numel(), device=idx.device)]

    def _reassign(self, overflow: torch.Tensor, probs: torch.Tensor, tokens: torch.Tensor,
                  loads: List[int], caps: List[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Baseline Switch drops every overflow token."""
        empty = overflow.new_empty(0)
        return [empty] * self.n_experts, overflow

    def _extra_aux_loss(self) -> Dict[str, torch.Tensor]:
        return {}

    # --- forward -------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch, seq, d = x.shape
        tokens = x.reshape(-1, d)
        n_tokens = tokens.size(0)

        probs = F.softmax(self.router(tokens), dim=-1)          # [T, E]
        routes = probs.argmax(dim=-1)                           # [T]
        counts = torch.bincount(routes, minlength=self.n_experts)
        caps = self._capacities(counts, n_tokens)

        assigned: List[torch.Tensor] = []
        overflow_parts: List[torch.Tensor] = []
        for e in range(self.n_experts):
            idx = torch.nonzero(routes == e, as_tuple=True)[0]
            if idx.numel() > caps[e]:
                idx = self._trim_order(idx, tokens)
                overflow_parts.append(idx[caps[e]:])
                idx = idx[: caps[e]]
            assigned.append(idx)

        overflow = torch.cat(overflow_parts) if overflow_parts else routes.new_empty(0)
        n_overflow = overflow.numel()
        loads = [a.numel() for a in assigned]
        moved, dropped = self._reassign(overflow, probs, tokens, loads, caps)

        out = torch.zeros_like(tokens)
        expert_of = routes.clone()
        for e in range(self.n_experts):
            idx = torch.cat([assigned[e], moved[e]]) if moved[e].numel() else assigned[e]
            expert_of[moved[e]] = e
            if idx.numel():
                out[idx] = self.experts[e](tokens[idx])
        if dropped.numel():
            out[dropped] = tokens[dropped]                       # residual pass-through

        gate = probs.gather(1, expert_of.unsqueeze(1))           # prob of the expert actually used
        out = out * gate

        # Switch load-balancing loss: E * sum_i f_i * P_i
        f = counts.float() / n_tokens
        load_balance = self.n_experts * torch.sum(f * probs.mean(dim=0))
        aux = {"load_balance": self.load_balance_coef * load_balance, **self._extra_aux_loss()}

        stats = {
            "aux_loss": sum(aux.values()),
            "n_tokens": n_tokens,
            "n_dropped": int(dropped.numel()),
            "n_reassigned": int(n_overflow - dropped.numel()),
            "expert_counts": torch.bincount(expert_of, minlength=self.n_experts).detach(),
        }
        return out.view(batch, seq, d), stats


class AdaptiveSwitchFFN(SwitchFFN):
    """Switch layer with adaptive capacity, importance-priority trimming,
    No-Token-Left-Behind reassignment and SimBal router regularisation."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.adaptive_k = cfg.adaptive_k
        self.simbal_coef = cfg.simbal_coef
        self.importance_priority = cfg.importance_priority
        self.reassign = cfg.reassign

    def _capacities(self, counts: torch.Tensor, n_tokens: int) -> List[int]:
        base = int(self.capacity_factor * n_tokens / self.n_experts)
        mean = n_tokens / self.n_experts
        slack = ((counts.float() - mean) * self.adaptive_k).clamp(min=0, max=base * 0.5)
        return [base + int(s) for s in slack.tolist()]

    def _trim_order(self, idx: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if not self.importance_priority:
            return super()._trim_order(idx, tokens)
        importance = tokens[idx].detach().norm(dim=-1)
        return idx[torch.argsort(importance, descending=True)]

    def _reassign(self, overflow, probs, tokens, loads, caps):
        if not self.reassign or overflow.numel() == 0:
            return super()._reassign(overflow, probs, tokens, loads, caps)

        p = probs[overflow].detach().float().cpu().numpy()      # [n_over, E]
        load = np.asarray(loads, dtype=np.float64)
        spare = np.asarray(caps) - np.asarray(loads)
        target = np.full(len(p), -1, dtype=np.int64)
        for k in range(len(p)):                                 # greedy, in priority order
            free = spare > 0
            if not free.any():
                break
            score = np.where(free, p[k] / (1.0 + load), -np.inf)
            j = int(score.argmax())
            target[k] = j
            spare[j] -= 1
            load[j] += 1

        target_t = torch.from_numpy(target).to(overflow.device)
        moved = [overflow[target_t == e] for e in range(self.n_experts)]
        return moved, overflow[target_t < 0]

    def _extra_aux_loss(self) -> Dict[str, torch.Tensor]:
        if self.simbal_coef == 0:
            return {}
        w = F.normalize(self.router.weight, dim=-1)              # [E, d]
        gram = w @ w.t()
        off_diag = gram - torch.diag(torch.diag(gram))
        return {"simbal": self.simbal_coef * off_diag.pow(2).sum()}


class Block(nn.Module):
    """Pre-norm Transformer block: causal self-attention + Switch MoE FFN."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = AdaptiveSwitchFFN(cfg) if cfg.moe == "adaptive" else SwitchFFN(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        h = self.ln1(x)
        attn, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(attn)
        ff, stats = self.ffn(self.ln2(x))
        return x + self.drop(ff), stats


class SwitchTransformerLM(nn.Module):
    """Decoder-only character language model built from Switch MoE blocks."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.moe not in ("baseline", "adaptive"):
            raise ValueError(f"moe must be 'baseline' or 'adaptive', got {cfg.moe!r}")
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        _, seq = idx.shape
        pos = torch.arange(seq, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=idx.device), diagonal=1)

        layer_stats = []
        for block in self.blocks:
            x, s = block(x, mask)
            layer_stats.append(s)
        logits = self.head(self.ln_f(x))

        stats = {
            "aux_loss": sum(s["aux_loss"] for s in layer_stats),
            "n_tokens": sum(s["n_tokens"] for s in layer_stats),
            "n_dropped": sum(s["n_dropped"] for s in layer_stats),
            "n_reassigned": sum(s["n_reassigned"] for s in layer_stats),
            "expert_counts": torch.stack([s["expert_counts"] for s in layer_stats]),
        }
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, stats
