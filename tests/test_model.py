"""Sanity tests for the MoE layers. Run with `python -m pytest tests`."""

import torch

from moe.model import AdaptiveSwitchFFN, ModelConfig, SwitchFFN, SwitchTransformerLM


def make_cfg(**kw):
    base = dict(vocab_size=32, block_size=16, d_model=32, n_heads=4, d_ff=64,
                n_layers=2, n_experts=4, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def skew_router(layer):
    """Send almost every token to expert 0 so capacity limits are hit."""
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[0] = 5.0


def test_model_is_causal():
    torch.manual_seed(0)
    for moe in ("baseline", "adaptive"):
        model = SwitchTransformerLM(make_cfg(moe=moe, capacity_factor=10.0)).eval()
        x = torch.randint(0, 32, (2, 16))
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % 32
        a, _, _ = model(x)
        b, _, _ = model(x2)
        assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5), moe


def test_layers_do_not_share_weights():
    model = SwitchTransformerLM(make_cfg(moe="adaptive"))
    assert model.blocks[0].ffn is not model.blocks[1].ffn
    assert model.blocks[0].ffn.router.weight.data_ptr() != model.blocks[1].ffn.router.weight.data_ptr()


def test_baseline_counts_drops():
    torch.manual_seed(0)
    layer = SwitchFFN(make_cfg(capacity_factor=1.0))
    skew_router(layer)
    x = torch.randn(4, 16, 32).abs()
    _, stats = layer(x)
    assert stats["n_dropped"] > 0
    assert stats["n_reassigned"] == 0
    assert stats["expert_counts"].sum() == stats["n_tokens"]


def test_adaptive_reassigns_instead_of_dropping():
    torch.manual_seed(0)
    layer = AdaptiveSwitchFFN(make_cfg(moe="adaptive", capacity_factor=1.0))
    skew_router(layer)
    x = torch.randn(4, 16, 32).abs()
    _, stats = layer(x)
    # total capacity >= T, so every overflow token fits somewhere
    assert stats["n_dropped"] == 0
    assert stats["n_reassigned"] > 0


def test_aux_loss_reaches_router():
    torch.manual_seed(0)
    layer = AdaptiveSwitchFFN(make_cfg(moe="adaptive"))
    _, stats = layer(torch.randn(2, 16, 32))
    stats["aux_loss"].backward()
    assert layer.router.weight.grad is not None
    assert layer.router.weight.grad.abs().sum() > 0
