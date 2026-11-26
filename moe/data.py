"""Tiny Shakespeare character-level data utilities."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Tuple

import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class CharCorpus:
    """Encodes a text file character by character with a contiguous train/val split."""

    def __init__(self, path: Path, val_fraction: float = 0.1):
        path = Path(path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading Tiny Shakespeare to {path} ...")
            urllib.request.urlretrieve(URL, path)
        text = path.read_text(encoding="utf-8")
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        stoi = {c: i for i, c in enumerate(self.chars)}
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        split = int(len(data) * (1 - val_fraction))
        # Contiguous split, so validation text never overlaps a training window.
        self.train, self.val = data[:split], data[split:]

    def batch(self, split: str, batch_size: int, block_size: int,
              device: torch.device, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)
