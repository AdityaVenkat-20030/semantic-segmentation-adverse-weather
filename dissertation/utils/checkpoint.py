"""utils/checkpoint.py — Save and load model checkpoints."""

from pathlib import Path
import torch
import torch.nn as nn


def save_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    best_miou: float,
    path:      "str | Path",
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "best_miou":  best_miou,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }, path)


def load_checkpoint(
    path:      "str | Path",
    model:     nn.Module,
    optimizer: torch.optim.Optimizer = None,
) -> tuple:
    """Returns (start_epoch, best_miou)."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_miou", 0.0)


def load_model_only(path: "str | Path", model: nn.Module) -> nn.Module:
    """Load only model weights (no optimizer state)."""
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt)  # handle both wrapped and raw state_dicts
    model.load_state_dict(state, strict=False)
    return model
