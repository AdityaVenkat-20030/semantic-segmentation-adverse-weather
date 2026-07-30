"""
training/trainer.py — Generic Trainer
--------------------------------------
A single Trainer class used by all three models (UNet, DeepLabV3+, SegFormer).
Controls the training loop, validation, checkpointing, and logging.
Each model's entry-point script (train_unet.py, etc.) instantiates this.
"""
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import PolynomialLR
from torch.amp import GradScaler, autocast
from dissertation.evaluation.metrics import SegmentationMetrics
from dissertation.utils.checkpoint import save_checkpoint, load_checkpoint
from dissertation.utils.logger import Logger


class Trainer:
    """Generic segmentation trainer.

    Args:
        model:        A PyTorch model with forward(x) → logits (B, C, H, W).
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        config:       Dict of hyperparameters (from YAML config file).
        output_dir:   Directory to save checkpoints and logs.
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        config:       dict,
        output_dir:   str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Training device: {self.device}")

        # ---- Loss ----
        ignore_index = config.get("ignore_index", 255)
        
        if config.get("class_weights", False) and hasattr(train_loader.dataset, "get_class_weights"):
            weights = train_loader.dataset.get_class_weights()
            weights = weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(
                weight=weights,
                ignore_index=ignore_index,
            )
            print("Using class weighted CrossEntropyLoss")

        else:
            self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            )
            print("Using standard CrossEntropyLoss")

        # ---- Optimizer: differential LR (encoder 1x, decoder 10x) ----
        lr = config.get("lr", 6e-5)
        if hasattr(model, "get_encoder_params"):
            params = [
                {"params": model.get_encoder_params(), "lr": lr},
                {"params": model.get_decoder_params(), "lr": lr * 10},
            ]
        else:
            params = model.parameters()

        self.optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=config.get("weight_decay", 1e-4),
        )

        # ---- LR Scheduler: polynomial decay ----
        total_steps = config.get("epochs", 50) * len(train_loader)
        self.scheduler = PolynomialLR(
            self.optimizer,
            total_iters=total_steps,
            power=0.9,
        )

        # ---- Mixed precision ----
        self.scaler = GradScaler("cuda", enabled=(config.get("amp", True) and self.device.type == "cuda"))

        # ---- Metrics ----
        self.train_metrics = SegmentationMetrics(config["num_classes"], ignore_index)
        self.val_metrics   = SegmentationMetrics(config["num_classes"], ignore_index)

        # ---- Logging ----
        self.logger = Logger(self.output_dir / "training_log.csv")

        # ---- State ----
        self.best_miou    = 0.0
        self.start_epoch  = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, resume: Optional[str] = None):
        if resume:
            self.start_epoch, self.best_miou = load_checkpoint(
                resume, self.model, self.optimizer
            )
            print(f"Resumed from epoch {self.start_epoch}, best mIoU={self.best_miou:.4f}")

        epochs = self.config.get("epochs", 50)
        print(f"\nTraining on {self.device} for {epochs} epochs")
        print(f"Output dir: {self.output_dir}\n")

        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()

            train_loss, train_miou = self._train_epoch(epoch)
            val_loss,   val_miou   = self._val_epoch(epoch)

            elapsed = time.time() - t0
            lr_now  = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch+1:03d}/{epochs}] "
                f"Train loss={train_loss:.4f} mIoU={train_miou:.4f} | "
                f"Val loss={val_loss:.4f} mIoU={val_miou:.4f} | "
                f"LR={lr_now:.2e} | {elapsed:.0f}s"
            )

            self.logger.log({
                "epoch":      epoch + 1,
                "train_loss": train_loss,
                "train_miou": train_miou,
                "val_loss":   val_loss,
                "val_miou":   val_miou,
                "lr":         lr_now,
            })

            # Save best checkpoint
            is_best = val_miou > self.best_miou
            if is_best:
                self.best_miou = val_miou
                save_checkpoint(
                    self.model, self.optimizer, epoch + 1, self.best_miou,
                    self.output_dir / "best_checkpoint.pth",
                )
                print(f"  ✓ New best mIoU: {self.best_miou:.4f} — checkpoint saved.")

            # Save latest checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch + 1, val_miou,
                    self.output_dir / f"checkpoint_epoch{epoch+1:03d}.pth",
                )

        print(f"\nTraining complete. Best val mIoU: {self.best_miou:.4f}")

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0.0

        for batch in self.train_loader:
            images  = batch["image"].to(self.device)
            masks   = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=(self.config.get("amp", True) and self.device.type == "cuda")):
                logits = self.model(images)
                loss   = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                self.train_metrics.update(preds, masks)

        avg_loss = total_loss / len(self.train_loader)
        miou     = self.train_metrics.compute()["mIoU"]
        return avg_loss, miou

    @torch.no_grad()
    def _val_epoch(self, epoch: int):
        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0.0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            masks  = batch["mask"].to(self.device)

            logits = self.model(images)
            loss   = self.criterion(logits, masks)
            preds  = logits.argmax(dim=1)

            total_loss += loss.item()
            self.val_metrics.update(preds, masks)

        avg_loss = total_loss / len(self.val_loader)
        miou     = self.val_metrics.compute()["mIoU"]
        return avg_loss, miou
