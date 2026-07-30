"""
evaluation/plot_training_curves.py

Generate loss and mIoU curves from training_log.csv.

Usage:
python -m dissertation.evaluation.plot_training_curves

Optional:
python -m dissertation.evaluation.plot_training_curves \
    --log dissertation/results/unet/training_log.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def find_log_file(log_path):
    if log_path is not None:
        log_path = Path(log_path)

        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

        return log_path

    candidates = [
        Path("dissertation/results/unet/training_log.csv"),
        Path("results/unet/training_log.csv"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find training_log.csv. "
        "Please pass it using --log."
    )


def safe_float(value):
    try:
        return float(value)
    except ValueError:
        return np.nan


def read_training_log(log_file):
    epochs = []
    train_loss = []
    val_loss = []
    train_miou = []
    val_miou = []

    with open(log_file, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(safe_float(row["train_loss"]))
            val_loss.append(safe_float(row["val_loss"]))
            train_miou.append(safe_float(row["train_miou"]))
            val_miou.append(safe_float(row["val_miou"]))

    return {
        "epoch": np.array(epochs),
        "train_loss": np.array(train_loss),
        "val_loss": np.array(val_loss),
        "train_miou": np.array(train_miou),
        "val_miou": np.array(val_miou),
    }


def plot_loss_curve(history, output_dir):
    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        label="Train Loss",
    )

    plt.plot(
        history["epoch"],
        history["val_loss"],
        marker="o",
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / "loss_curve.png"

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()

    print(f"Saved loss curve to {output_file}")


def plot_miou_curve(history, output_dir):
    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_miou"],
        marker="o",
        label="Train mIoU",
    )

    plt.plot(
        history["epoch"],
        history["val_miou"],
        marker="o",
        label="Validation mIoU",
    )

    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("Training and Validation mIoU")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / "miou_curve.png"

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()

    print(f"Saved mIoU curve to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot U-Net training curves"
    )

    parser.add_argument(
        "--log",
        default=None,
        help="Path to training_log.csv",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save plots",
    )

    args = parser.parse_args()

    log_file = find_log_file(args.log)

    if args.output_dir is None:
        output_dir = log_file.parent / "plots"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Reading log file: {log_file}")
    print(f"Saving plots to: {output_dir}")

    history = read_training_log(log_file)

    plot_loss_curve(
        history,
        output_dir,
    )

    plot_miou_curve(
        history,
        output_dir,
    )


if __name__ == "__main__":
    main()