"""
evaluation/plot_model_efficiency.py

Generate efficiency comparison plots for U-Net, DeepLabV3+, and SegFormer-B2.

Inputs:
dissertation/results/summary/efficiency/model_efficiency_summary.csv

Outputs:
1. accuracy_vs_fps.png
2. latency_comparison.png
3. memory_comparison.png
4. model_size_comparison.png
5. parameters_comparison.png

Usage:
python -m dissertation.evaluation.plot_model_efficiency
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_results(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Efficiency CSV not found: {csv_path}")

    return pd.read_csv(csv_path)


def save_accuracy_vs_fps(df, output_dir):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["fps"],
        df["iddaw_miou"],
        s=120,
    )

    for _, row in df.iterrows():
        plt.text(
            row["fps"] + 1.0,
            row["iddaw_miou"],
            row["model"],
            fontsize=9,
            va="center",
        )

    plt.xlabel("FPS")
    plt.ylabel("IDD-AW mIoU")
    plt.title("Accuracy vs Real-time Inference Speed")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / "accuracy_vs_fps.png"

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()

    print(f"Saved {output_file}")


def save_latency_comparison(df, output_dir):
    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        df["model"],
        df["latency_ms"],
    )

    plt.ylabel("Latency per image (ms)")
    plt.title("Inference Latency Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, df["latency_ms"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    output_file = output_dir / "latency_comparison.png"

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()

    print(f"Saved {output_file}")


def save_memory_comparison(df, output_dir):
    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        df["model"],
        df["peak_gpu_memory_mb"],
    )

    plt.ylabel("Peak GPU memory (MB)")
    plt.title("Peak GPU Memory Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, df["peak_gpu_memory_mb"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    output_file = output_dir / "memory_comparison.png"

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()

    print(f"Saved {output_file}")


def save_model_size_comparison(df, output_dir):
    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        df["model"],
        df["checkpoint_size_mb"],
    )

    plt.ylabel("Checkpoint size (MB)")
    plt.title("Model Checkpoint Size Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, df["checkpoint_size_mb"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    output_file = output_dir / "model_size_comparison.png"

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()

    print(f"Saved {output_file}")


def save_parameters_comparison(df, output_dir):
    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        df["model"],
        df["parameters_million"],
    )

    plt.ylabel("Parameters (million)")
    plt.title("Model Parameter Count Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, df["parameters_million"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}M",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    output_file = output_dir / "parameters_comparison.png"

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()

    print(f"Saved {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot model efficiency comparison"
    )

    parser.add_argument(
        "--csv",
        default=(
            "dissertation/results/summary/efficiency/"
            "model_efficiency_summary.csv"
        ),
    )

    parser.add_argument(
        "--output_dir",
        default="dissertation/results/summary/efficiency/plots",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_results(args.csv)

    print(df)

    save_accuracy_vs_fps(
        df,
        output_dir,
    )

    save_latency_comparison(
        df,
        output_dir,
    )

    save_memory_comparison(
        df,
        output_dir,
    )

    save_model_size_comparison(
        df,
        output_dir,
    )

    save_parameters_comparison(
        df,
        output_dir,
    )

    print("\nEfficiency plots generated successfully.")


if __name__ == "__main__":
    main()