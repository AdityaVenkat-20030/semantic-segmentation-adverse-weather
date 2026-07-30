"""
evaluation/plot_midsem_comparison.py

Generate U-Net midsem comparison plot.

Usage:
python -m dissertation.evaluation.plot_midsem_comparison
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = Path("dissertation/results/summary/unet_midsem_results.csv")
    output_dir = Path("dissertation/results/summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    labels = [
        "IDD-AW → IDD-AW",
        "IDD Clean → IDD Clean",
        "IDD Clean → IDD-AW",
    ]

    miou_values = df["miou"].values

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        labels,
        miou_values,
    )

    plt.ylabel("mIoU")
    plt.title("U-Net Robustness Comparison")
    plt.ylim(0, max(miou_values) + 0.1)
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, miou_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_file = output_dir / "unet_midsem_miou_comparison.png"

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()

    print(f"Saved comparison plot to {output_file}")


if __name__ == "__main__":
    main()
