"""
evaluation/plot_deeplabv3plus_comparison.py

Generate DeepLabV3+ robustness comparison CSV and plot.

Outputs:
1. dissertation/results/summary/deeplabv3plus_results.csv
2. dissertation/results/summary/deeplabv3plus_miou_comparison.png

Usage:
python -m dissertation.evaluation.plot_deeplabv3plus_comparison
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    output_dir = Path("dissertation/results/summary")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = [
        {
            "model": "DeepLabV3+ ResNet50",
            "train_dataset": "IDD-AW",
            "test_dataset": "IDD-AW",
            "loss": 0.4691,
            "pixel_accuracy": 0.8949,
            "miou": 0.4495,
            "experiment_type": "adverse_weather_baseline",
        },
        {
            "model": "DeepLabV3+ ResNet50",
            "train_dataset": "IDD Clean",
            "test_dataset": "IDD Clean",
            "loss": 0.4256,
            "pixel_accuracy": 0.9031,
            "miou": 0.5096,
            "experiment_type": "clean_weather_baseline",
        },
        {
            "model": "DeepLabV3+ ResNet50",
            "train_dataset": "IDD Clean",
            "test_dataset": "IDD-AW",
            "loss": 0.9744,
            "pixel_accuracy": 0.8572,
            "miou": 0.3715,
            "experiment_type": "cross_domain_robustness_test",
        },
    ]

    df = pd.DataFrame(results)

    csv_path = output_dir / "deeplabv3plus_results.csv"

    df.to_csv(
        csv_path,
        index=False,
    )

    print(f"Saved results CSV to {csv_path}")

    labels = [
        "IDD-AW → IDD-AW",
        "IDD Clean → IDD Clean",
        "IDD Clean → IDD-AW",
    ]

    values = df["miou"].tolist()

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        labels,
        values,
    )

    plt.title("DeepLabV3+ Robustness Comparison")
    plt.ylabel("mIoU")

    plt.ylim(
        0,
        max(values) + 0.1,
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()

    output_file = output_dir / "deeplabv3plus_miou_comparison.png"

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()

    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    main()