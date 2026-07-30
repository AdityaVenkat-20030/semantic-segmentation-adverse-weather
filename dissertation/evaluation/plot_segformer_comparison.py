"""
evaluation/plot_segformer_comparison.py

Generate SegFormer robustness comparison CSV and plot.

Inputs:
1. SegFormer trained on IDD-AW and evaluated on IDD-AW
2. SegFormer trained on clean IDD and evaluated on clean IDD
3. SegFormer trained on clean IDD and evaluated on IDD-AW

Outputs:
1. dissertation/results/summary/segformer_results.csv
2. dissertation/results/summary/segformer_miou_comparison.png

Usage:
python -m dissertation.evaluation.plot_segformer_comparison
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def check_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return path


def read_summary_csv(path):
    path = check_file(path)

    df = pd.read_csv(path)

    if set(["metric", "value"]).issubset(df.columns):
        values = dict(
            zip(
                df["metric"],
                df["value"],
            )
        )

        return {
            "loss": float(values["loss"]),
            "pixel_accuracy": float(values["pixel_accuracy"]),
            "miou": float(values["miou"]),
        }

    raise ValueError(
        f"Unexpected summary CSV format: {path}. "
        "Expected columns: metric,value"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate SegFormer comparison CSV and mIoU plot"
    )

    parser.add_argument(
        "--iddaw_summary",
        default="dissertation/results/segformer/evaluation_summary_iddaw_trained.csv",
    )

    parser.add_argument(
        "--idd_clean_summary",
        default="dissertation/results/segformer_idd_clean/evaluation_summary_idd_clean.csv",
    )

    parser.add_argument(
        "--clean_to_iddaw_summary",
        default="dissertation/results/segformer/evaluation_summary_clean_to_iddaw.csv",
    )

    parser.add_argument(
        "--output_dir",
        default="dissertation/results/summary",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    iddaw_metrics = read_summary_csv(args.iddaw_summary)
    idd_clean_metrics = read_summary_csv(args.idd_clean_summary)
    clean_to_iddaw_metrics = read_summary_csv(args.clean_to_iddaw_summary)

    results = [
        {
            "model": "SegFormer-B2",
            "train_dataset": "IDD-AW",
            "test_dataset": "IDD-AW",
            "loss": iddaw_metrics["loss"],
            "pixel_accuracy": iddaw_metrics["pixel_accuracy"],
            "miou": iddaw_metrics["miou"],
            "experiment_type": "adverse_weather_baseline",
        },
        {
            "model": "SegFormer-B2",
            "train_dataset": "IDD Clean",
            "test_dataset": "IDD Clean",
            "loss": idd_clean_metrics["loss"],
            "pixel_accuracy": idd_clean_metrics["pixel_accuracy"],
            "miou": idd_clean_metrics["miou"],
            "experiment_type": "clean_weather_baseline",
        },
        {
            "model": "SegFormer-B2",
            "train_dataset": "IDD Clean",
            "test_dataset": "IDD-AW",
            "loss": clean_to_iddaw_metrics["loss"],
            "pixel_accuracy": clean_to_iddaw_metrics["pixel_accuracy"],
            "miou": clean_to_iddaw_metrics["miou"],
            "experiment_type": "cross_domain_robustness_test",
        },
    ]

    df = pd.DataFrame(results)

    csv_path = output_dir / "segformer_results.csv"

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

    plt.title("SegFormer Robustness Comparison")
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

    output_file = output_dir / "segformer_miou_comparison.png"

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()

    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    main()