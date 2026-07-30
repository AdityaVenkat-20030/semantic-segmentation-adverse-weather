"""
evaluation/plot_detailed_segformer_eval.py

Generate plots from detailed SegFormer IDD-AW evaluation CSV files.

Inputs:
1. IDD-AW trained SegFormer detailed evaluation
2. Clean IDD trained SegFormer tested on IDD-AW detailed evaluation

Outputs:
1. weather_miou_comparison.png
2. weather_pixel_accuracy_comparison.png
3. safety_iou_overall_comparison.png
4. safety_iou_heatmap_iddaw_trained.png
5. safety_iou_heatmap_clean_trained_on_iddaw.png

Usage:
python -m dissertation.evaluation.plot_detailed_segformer_eval \
    --iddaw_dir dissertation/results/segformer/detailed_eval \
    --cross_domain_dir dissertation/results/cross_domain/segformer_idd_clean_to_iddaw/detailed_eval \
    --output_dir dissertation/results/summary/detailed_segformer_eval
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return path


def load_data(iddaw_dir, cross_domain_dir):
    iddaw_dir = Path(iddaw_dir)
    cross_domain_dir = Path(cross_domain_dir)

    iddaw_weather_csv = check_file(
        iddaw_dir / "weather_wise_summary.csv"
    )

    iddaw_safety_csv = check_file(
        iddaw_dir / "safety_critical_iou_by_weather.csv"
    )

    clean_to_aw_weather_csv = check_file(
        cross_domain_dir / "weather_wise_summary.csv"
    )

    clean_to_aw_safety_csv = check_file(
        cross_domain_dir / "safety_critical_iou_by_weather.csv"
    )

    iddaw_weather_df = pd.read_csv(iddaw_weather_csv)
    iddaw_safety_df = pd.read_csv(iddaw_safety_csv)

    clean_to_aw_weather_df = pd.read_csv(clean_to_aw_weather_csv)
    clean_to_aw_safety_df = pd.read_csv(clean_to_aw_safety_csv)

    return (
        iddaw_weather_df,
        iddaw_safety_df,
        clean_to_aw_weather_df,
        clean_to_aw_safety_df,
    )


def prepare_output_dir(output_dir):
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def plot_weather_miou_comparison(iddaw_df, clean_to_aw_df, output_dir):
    weather_order = [
        "all",
        "fog",
        "rain",
        "snow",
        "lowlight",
    ]

    iddaw_df = iddaw_df.set_index("weather").loc[weather_order]
    clean_to_aw_df = clean_to_aw_df.set_index("weather").loc[weather_order]

    x = np.arange(len(weather_order))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        x - width / 2,
        iddaw_df["miou"],
        width,
        label="Trained on IDD-AW",
    )

    plt.bar(
        x + width / 2,
        clean_to_aw_df["miou"],
        width,
        label="Trained on IDD Clean",
    )

    plt.xlabel("Weather condition")
    plt.ylabel("mIoU")
    plt.title("SegFormer: Weather-wise mIoU on IDD-AW")
    plt.xticks(x, weather_order)

    plt.ylim(
        0,
        max(
            iddaw_df["miou"].max(),
            clean_to_aw_df["miou"].max(),
        ) + 0.1,
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    output_file = output_dir / "weather_miou_comparison.png"

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()

    print(f"Saved {output_file}")


def plot_weather_pixel_accuracy_comparison(iddaw_df, clean_to_aw_df, output_dir):
    weather_order = [
        "all",
        "fog",
        "rain",
        "snow",
        "lowlight",
    ]

    iddaw_df = iddaw_df.set_index("weather").loc[weather_order]
    clean_to_aw_df = clean_to_aw_df.set_index("weather").loc[weather_order]

    x = np.arange(len(weather_order))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        x - width / 2,
        iddaw_df["pixel_accuracy"],
        width,
        label="Trained on IDD-AW",
    )

    plt.bar(
        x + width / 2,
        clean_to_aw_df["pixel_accuracy"],
        width,
        label="Trained on IDD Clean",
    )

    plt.xlabel("Weather condition")
    plt.ylabel("Pixel accuracy")
    plt.title("SegFormer: Weather-wise Pixel Accuracy on IDD-AW")
    plt.xticks(x, weather_order)
    plt.ylim(0, 1.0)

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    output_file = output_dir / "weather_pixel_accuracy_comparison.png"

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()

    print(f"Saved {output_file}")


def plot_safety_iou_overall_comparison(iddaw_df, clean_to_aw_df, output_dir):
    iddaw_all = iddaw_df[
        iddaw_df["weather"] == "all"
    ].copy()

    clean_to_aw_all = clean_to_aw_df[
        clean_to_aw_df["weather"] == "all"
    ].copy()

    merged = iddaw_all.merge(
        clean_to_aw_all,
        on=[
            "class_id",
            "class_name",
        ],
        suffixes=(
            "_iddaw_trained",
            "_clean_trained",
        ),
    )

    merged = merged.sort_values(
        by="class_id"
    )

    classes = merged["class_name"].tolist()

    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width / 2,
        merged["iou_iddaw_trained"],
        width,
        label="Trained on IDD-AW",
    )

    plt.bar(
        x + width / 2,
        merged["iou_clean_trained"],
        width,
        label="Trained on IDD Clean",
    )

    plt.xlabel("Safety-critical class")
    plt.ylabel("IoU")
    plt.title("SegFormer: Safety-critical Class IoU on IDD-AW")

    plt.xticks(
        x,
        classes,
        rotation=35,
        ha="right",
    )

    plt.ylim(0, 1.0)

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    output_file = output_dir / "safety_iou_overall_comparison.png"

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()

    print(f"Saved {output_file}")


def plot_safety_iou_heatmap(df, title, output_file):
    weather_order = [
        "fog",
        "rain",
        "snow",
        "lowlight",
    ]

    df = df[
        df["weather"].isin(weather_order)
    ].copy()

    pivot = df.pivot(
        index="class_name",
        columns="weather",
        values="iou",
    )

    pivot = pivot[weather_order]

    pivot = pivot.sort_index()

    values = pivot.values.astype(float)

    plt.figure(figsize=(9, 7))

    image = plt.imshow(
        values,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    plt.colorbar(
        image,
        label="IoU",
    )

    plt.xticks(
        np.arange(len(weather_order)),
        weather_order,
    )

    plt.yticks(
        np.arange(len(pivot.index)),
        pivot.index,
    )

    plt.xlabel("Weather condition")
    plt.ylabel("Safety-critical class")
    plt.title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]

            if np.isnan(value):
                text = "nan"
            else:
                text = f"{value:.2f}"

            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()

    print(f"Saved {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SegFormer detailed evaluation plots"
    )

    parser.add_argument(
        "--iddaw_dir",
        default="dissertation/results/segformer/detailed_eval",
        help="Detailed evaluation folder for IDD-AW trained SegFormer",
    )

    parser.add_argument(
        "--cross_domain_dir",
        default=(
            "dissertation/results/cross_domain/"
            "segformer_idd_clean_to_iddaw/detailed_eval"
        ),
        help="Detailed evaluation folder for clean IDD trained SegFormer tested on IDD-AW",
    )

    parser.add_argument(
        "--output_dir",
        default="dissertation/results/summary/detailed_segformer_eval",
        help="Output folder for generated plots",
    )

    args = parser.parse_args()

    (
        iddaw_weather_df,
        iddaw_safety_df,
        clean_to_aw_weather_df,
        clean_to_aw_safety_df,
    ) = load_data(
        args.iddaw_dir,
        args.cross_domain_dir,
    )

    output_dir = prepare_output_dir(
        args.output_dir
    )

    plot_weather_miou_comparison(
        iddaw_weather_df,
        clean_to_aw_weather_df,
        output_dir,
    )

    plot_weather_pixel_accuracy_comparison(
        iddaw_weather_df,
        clean_to_aw_weather_df,
        output_dir,
    )

    plot_safety_iou_overall_comparison(
        iddaw_safety_df,
        clean_to_aw_safety_df,
        output_dir,
    )

    plot_safety_iou_heatmap(
        iddaw_safety_df,
        "SegFormer: Safety-critical IoU by Weather — Trained on IDD-AW",
        output_dir / "safety_iou_heatmap_iddaw_trained.png",
    )

    plot_safety_iou_heatmap(
        clean_to_aw_safety_df,
        "SegFormer: Safety-critical IoU by Weather — Trained on IDD Clean",
        output_dir / "safety_iou_heatmap_clean_trained_on_iddaw.png",
    )

    print("\nAll detailed SegFormer evaluation plots generated successfully.")


if __name__ == "__main__":
    main()