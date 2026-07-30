"""
evaluation/plot_detailed_unet_eval.py

Generate plots from detailed U-Net IDD-AW evaluation CSV files.

Inputs:
1. IDD-AW trained U-Net detailed evaluation
2. Clean IDD trained U-Net tested on IDD-AW detailed evaluation

Outputs:
1. weather_miou_comparison.png
2. weather_pixel_accuracy_comparison.png
3. safety_iou_overall_comparison.png
4. safety_iou_heatmap_iddaw_trained.png
5. safety_iou_heatmap_clean_trained_on_iddaw.png

Usage:
python -m dissertation.evaluation.plot_detailed_unet_eval
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return path


def load_data():
    iddaw_weather_csv = check_file(
        "dissertation/results/unet/detailed_eval/weather_wise_summary.csv"
    )

    iddaw_safety_csv = check_file(
        "dissertation/results/unet/detailed_eval/safety_critical_iou_by_weather.csv"
    )

    clean_to_aw_weather_csv = check_file(
        "dissertation/results/cross_domain/unet_idd_clean_to_idd_aw/detailed_eval/weather_wise_summary.csv"
    )

    clean_to_aw_safety_csv = check_file(
        "dissertation/results/cross_domain/unet_idd_clean_to_idd_aw/detailed_eval/safety_critical_iou_by_weather.csv"
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


def prepare_output_dir():
    output_dir = Path(
        "dissertation/results/summary/detailed_unet_eval"
    )

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
    plt.title("Weather-wise mIoU on IDD-AW")
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
    plt.title("Weather-wise Pixel Accuracy on IDD-AW")
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
    plt.title("Safety-critical Class IoU on IDD-AW")
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

    im = plt.imshow(
        values,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    plt.colorbar(
        im,
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
    (
        iddaw_weather_df,
        iddaw_safety_df,
        clean_to_aw_weather_df,
        clean_to_aw_safety_df,
    ) = load_data()

    output_dir = prepare_output_dir()

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
        "Safety-critical Class IoU by Weather: Trained on IDD-AW",
        output_dir / "safety_iou_heatmap_iddaw_trained.png",
    )

    plot_safety_iou_heatmap(
        clean_to_aw_safety_df,
        "Safety-critical Class IoU by Weather: Trained on IDD Clean",
        output_dir / "safety_iou_heatmap_clean_trained_on_iddaw.png",
    )

    print("\nAll detailed U-Net evaluation plots generated successfully.")


if __name__ == "__main__":
    main()
