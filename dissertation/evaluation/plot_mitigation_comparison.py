"""
evaluation/plot_mitigation_comparison.py

Compare baseline vs class-weighted loss mitigation for:
1. U-Net
2. DeepLabV3+
3. SegFormer-B2

Inputs:
- weather_wise_summary.csv
- safety_critical_iou_by_weather.csv

Outputs:
- mitigation_overall_miou_summary.csv
- mitigation_safety_critical_iou_summary.csv
- mitigation_safety_critical_iou_delta.csv
- plots for overall mIoU and safety-critical class changes

Usage:
python -m dissertation.evaluation.plot_mitigation_comparison
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("dissertation/results/summary/mitigation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


EXPERIMENTS = {
    "U-Net": {
        "baseline": "dissertation/results/unet/detailed_eval",
        "weighted": "dissertation/results/unet_weighted/detailed_eval",
    },
    "DeepLabV3+": {
        "baseline": "dissertation/results/deeplabv3plus/detailed_eval",
        "weighted": "dissertation/results/deeplabv3plus_weighted/detailed_eval",
    },
    "SegFormer-B2": {
        "baseline": "dissertation/results/segformer/detailed_eval",
        "weighted": "dissertation/results/segformer_weighted/detailed_eval",
    },
}


def find_class_column(df):
    possible_columns = [
        "class",
        "class_name",
        "name",
        "label",
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    return df.columns[0]


def load_overall_miou(exp_dir):
    path = Path(exp_dir) / "weather_wise_summary.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "weather" not in df.columns or "miou" not in df.columns:
        raise ValueError(f"Unexpected format in {path}")

    all_row = df[df["weather"].astype(str).str.lower() == "all"]

    if len(all_row) == 0:
        raise ValueError(f"No 'all' row found in {path}")

    return float(all_row.iloc[0]["miou"])


def load_safety_iou(exp_dir):
    path = Path(exp_dir) / "safety_critical_iou_by_weather.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [str(col).strip() for col in df.columns]

    # Case 1:
    # Long format:
    # class/weather/iou
    if {"weather", "iou"}.issubset(set(df.columns)):
        class_col = find_class_column(df)

        df = df.rename(columns={class_col: "class"})

        pivot_df = df.pivot_table(
            index="class",
            columns="weather",
            values="iou",
            aggfunc="mean",
        ).reset_index()

        pivot_df.columns = [str(col).strip() for col in pivot_df.columns]

        if "all" not in pivot_df.columns:
            numeric_cols = [
                col for col in pivot_df.columns
                if col != "class"
            ]

            pivot_df["all"] = pivot_df[numeric_cols].mean(
                axis=1,
                skipna=True,
            )

        return pivot_df

    # Case 2:
    # Wide format:
    # class,fog,rain,snow,lowlight
    class_col = find_class_column(df)
    df = df.rename(columns={class_col: "class"})

    # Remove non-IoU identifier columns if present
    drop_cols = [
        col for col in df.columns
        if col.lower() in ["class_id", "id", "index"]
    ]

    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Convert all non-class columns to numeric
    for col in df.columns:
        if col != "class":
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # Some scripts may use "overall" instead of "all"
    if "overall" in df.columns and "all" not in df.columns:
        df = df.rename(columns={"overall": "all"})

    # If no all/overall column exists, compute mean over weather columns
    if "all" not in df.columns:
        weather_cols = [
            col for col in df.columns
            if col != "class"
        ]

        df["all"] = df[weather_cols].mean(
            axis=1,
            skipna=True,
        )

    return df


def create_overall_summary():
    rows = []

    for model_name, paths in EXPERIMENTS.items():
        baseline_miou = load_overall_miou(paths["baseline"])
        weighted_miou = load_overall_miou(paths["weighted"])

        rows.append(
            {
                "model": model_name,
                "baseline_miou": baseline_miou,
                "weighted_miou": weighted_miou,
                "delta_miou": weighted_miou - baseline_miou,
            }
        )

    df = pd.DataFrame(rows)

    output_csv = OUTPUT_DIR / "mitigation_overall_miou_summary.csv"
    df.to_csv(output_csv, index=False)

    print(f"Saved {output_csv}")

    return df


def create_safety_class_summary():
    all_rows = []

    for model_name, paths in EXPERIMENTS.items():
        baseline_df = load_safety_iou(paths["baseline"])
        weighted_df = load_safety_iou(paths["weighted"])

        merged = baseline_df.merge(
            weighted_df,
            on="class",
            suffixes=("_baseline", "_weighted"),
        )

        weather_columns = [
            col.replace("_baseline", "")
            for col in merged.columns
            if col.endswith("_baseline")
        ]

        for _, row in merged.iterrows():
            class_name = row["class"]

            for weather in weather_columns:
                baseline_value = row[f"{weather}_baseline"]
                weighted_value = row[f"{weather}_weighted"]

                all_rows.append(
                    {
                        "model": model_name,
                        "class": class_name,
                        "weather": weather,
                        "baseline_iou": baseline_value,
                        "weighted_iou": weighted_value,
                        "delta_iou": weighted_value - baseline_value,
                    }
                )

    summary_df = pd.DataFrame(all_rows)

    output_csv = OUTPUT_DIR / "mitigation_safety_critical_iou_summary.csv"
    summary_df.to_csv(output_csv, index=False)

    delta_df = summary_df[summary_df["weather"] == "all"].copy()
    delta_df = delta_df.sort_values(
        by=["model", "delta_iou"],
        ascending=[True, False],
    )

    output_delta_csv = OUTPUT_DIR / "mitigation_safety_critical_iou_delta.csv"
    delta_df.to_csv(output_delta_csv, index=False)

    print(f"Saved {output_csv}")
    print(f"Saved {output_delta_csv}")

    return summary_df, delta_df


def plot_overall_miou(overall_df):
    x = range(len(overall_df))

    plt.figure(figsize=(9, 5))

    bar_width = 0.35

    plt.bar(
        [i - bar_width / 2 for i in x],
        overall_df["baseline_miou"],
        width=bar_width,
        label="Baseline",
    )

    plt.bar(
        [i + bar_width / 2 for i in x],
        overall_df["weighted_miou"],
        width=bar_width,
        label="Class-weighted",
    )

    plt.xticks(
        list(x),
        overall_df["model"],
        rotation=15,
        ha="right",
    )

    plt.ylabel("IDD-AW mIoU")
    plt.title("Baseline vs Class-weighted Loss: Overall mIoU")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_file = OUTPUT_DIR / "overall_miou_baseline_vs_weighted.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved {output_file}")


def plot_safety_delta(delta_df):
    for model_name in delta_df["model"].unique():
        model_df = delta_df[delta_df["model"] == model_name].copy()

        plt.figure(figsize=(10, 5))

        bars = plt.bar(
            model_df["class"],
            model_df["delta_iou"],
        )

        plt.axhline(0, linewidth=1)
        plt.ylabel("IoU change")
        plt.title(f"{model_name}: Safety-critical IoU Change after Class Weighting")
        plt.xticks(rotation=35, ha="right")
        plt.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars, model_df["delta_iou"]):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )

        plt.tight_layout()

        safe_name = (
            model_name.lower()
            .replace("+", "plus")
            .replace("-", "_")
            .replace(" ", "_")
        )

        output_file = OUTPUT_DIR / f"{safe_name}_safety_critical_iou_delta.png"
        plt.savefig(output_file, dpi=300)
        plt.close()

        print(f"Saved {output_file}")


def plot_safety_baseline_vs_weighted(summary_df):
    all_weather_df = summary_df[summary_df["weather"] == "all"].copy()

    for model_name in all_weather_df["model"].unique():
        model_df = all_weather_df[all_weather_df["model"] == model_name].copy()

        x = range(len(model_df))
        bar_width = 0.35

        plt.figure(figsize=(11, 5))

        plt.bar(
            [i - bar_width / 2 for i in x],
            model_df["baseline_iou"],
            width=bar_width,
            label="Baseline",
        )

        plt.bar(
            [i + bar_width / 2 for i in x],
            model_df["weighted_iou"],
            width=bar_width,
            label="Class-weighted",
        )

        plt.xticks(
            list(x),
            model_df["class"],
            rotation=35,
            ha="right",
        )

        plt.ylabel("IoU")
        plt.title(f"{model_name}: Safety-critical IoU Baseline vs Class-weighted")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        safe_name = (
            model_name.lower()
            .replace("+", "plus")
            .replace("-", "_")
            .replace(" ", "_")
        )

        output_file = OUTPUT_DIR / f"{safe_name}_safety_critical_baseline_vs_weighted.png"
        plt.savefig(output_file, dpi=300)
        plt.close()

        print(f"Saved {output_file}")


def main():
    overall_df = create_overall_summary()
    summary_df, delta_df = create_safety_class_summary()

    print("\nOverall mIoU summary:")
    print(overall_df)

    print("\nSafety-critical IoU delta:")
    print(delta_df)

    plot_overall_miou(overall_df)
    plot_safety_delta(delta_df)
    plot_safety_baseline_vs_weighted(summary_df)

    print("\nMitigation comparison complete.")


if __name__ == "__main__":
    main()