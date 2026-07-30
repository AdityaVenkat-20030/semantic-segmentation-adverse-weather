"""
evaluation/plot_realtime_efficiency.py

Generate real-time efficiency plots from:
dissertation/results/summary/efficiency/pytorch_precision_efficiency_summary.csv

Plots generated:
1. mIoU vs FPS
2. Average latency comparison
3. P95 latency comparison
4. FPS comparison
5. Peak GPU memory comparison
6. Model size comparison
7. AMP/FP16 speedup over FP32

Usage:
python -m dissertation.evaluation.plot_realtime_efficiency
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_CSV = Path(
    "dissertation/results/summary/efficiency/pytorch_precision_efficiency_summary.csv"
)

OUTPUT_DIR = Path(
    "dissertation/results/summary/efficiency/plots"
)

SUMMARY_CSV = Path(
    "dissertation/results/summary/efficiency/precision_improvement_summary.csv"
)


def clean_model_name(name):
    name = name.replace(" ResNet50", "")
    return name


def load_results():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = [
        "model",
        "precision",
        "parameters_million",
        "checkpoint_size_mb",
        "avg_latency_ms",
        "p95_latency_ms",
        "fps",
        "peak_gpu_memory_mb",
        "iddaw_miou",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in CSV: {missing_columns}"
        )

    df["model_short"] = df["model"].apply(clean_model_name)

    return df


def save_plot(fig, filename):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / filename

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_miou_vs_fps(df):
    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for _, row in df.iterrows():
        label = f"{row['model_short']} ({row['precision']})"

        ax.scatter(
            row["fps"],
            row["iddaw_miou"],
            s=90,
        )

        ax.annotate(
            label,
            (
                row["fps"],
                row["iddaw_miou"],
            ),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    ax.axvline(
        30,
        linestyle="--",
        linewidth=1,
        label="30 FPS real-time threshold",
    )

    ax.set_title(
        "Accuracy vs Real-Time Inference Speed"
    )
    ax.set_xlabel(
        "FPS"
    )
    ax.set_ylabel(
        "IDD-AW mIoU"
    )
    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )
    ax.legend()

    save_plot(
        fig,
        "miou_vs_fps_precision.png",
    )


def plot_grouped_bar(df, metric, ylabel, title, filename):
    pivot = df.pivot(
        index="model_short",
        columns="precision",
        values=metric,
    )

    ordered_columns = [
        col for col in ["fp32", "amp_fp16"]
        if col in pivot.columns
    ]

    pivot = pivot[ordered_columns]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    pivot.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )
    ax.legend(
        title="Precision"
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            fontsize=8,
        )

    save_plot(
        fig,
        filename,
    )


def plot_model_size(df):
    size_df = (
        df.drop_duplicates(
            subset=["model_short"]
        )
        .set_index("model_short")
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    size_df["checkpoint_size_mb"].plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Model Checkpoint Size Comparison"
    )
    ax.set_xlabel(
        "Model"
    )
    ax.set_ylabel(
        "Checkpoint Size (MB)"
    )
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.1f",
            fontsize=8,
        )

    save_plot(
        fig,
        "model_size_comparison.png",
    )


def create_precision_improvement_summary(df):
    fp32_df = df[df["precision"] == "fp32"].copy()
    amp_df = df[df["precision"] == "amp_fp16"].copy()

    merged = fp32_df.merge(
        amp_df,
        on="model",
        suffixes=("_fp32", "_amp_fp16"),
    )

    summary = pd.DataFrame()

    summary["model"] = merged["model"]
    summary["iddaw_miou"] = merged["iddaw_miou_fp32"]

    summary["fp32_latency_ms"] = merged["avg_latency_ms_fp32"]
    summary["amp_fp16_latency_ms"] = merged["avg_latency_ms_amp_fp16"]

    summary["latency_reduction_percent"] = (
        (
            summary["fp32_latency_ms"]
            - summary["amp_fp16_latency_ms"]
        )
        / summary["fp32_latency_ms"]
        * 100.0
    )

    summary["fp32_fps"] = merged["fps_fp32"]
    summary["amp_fp16_fps"] = merged["fps_amp_fp16"]

    summary["fps_improvement_percent"] = (
        (
            summary["amp_fp16_fps"]
            - summary["fp32_fps"]
        )
        / summary["fp32_fps"]
        * 100.0
    )

    summary["fp32_peak_gpu_memory_mb"] = merged["peak_gpu_memory_mb_fp32"]
    summary["amp_fp16_peak_gpu_memory_mb"] = merged["peak_gpu_memory_mb_amp_fp16"]

    summary["gpu_memory_change_percent"] = (
        (
            summary["amp_fp16_peak_gpu_memory_mb"]
            - summary["fp32_peak_gpu_memory_mb"]
        )
        / summary["fp32_peak_gpu_memory_mb"]
        * 100.0
    )

    SUMMARY_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    print(f"Saved: {SUMMARY_CSV}")

    return summary


def plot_amp_speedup(summary):
    plot_df = summary.copy()
    plot_df["model_short"] = plot_df["model"].apply(clean_model_name)
    plot_df = plot_df.set_index("model_short")

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    plot_df["fps_improvement_percent"].plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "AMP/FP16 FPS Improvement over FP32"
    )
    ax.set_xlabel(
        "Model"
    )
    ax.set_ylabel(
        "FPS Improvement (%)"
    )
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.1f%%",
            fontsize=8,
        )

    save_plot(
        fig,
        "amp_fp16_fps_improvement.png",
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_results()

    print("\nLoaded benchmark results:")
    print(df)

    plot_miou_vs_fps(df)

    plot_grouped_bar(
        df=df,
        metric="avg_latency_ms",
        ylabel="Average Latency (ms)",
        title="Average Inference Latency: FP32 vs AMP/FP16",
        filename="average_latency_comparison.png",
    )

    plot_grouped_bar(
        df=df,
        metric="p95_latency_ms",
        ylabel="P95 Latency (ms)",
        title="P95 Inference Latency: FP32 vs AMP/FP16",
        filename="p95_latency_comparison.png",
    )

    plot_grouped_bar(
        df=df,
        metric="fps",
        ylabel="FPS",
        title="Inference FPS: FP32 vs AMP/FP16",
        filename="fps_comparison.png",
    )

    plot_grouped_bar(
        df=df,
        metric="peak_gpu_memory_mb",
        ylabel="Peak GPU Memory (MB)",
        title="Peak GPU Memory Usage: FP32 vs AMP/FP16",
        filename="gpu_memory_comparison.png",
    )

    plot_model_size(df)

    summary = create_precision_improvement_summary(df)

    plot_amp_speedup(summary)

    print("\nReal-time efficiency plots generated successfully.")


if __name__ == "__main__":
    main()