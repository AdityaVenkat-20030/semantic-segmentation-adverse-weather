"""
evaluation/benchmark_model_efficiency.py

Benchmark real-time model efficiency for U-Net, DeepLabV3+, and SegFormer-B2.

This version performes:
1. PyTorch FP32 inference
2. PyTorch AMP / FP16 mixed-precision inference

Metrics:
1. Number of parameters
2. Checkpoint size
3. Average inference latency
4. P95 latency
5. FPS
6. Peak GPU memory usage
7. IDD-AW mIoU

Usage:
python -m dissertation.evaluation.benchmark_model_efficiency
"""

import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from dissertation.configs.label_mapping import NUM_CLASSES
from dissertation.models.unet import build_unet
from dissertation.models.deeplabv3plus import build_deeplabv3plus
from dissertation.models.segformer import build_segformer


IMAGE_HEIGHT = 512
IMAGE_WIDTH = 1024
BATCH_SIZE = 1

WARMUP_RUNS = 20
MEASURED_RUNS = 100

OUTPUT_DIR = Path("dissertation/results/summary/efficiency")
OUTPUT_CSV = OUTPUT_DIR / "pytorch_precision_efficiency_summary.csv"


PRECISION_MODES = [
    "fp32",
    "amp_fp16",
]


MODEL_CONFIGS = [
    {
        "model_name": "U-Net ResNet50",
        "checkpoint": "dissertation/results/unet/best_checkpoint.pth",
        "iddaw_miou": 0.4036,
        "builder": "unet",
    },
    {
        "model_name": "DeepLabV3+ ResNet50",
        "checkpoint": "dissertation/results/deeplabv3plus/best_checkpoint.pth",
        "iddaw_miou": 0.4495,
        "builder": "deeplabv3plus",
    },
    {
        "model_name": "SegFormer-B2",
        "checkpoint": "dissertation/results/segformer/best_checkpoint.pth",
        "iddaw_miou": 0.4840,
        "builder": "segformer",
    },
]


def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.replace("module.", "", 1)

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    return model


def build_model(builder_name):
    if builder_name == "unet":
        return build_unet(
            num_classes=NUM_CLASSES,
            pretrained=False,
        )

    if builder_name == "deeplabv3plus":
        return build_deeplabv3plus(
            num_classes=NUM_CLASSES,
            pretrained=False,
            output_stride=16,
        )

    if builder_name == "segformer":
        return build_segformer(
            num_classes=NUM_CLASSES,
            pretrained=False,
            backbone="mit_b2",
            pretrained_checkpoint=None,
        )

    raise ValueError(f"Unknown builder: {builder_name}")


def count_parameters(model):
    if hasattr(model, "count_parameters"):
        return model.count_parameters()

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def get_checkpoint_size_mb(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    size_mb = checkpoint_path.stat().st_size / (1024 ** 2)

    return size_mb


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_latency(model, device, precision_mode):
    model.eval()

    dummy_input = torch.randn(
        BATCH_SIZE,
        3,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        device=device,
    )

    use_amp = (
        precision_mode == "amp_fp16"
        and device.type == "cuda"
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    latency_values_ms = []

    with torch.no_grad():
        # Warm-up runs
        for _ in range(WARMUP_RUNS):
            with autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                _ = model(dummy_input)

        synchronize_if_cuda(device)

        # Measured runs
        for _ in range(MEASURED_RUNS):
            synchronize_if_cuda(device)

            start_time = time.perf_counter()

            with autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                _ = model(dummy_input)

            synchronize_if_cuda(device)

            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000.0
            latency_values_ms.append(latency_ms)

    latency_values_ms = np.array(latency_values_ms)

    avg_latency_ms = float(
        np.mean(latency_values_ms)
    )

    p95_latency_ms = float(
        np.percentile(latency_values_ms, 95)
    )

    fps = 1000.0 / avg_latency_ms

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory_mb = 0.0

    return {
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "fps": fps,
        "peak_gpu_memory_mb": peak_memory_mb,
    }


def benchmark_model(model_config, device, precision_mode):
    model_name = model_config["model_name"]
    checkpoint_path = model_config["checkpoint"]

    print(f"\nBenchmarking: {model_name}")
    print(f"Precision mode: {precision_mode}")
    print(f"Checkpoint: {checkpoint_path}")

    model = build_model(
        model_config["builder"]
    )

    model = load_model_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    model.to(device)
    model.eval()

    num_params = count_parameters(model)
    params_million = num_params / 1e6

    checkpoint_size_mb = get_checkpoint_size_mb(
        checkpoint_path
    )

    latency_metrics = benchmark_latency(
        model,
        device,
        precision_mode,
    )

    result = {
        "model": model_name,
        "precision": precision_mode,
        "parameters_million": params_million,
        "checkpoint_size_mb": checkpoint_size_mb,
        "avg_latency_ms": latency_metrics["avg_latency_ms"],
        "p95_latency_ms": latency_metrics["p95_latency_ms"],
        "fps": latency_metrics["fps"],
        "peak_gpu_memory_mb": latency_metrics["peak_gpu_memory_mb"],
        "iddaw_miou": model_config["iddaw_miou"],
    }

    print(
        f"Params={params_million:.2f}M | "
        f"Size={checkpoint_size_mb:.2f} MB | "
        f"Avg latency={result['avg_latency_ms']:.2f} ms | "
        f"P95 latency={result['p95_latency_ms']:.2f} ms | "
        f"FPS={result['fps']:.2f} | "
        f"Peak GPU Mem={result['peak_gpu_memory_mb']:.2f} MB | "
        f"IDD-AW mIoU={result['iddaw_miou']:.4f}"
    )

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def save_results(results):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
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

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)

    print(f"\nSaved efficiency summary to: {OUTPUT_CSV}")


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Input size: {IMAGE_HEIGHT} x {IMAGE_WIDTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Warmup runs: {WARMUP_RUNS}")
    print(f"Measured runs: {MEASURED_RUNS}")
    print(f"Precision modes: {PRECISION_MODES}")

    results = []

    for model_config in MODEL_CONFIGS:
        for precision_mode in PRECISION_MODES:
            result = benchmark_model(
                model_config,
                device,
                precision_mode,
            )

            results.append(result)

    save_results(results)

    print("\nReal-time precision efficiency benchmarking complete.")


if __name__ == "__main__":
    main()