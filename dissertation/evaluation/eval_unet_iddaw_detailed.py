"""
evaluation/eval_unet_iddaw_detailed.py

Detailed U-Net evaluation on IDD-AW.

This script produces:
1. weather_wise_summary.csv
2. safety_critical_iou_by_weather.csv
"""

import argparse
import csv
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast

from dissertation.datasets.iddaw_dataset import IDDAWDataset
from dissertation.models.unet import build_unet
from dissertation.configs.label_mapping import LABEL_MAP, NUM_CLASSES
from dissertation.utils.transforms import get_val_transforms


WEATHER_LIST = [
    None,
    "fog",
    "rain",
    "snow",
    "lowlight",
]


SAFETY_CRITICAL_CLASSES = [
    "road",
    "sidewalk",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "autorickshaw",
    "traffic light",
    "traffic sign",
]


def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

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

    model.load_state_dict(cleaned_state_dict)

    return model


def get_class_names(num_classes):
    class_names = [f"class_{i}" for i in range(num_classes)]

    for label_name, class_id in LABEL_MAP.items():
        if 0 <= class_id < num_classes:
            if class_names[class_id] == f"class_{class_id}":
                class_names[class_id] = label_name

    return class_names


def update_confusion_matrix(confusion_matrix, preds, targets, num_classes, ignore_index):
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    valid_mask = targets != ignore_index

    preds = preds[valid_mask]
    targets = targets[valid_mask]

    valid_class_mask = (
        (targets >= 0)
        & (targets < num_classes)
        & (preds >= 0)
        & (preds < num_classes)
    )

    preds = preds[valid_class_mask]
    targets = targets[valid_class_mask]

    encoded = num_classes * targets + preds

    bincount = np.bincount(
        encoded,
        minlength=num_classes ** 2,
    )

    confusion_matrix += bincount.reshape(
        num_classes,
        num_classes,
    )

    return confusion_matrix


def compute_metrics(confusion_matrix):
    true_positive = np.diag(confusion_matrix)

    false_positive = confusion_matrix.sum(axis=0) - true_positive
    false_negative = confusion_matrix.sum(axis=1) - true_positive

    denominator = true_positive + false_positive + false_negative

    iou_per_class = np.full(
        true_positive.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = denominator > 0

    iou_per_class[valid] = (
        true_positive[valid]
        / denominator[valid]
    )

    miou = np.nanmean(iou_per_class)

    if confusion_matrix.sum() > 0:
        pixel_accuracy = true_positive.sum() / confusion_matrix.sum()
    else:
        pixel_accuracy = 0.0

    return {
        "pixel_accuracy": float(pixel_accuracy),
        "miou": float(miou),
        "per_class_iou": iou_per_class,
    }


def evaluate(model, dataloader, criterion, device, config):
    model.eval()

    num_classes = config["num_classes"]
    ignore_index = config.get("ignore_index", 255)

    confusion_matrix = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    total_loss = 0.0

    amp_enabled = (
        config.get("amp", True)
        and device.type == "cuda"
    )

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, masks)

            preds = logits.argmax(dim=1)

            total_loss += loss.item()

            confusion_matrix = update_confusion_matrix(
                confusion_matrix,
                preds,
                masks,
                num_classes,
                ignore_index,
            )

    avg_loss = total_loss / len(dataloader)

    metrics = compute_metrics(confusion_matrix)
    metrics["loss"] = avg_loss

    return metrics, confusion_matrix


def write_weather_summary(output_dir, rows):
    output_file = output_dir / "weather_wise_summary.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "weather",
                "samples",
                "loss",
                "pixel_accuracy",
                "miou",
            ]
        )

        for row in rows:
            writer.writerow(
                [
                    row["weather"],
                    row["samples"],
                    row["loss"],
                    row["pixel_accuracy"],
                    row["miou"],
                ]
            )

    print(f"Saved weather-wise summary to {output_file}")


def write_safety_class_iou(output_dir, rows):
    output_file = output_dir / "safety_critical_iou_by_weather.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "weather",
                "class_id",
                "class_name",
                "iou",
            ]
        )

        for row in rows:
            writer.writerow(
                [
                    row["weather"],
                    row["class_id"],
                    row["class_name"],
                    row["iou"],
                ]
            )

    print(f"Saved safety-critical IoU table to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Detailed U-Net evaluation on IDD-AW"
    )

    parser.add_argument(
        "--config",
        default="dissertation/configs/unet.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        default="dissertation/results/unet/best_checkpoint.pth",
    )

    parser.add_argument(
        "--output_dir",
        default="dissertation/results/unet/detailed_eval",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    model = build_unet(
        num_classes=config.get("num_classes", NUM_CLASSES),
        pretrained=False,
    )

    model = load_model_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    model.to(device)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_parameters() / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(
        ignore_index=config.get("ignore_index", 255)
    )

    image_height = config.get("image_height", 512)
    image_width = config.get("image_width", 1024)

    class_names = get_class_names(
        config["num_classes"]
    )

    weather_rows = []
    safety_rows = []

    for weather in WEATHER_LIST:
        weather_name = "all" if weather is None else weather

        print(f"\nEvaluating weather split: {weather_name}")

        dataset = IDDAWDataset(
            image_root=config["image_root"],
            mask_root=config["mask_root"],
            split="val",
            weather=weather,
            transform=get_val_transforms(
                image_height,
                image_width,
            ),
        )

        print(dataset.summary())

        dataloader = DataLoader(
            dataset,
            batch_size=config.get("val_batch_size", 2),
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=(device.type == "cuda"),
        )

        metrics, confusion_matrix = evaluate(
            model,
            dataloader,
            criterion,
            device,
            config,
        )

        np.save(
            output_dir / f"confusion_matrix_{weather_name}.npy",
            confusion_matrix,
        )

        weather_rows.append(
            {
                "weather": weather_name,
                "samples": len(dataset),
                "loss": metrics["loss"],
                "pixel_accuracy": metrics["pixel_accuracy"],
                "miou": metrics["miou"],
            }
        )

        for class_name in SAFETY_CRITICAL_CLASSES:
            if class_name not in LABEL_MAP:
                continue

            class_id = LABEL_MAP[class_name]

            if class_id >= len(metrics["per_class_iou"]):
                continue

            iou = metrics["per_class_iou"][class_id]

            safety_rows.append(
                {
                    "weather": weather_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "iou": iou,
                }
            )

        print(
            f"{weather_name}: "
            f"loss={metrics['loss']:.4f}, "
            f"pixel_acc={metrics['pixel_accuracy']:.4f}, "
            f"mIoU={metrics['miou']:.4f}"
        )

    write_weather_summary(
        output_dir,
        weather_rows,
    )

    write_safety_class_iou(
        output_dir,
        safety_rows,
    )

    print("\nDetailed evaluation complete.")


if __name__ == "__main__":
    main()
