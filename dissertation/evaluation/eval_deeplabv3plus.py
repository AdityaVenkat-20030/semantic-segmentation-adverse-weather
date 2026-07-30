"""
evaluation/eval_deeplabv3plus.py

Evaluate trained DeepLabV3+ checkpoint on IDD-AW validation set.

Usage:
python -m dissertation.evaluation.eval_deeplabv3plus \
    --config dissertation/configs/deeplabv3plus.yaml \
    --checkpoint dissertation/results/deeplabv3plus/best_checkpoint.pth
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
from dissertation.models.deeplabv3plus import build_deeplabv3plus
from dissertation.configs.label_mapping import LABEL_MAP, NUM_CLASSES
from dissertation.utils.transforms import get_val_transforms


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


def save_summary(output_dir, metrics):
    output_file = output_dir / "evaluation_summary.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["metric", "value"])
        writer.writerow(["loss", metrics["loss"]])
        writer.writerow(["pixel_accuracy", metrics["pixel_accuracy"]])
        writer.writerow(["miou", metrics["miou"]])

    print(f"Saved evaluation summary to {output_file}")


def save_per_class_iou(output_dir, class_names, per_class_iou):
    output_file = output_dir / "per_class_iou.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["class_id", "class_name", "iou"])

        for class_id, iou in enumerate(per_class_iou):
            writer.writerow(
                [
                    class_id,
                    class_names[class_id],
                    iou,
                ]
            )

    print(f"Saved per-class IoU to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained DeepLabV3+ checkpoint on IDD-AW"
    )

    parser.add_argument(
        "--config",
        default="dissertation/configs/deeplabv3plus.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        default="dissertation/results/deeplabv3plus/best_checkpoint.pth",
    )

    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
    )

    parser.add_argument(
        "--weather",
        default=None,
        choices=["fog", "rain", "snow", "lowlight"],
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(
        config.get("output_dir", "dissertation/results/deeplabv3plus")
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    image_height = config.get("image_height", 512)
    image_width = config.get("image_width", 1024)

    dataset = IDDAWDataset(
        image_root=config["image_root"],
        mask_root=config["mask_root"],
        split=args.split,
        weather=args.weather,
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

    model = build_deeplabv3plus(
        num_classes=config.get("num_classes", NUM_CLASSES),
        pretrained=False,
        output_stride=config.get("output_stride", 16),
    )

    model = load_model_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=config.get("ignore_index", 255)
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_parameters() / 1e6:.2f}M")

    metrics, confusion_matrix = evaluate(
        model,
        dataloader,
        criterion,
        device,
        config,
    )

    class_names = get_class_names(
        config["num_classes"]
    )

    np.save(
        output_dir / "confusion_matrix.npy",
        confusion_matrix,
    )

    save_summary(
        output_dir,
        metrics,
    )

    save_per_class_iou(
        output_dir,
        class_names,
        metrics["per_class_iou"],
    )

    print("\nEvaluation Results")
    print("------------------")
    print(f"Loss           : {metrics['loss']:.4f}")
    print(f"Pixel Accuracy : {metrics['pixel_accuracy']:.4f}")
    print(f"mIoU           : {metrics['miou']:.4f}")


if __name__ == "__main__":
    main()