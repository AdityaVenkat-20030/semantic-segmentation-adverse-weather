"""
inference/predict_unet.py

Generate qualitative U-Net prediction samples.

Usage:
python -m dissertation.inference.predict_unet \
    --config dissertation/configs/unet.yaml \
    --checkpoint dissertation/results/unet/best_checkpoint.pth \
    --num_samples 20
"""

import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from PIL import Image, ImageDraw

from dissertation.datasets.iddaw_dataset import IDDAWDataset
from dissertation.models.unet import build_unet
from dissertation.configs.label_mapping import NUM_CLASSES
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


def tensor_to_image(tensor):
    image = tensor.detach().cpu().float()

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=image.dtype,
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=image.dtype,
    ).view(3, 1, 1)

    image = image * std + mean
    image = image.clamp(0, 1)

    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)

    return image


def build_palette(num_classes):
    base_palette = [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 0, 230),
        (119, 11, 32),
        (255, 128, 0),
        (64, 128, 64),
        (192, 0, 128),
        (64, 0, 192),
        (192, 128, 64),
        (128, 64, 0),
        (0, 128, 192),
        (128, 128, 0),
        (64, 64, 128),
        (192, 64, 128),
        (64, 192, 128),
        (192, 192, 128),
        (0, 64, 64),
        (128, 0, 64),
        (0, 128, 64),
        (128, 128, 64),
        (64, 0, 64),
        (192, 0, 64),
        (64, 128, 192),
        (192, 128, 192),
        (0, 64, 192),
        (128, 64, 192),
        (0, 192, 64),
        (128, 192, 64),
    ]

    palette = {}

    for class_id in range(num_classes):
        palette[class_id] = base_palette[
            class_id % len(base_palette)
        ]

    return palette


def mask_to_rgb(mask, palette, ignore_index=255):
    mask = np.array(
        mask,
        dtype=np.uint8,
    )

    rgb = np.zeros(
        (
            mask.shape[0],
            mask.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    for class_id, color in palette.items():
        rgb[mask == class_id] = color

    rgb[mask == ignore_index] = (
        0,
        0,
        0,
    )

    return rgb


def create_overlay(image, pred_rgb, alpha=0.45):
    image_pil = Image.fromarray(image).convert("RGB")
    pred_pil = Image.fromarray(pred_rgb).convert("RGB")

    overlay = Image.blend(
        image_pil,
        pred_pil,
        alpha,
    )

    return np.array(overlay)


def add_title(image, title):
    image = Image.fromarray(image).convert("RGB")

    title_height = 32

    canvas = Image.new(
        "RGB",
        (
            image.width,
            image.height + title_height,
        ),
        color=(255, 255, 255),
    )

    canvas.paste(
        image,
        (
            0,
            title_height,
        ),
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (
            10,
            8,
        ),
        title,
        fill=(0, 0, 0),
    )

    return np.array(canvas)


def make_panel(image, gt_rgb, pred_rgb, overlay):
    image_panel = add_title(
        image,
        "Input Image",
    )

    gt_panel = add_title(
        gt_rgb,
        "Ground Truth",
    )

    pred_panel = add_title(
        pred_rgb,
        "Prediction",
    )

    overlay_panel = add_title(
        overlay,
        "Overlay",
    )

    top_row = np.concatenate(
        [
            image_panel,
            gt_panel,
        ],
        axis=1,
    )

    bottom_row = np.concatenate(
        [
            pred_panel,
            overlay_panel,
        ],
        axis=1,
    )

    panel = np.concatenate(
        [
            top_row,
            bottom_row,
        ],
        axis=0,
    )

    return panel


def save_image(path, image):
    Image.fromarray(image).save(path)


def predict(model, dataloader, device, config, output_dir, num_samples):
    model.eval()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    palette = build_palette(
        config["num_classes"]
    )

    ignore_index = config.get("ignore_index", 255)

    amp_enabled = (
        config.get("amp", True)
        and device.type == "cuda"
    )

    saved = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"]

            with autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                logits = model(images)

            preds = logits.argmax(dim=1).cpu()

            batch_size = images.shape[0]

            for i in range(batch_size):
                if saved >= num_samples:
                    print(f"Saved {saved} samples to {output_dir}")
                    return

                image_np = tensor_to_image(images[i])

                gt_mask = masks[i].numpy().astype(np.uint8)
                pred_mask = preds[i].numpy().astype(np.uint8)

                gt_rgb = mask_to_rgb(
                    gt_mask,
                    palette,
                    ignore_index,
                )

                pred_rgb = mask_to_rgb(
                    pred_mask,
                    palette,
                    ignore_index,
                )

                overlay = create_overlay(
                    image_np,
                    pred_rgb,
                    alpha=0.45,
                )

                panel = make_panel(
                    image_np,
                    gt_rgb,
                    pred_rgb,
                    overlay,
                )

                prefix = f"sample_{saved:03d}"

                save_image(
                    output_dir / f"{prefix}_image.png",
                    image_np,
                )

                save_image(
                    output_dir / f"{prefix}_gt.png",
                    gt_rgb,
                )

                save_image(
                    output_dir / f"{prefix}_pred.png",
                    pred_rgb,
                )

                save_image(
                    output_dir / f"{prefix}_overlay.png",
                    overlay,
                )

                save_image(
                    output_dir / f"{prefix}_panel.png",
                    panel,
                )

                saved += 1

    print(f"Saved {saved} samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate U-Net prediction samples"
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
        "--split",
        default="val",
        choices=["train", "val"],
    )

    parser.add_argument(
        "--weather",
        default=None,
        choices=["fog", "rain", "snow", "lowlight"],
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--output_dir",
        default=None,
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

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

    if args.output_dir is None:
        output_dir = Path(
            config.get("output_dir", "dissertation/results/unet")
        ) / "predictions"
    else:
        output_dir = Path(args.output_dir)

    predict(
        model,
        dataloader,
        device,
        config,
        output_dir,
        args.num_samples,
    )


if __name__ == "__main__":
    main()