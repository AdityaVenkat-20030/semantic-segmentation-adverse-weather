"""
training/train_unet_idd_clean.py

Train U-Net on clean IDD dataset.

Usage:
python -m dissertation.training.train_unet_idd_clean \
    --config dissertation/configs/unet_idd_clean.yaml
"""

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from dissertation.datasets.idd_dataset import IDDDataset
from dissertation.models.unet import build_unet
from dissertation.utils.transforms import get_train_transforms, get_val_transforms
from dissertation.training.trainer import Trainer


def print_config(config):
    print("\nConfiguration:")

    for key, value in config.items():
        print(f"{key}: {value}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net on clean IDD dataset"
    )

    parser.add_argument(
        "--config",
        default="dissertation/configs/unet_idd_clean.yaml",
    )

    parser.add_argument(
        "--resume",
        default=None,
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print_config(config)

    output_dir = Path(config["output_dir"])

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_height = config.get("image_height", 512)
    image_width = config.get("image_width", 1024)

    train_dataset = IDDDataset(
        image_root=config["image_root"],
        mask_root=config["mask_root"],
        split="train",
        transform=get_train_transforms(
            image_height,
            image_width,
        ),
    )

    val_dataset = IDDDataset(
        image_root=config["image_root"],
        mask_root=config["mask_root"],
        split="val",
        transform=get_val_transforms(
            image_height,
            image_width,
        ),
    )

    print(train_dataset.summary())
    print(val_dataset.summary())

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 2),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("val_batch_size", 2),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    model = build_unet(
        num_classes=config["num_classes"],
        pretrained=config.get("pretrained", True),
    )

    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_parameters() / 1e6:.2f}M")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
    )

    trainer.train(
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
