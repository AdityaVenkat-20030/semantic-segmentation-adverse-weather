"""
training/train_unet.py

U-Net training entry point.

Usage:

python -m dissertation.training.train_unet \
    --config dissertation/configs/unet.yaml

Resume:

python -m dissertation.training.train_unet \
    --config dissertation/configs/unet.yaml \
    --resume results/unet/checkpoint_epoch010.pth
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from dissertation.datasets.iddaw_dataset import IDDAWDataset
from dissertation.configs.label_mapping import NUM_CLASSES
from dissertation.models.unet import build_unet
from dissertation.utils.transforms import get_train_transforms, get_val_transforms
from dissertation.training.trainer import Trainer


def main():

    parser = argparse.ArgumentParser(description="Train U-Net")

    parser.add_argument("--config", default="dissertation/configs/unet.yaml")

    parser.add_argument("--resume", default=None)

    args = parser.parse_args()

    # Load configuration

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("\nConfiguration:")
    print(yaml.dump(config, default_flow_style=False))

    h = config.get("image_height", 512)

    w = config.get("image_width", 1024)

    # Dataset
    train_dataset = IDDAWDataset(
        image_root=config["image_root"],
        mask_root=config["mask_root"],
        split="train",
        transform=get_train_transforms(
            h,
            w,
        ),
    )

    val_dataset = IDDAWDataset(
        image_root=config["image_root"],
        mask_root=config["mask_root"],
        split="val",
        transform=get_val_transforms(
            h,
            w,
        ),
    )

    print(train_dataset.summary())
    print(val_dataset.summary())

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # Model
    model = build_unet(
        num_classes=config.get(
            "num_classes",
            NUM_CLASSES,
        ),
        pretrained=config.get(
            "pretrained",
            True,
        ),
    )

    print(
        f"Model: {model.name}"
    )

    print(
        f"Parameters: "
        f"{model.count_parameters()/1e6:.2f}M"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=config["output_dir"],
    )

    trainer.train(
        resume=args.resume
    )

if __name__ == "__main__":
    main()