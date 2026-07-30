"""
training/train_segformer.py

SegFormer training entry point for IDD-AW dataset.

Usage:
python -m dissertation.training.train_segformer \
    --config dissertation/configs/segformer.yaml
"""

import argparse
import yaml
from torch.utils.data import DataLoader

from dissertation.configs.label_mapping import NUM_CLASSES
from dissertation.datasets.iddaw_dataset import IDDAWDataset
from dissertation.models.segformer import build_segformer
from dissertation.training.trainer import Trainer
from dissertation.utils.transforms import get_train_transforms, get_val_transforms


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer on IDD-AW")

    parser.add_argument(
        "--config",
        default="dissertation/configs/segformer.yaml",
    )

    parser.add_argument(
        "--resume",
        default=None,
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("\nConfiguration:")
    print(yaml.dump(config, default_flow_style=False))

    h = config.get("image_height", 512)
    w = config.get("image_width", 1024)

    train_dataset = IDDAWDataset(
        image_root=config["image_root"],
        mask_root=config["mask_root"],
        split="train",
        transform=get_train_transforms(h, w),
    )

    val_dataset = IDDAWDataset(
        image_root=config["image_root"],
        mask_root=config["mask_root"],
        split="val",
        transform=get_val_transforms(h, w),
    )

    print(train_dataset.summary())
    print(val_dataset.summary())

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

    model = build_segformer(
        num_classes=config.get("num_classes", NUM_CLASSES),
        pretrained=config.get("pretrained", True),
        backbone=config.get("backbone", "mit_b0"),
        pretrained_checkpoint=config.get("pretrained_checkpoint", None),
    )

    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_parameters() / 1e6:.2f}M")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=config["output_dir"],
    )

    trainer.train(resume=args.resume)


if __name__ == "__main__":
    main()