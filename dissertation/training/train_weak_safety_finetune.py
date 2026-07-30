"""
training/train_weak_safety_finetune.py

Weak safety-critical class oversampling fine-tuning.

Goal:
Improve weak safety-critical classes while preserving overall segmentation performance.

Target classes by default:
- bicycle
- traffic light
- sidewalk
- traffic sign
- rider
- motorcycle
- truck
- person

Supported models:
1. U-Net
2. DeepLabV3+
3. SegFormer-B2
"""

import argparse
import csv
import inspect
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dissertation.configs import label_mapping
from dissertation.configs.label_mapping import NUM_CLASSES
from dissertation.datasets.iddaw_dataset import IDDAWDataset
from dissertation.models.unet import build_unet
from dissertation.models.deeplabv3plus import build_deeplabv3plus
from dissertation.models.segformer import build_segformer
from dissertation.training.trainer import Trainer
from dissertation.utils.transforms import get_train_transforms, get_val_transforms


IGNORE_INDEX = 255

DEFAULT_WEAK_SAFETY_CLASS_NAMES = [
    "bicycle",
    "traffic light",
    "sidewalk",
    "traffic sign",
    "rider",
    "motorcycle",
    "truck",
    "person",
]


MODEL_DEFAULTS = {
    "unet": {
        "display_name": "U-Net",
        "config": "dissertation/configs/unet.yaml",
        "checkpoint": "dissertation/results/unet/best_checkpoint.pth",
        "output_dir": "dissertation/results/unet_weak_safety_finetune",
    },
    "deeplabv3plus": {
        "display_name": "DeepLabV3+",
        "config": "dissertation/configs/deeplabv3plus.yaml",
        "checkpoint": "dissertation/results/deeplabv3plus/best_checkpoint.pth",
        "output_dir": "dissertation/results/deeplabv3plus_weak_safety_finetune",
    },
    "segformer": {
        "display_name": "SegFormer-B2",
        "config": "dissertation/configs/segformer.yaml",
        "checkpoint": "dissertation/results/segformer/best_checkpoint.pth",
        "output_dir": "dissertation/results/segformer_weak_safety_finetune",
    },
}


def get_class_name_to_id_mapping():
    """
    Robustly find a dictionary in label_mapping.py where:
    class name -> class id
    """

    for _, obj in vars(label_mapping).items():
        if not isinstance(obj, dict):
            continue

        name_to_id = {}

        for key, value in obj.items():
            if isinstance(key, str) and isinstance(value, int):
                name_to_id[key.lower()] = value

        if "bicycle" in name_to_id:
            return name_to_id

    raise RuntimeError(
        "Could not find class-name-to-id dictionary in label_mapping.py"
    )


def resolve_target_class_ids(target_class_names, target_class_ids):
    name_to_id = get_class_name_to_id_mapping()

    resolved_ids = []

    for class_name in target_class_names:
        key = class_name.lower()

        if key not in name_to_id:
            raise ValueError(
                f"Class name not found in label mapping: {class_name}"
            )

        resolved_ids.append(name_to_id[key])

    resolved_ids.extend(target_class_ids)

    resolved_ids = sorted(set(resolved_ids))

    print("\nTarget weak safety-critical classes:")

    id_to_name = {
        value: key
        for key, value in name_to_id.items()
    }

    for class_id in resolved_ids:
        print(f"  {class_id:02d}: {id_to_name.get(class_id, 'unknown')}")

    return resolved_ids


class WeakSafetyOversampledDataset(Dataset):
    """
    Oversamples images containing at least one target weak safety-critical class.
    """

    def __init__(
        self,
        base_dataset,
        target_class_ids,
        oversample_factor=5,
        output_dir=None,
    ):
        self.base_dataset = base_dataset
        self.target_class_ids = set(target_class_ids)
        self.oversample_factor = oversample_factor
        self.output_dir = Path(output_dir) if output_dir else None

        self.positive_indices, self.positive_details = self._find_target_indices()

        original_indices = list(range(len(self.base_dataset)))

        extra_positive_indices = self.positive_indices * max(
            self.oversample_factor - 1,
            0,
        )

        self.indices = original_indices + extra_positive_indices

        print("\nWeak safety-critical class oversampling:")
        print(f"  Original training samples        : {len(self.base_dataset)}")
        print(f"  Target-positive samples          : {len(self.positive_indices)}")
        print(f"  Oversample factor                : {self.oversample_factor}")
        print(f"  Effective samples per epoch      : {len(self.indices)}")

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._save_positive_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.base_dataset[original_idx]

    def _save_positive_indices(self):
        output_csv = self.output_dir / "weak_safety_positive_training_indices.csv"

        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset_index",
                    "target_pixels",
                    "present_target_classes",
                    "mask_path",
                ],
            )

            writer.writeheader()

            for row in self.positive_details:
                writer.writerow(row)

        print(f"  Saved target-positive indices    : {output_csv}")

    def _find_target_indices(self):
        mask_paths = self._try_get_mask_paths()

        if mask_paths is not None:
            print("  Detecting target samples using mask paths...")
            return self._find_from_mask_paths(mask_paths)

        print("  Mask paths not found. Falling back to dataset scanning...")
        return self._find_from_dataset_items()

    def _try_get_mask_paths(self):
        candidate_attributes = [
            "mask_paths",
            "mask_files",
            "masks",
            "label_paths",
            "gt_paths",
            "annotation_paths",
        ]

        for attr in candidate_attributes:
            if not hasattr(self.base_dataset, attr):
                continue

            value = getattr(self.base_dataset, attr)

            if not isinstance(value, (list, tuple)):
                continue

            if len(value) != len(self.base_dataset):
                continue

            paths = [Path(v) for v in value]

            if all(path.exists() for path in paths[:5]):
                return paths

        if hasattr(self.base_dataset, "samples"):
            samples = getattr(self.base_dataset, "samples")

            if isinstance(samples, (list, tuple)) and len(samples) == len(self.base_dataset):
                paths = []

                for sample in samples:
                    mask_path = self._extract_mask_path_from_sample(sample)

                    if mask_path is None:
                        return None

                    paths.append(mask_path)

                if all(path.exists() for path in paths[:5]):
                    return paths

        return None

    def _extract_mask_path_from_sample(self, sample):
        if isinstance(sample, dict):
            for key in [
                "mask_path",
                "mask",
                "label_path",
                "gt_path",
                "annotation_path",
            ]:
                if key in sample:
                    candidate = Path(sample[key])

                    if candidate.exists():
                        return candidate

        if isinstance(sample, (list, tuple)):
            for item in sample:
                candidate = Path(item)

                if candidate.exists() and candidate.suffix.lower() in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".bmp",
                    ".tif",
                    ".tiff",
                    ".npy",
                }:
                    return candidate

        return None

    def _read_mask(self, mask_path):
        mask_path = Path(mask_path)

        if mask_path.suffix.lower() == ".npy":
            mask = np.load(mask_path)
        else:
            mask = np.array(Image.open(mask_path))

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        return mask

    def _analyze_mask(self, mask):
        present_classes = []
        total_target_pixels = 0

        for class_id in self.target_class_ids:
            pixels = int(np.sum(mask == class_id))

            if pixels > 0:
                present_classes.append(class_id)
                total_target_pixels += pixels

        return present_classes, total_target_pixels

    def _find_from_mask_paths(self, mask_paths):
        positive_indices = []
        positive_details = []

        for idx, mask_path in enumerate(mask_paths):
            mask = self._read_mask(mask_path)

            present_classes, target_pixels = self._analyze_mask(mask)

            if target_pixels > 0:
                positive_indices.append(idx)
                positive_details.append(
                    {
                        "dataset_index": idx,
                        "target_pixels": target_pixels,
                        "present_target_classes": present_classes,
                        "mask_path": str(mask_path),
                    }
                )

        return positive_indices, positive_details

    def _find_from_dataset_items(self):
        positive_indices = []
        positive_details = []

        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset[idx]
            mask = sample["mask"]

            if torch.is_tensor(mask):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = np.asarray(mask)

            present_classes, target_pixels = self._analyze_mask(mask_np)

            if target_pixels > 0:
                positive_indices.append(idx)
                positive_details.append(
                    {
                        "dataset_index": idx,
                        "target_pixels": target_pixels,
                        "present_target_classes": present_classes,
                        "mask_path": "unknown",
                    }
                )

        return positive_indices, positive_details


def load_yaml(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def call_with_supported_kwargs(function, **kwargs):
    signature = inspect.signature(function)

    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }

    return function(**supported_kwargs)


def build_transforms(config, train=True):
    height = config.get("image_height", 512)
    width = config.get("image_width", 1024)

    transform_fn = get_train_transforms if train else get_val_transforms

    try:
        return transform_fn(height, width)
    except TypeError:
        pass

    try:
        return transform_fn(image_height=height, image_width=width)
    except TypeError:
        pass

    try:
        return transform_fn(height=height, width=width)
    except TypeError:
        pass

    return transform_fn()


def build_iddaw_dataset(config, split, transform):
    image_root = config.get("image_root")
    mask_root = config.get("mask_root")

    constructor_attempts = [
        {
            "image_root": image_root,
            "mask_root": mask_root,
            "split": split,
            "transform": transform,
        },
        {
            "image_root": image_root,
            "mask_root": mask_root,
            "split": split,
            "transforms": transform,
        },
        {
            "root": image_root,
            "mask_root": mask_root,
            "split": split,
            "transform": transform,
        },
        {
            "data_root": image_root,
            "mask_root": mask_root,
            "split": split,
            "transform": transform,
        },
    ]

    last_error = None

    for kwargs in constructor_attempts:
        try:
            return IDDAWDataset(**kwargs)
        except TypeError as error:
            last_error = error

    raise TypeError(
        "Could not construct IDDAWDataset. "
        f"Last error: {last_error}"
    )


def build_model(model_name, config):
    if model_name == "unet":
        return call_with_supported_kwargs(
            build_unet,
            num_classes=config.get("num_classes", NUM_CLASSES),
            pretrained=False,
            encoder=config.get("encoder", "resnet50"),
            backbone=config.get("backbone", "resnet50"),
        )

    if model_name == "deeplabv3plus":
        return call_with_supported_kwargs(
            build_deeplabv3plus,
            num_classes=config.get("num_classes", NUM_CLASSES),
            pretrained=False,
            backbone=config.get("backbone", "resnet50"),
            output_stride=config.get("output_stride", 16),
        )

    if model_name == "segformer":
        return call_with_supported_kwargs(
            build_segformer,
            num_classes=config.get("num_classes", NUM_CLASSES),
            pretrained=False,
            backbone=config.get("backbone", "mit_b2"),
            pretrained_checkpoint=None,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def load_model_weights(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
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

    model.load_state_dict(cleaned_state_dict, strict=True)

    print(f"\nLoaded baseline checkpoint: {checkpoint_path}")

    return model


def prepare_config(args, target_class_ids):
    defaults = MODEL_DEFAULTS[args.model]

    config_path = args.config or defaults["config"]
    checkpoint_path = args.checkpoint or defaults["checkpoint"]
    output_dir = args.output_dir or defaults["output_dir"]

    config = load_yaml(config_path)

    config["model"] = args.model
    config["num_classes"] = config.get("num_classes", NUM_CLASSES)
    config["ignore_index"] = config.get("ignore_index", IGNORE_INDEX)

    config["epochs"] = args.epochs
    config["lr"] = args.lr

    # Important:
    # This experiment uses oversampling, not class weighting.
    config["class_weights"] = False

    config["target_class_ids"] = target_class_ids
    config["weak_safety_oversample_factor"] = args.oversample_factor
    config["fine_tune_from_checkpoint"] = checkpoint_path
    config["output_dir"] = output_dir

    return config, checkpoint_path, output_dir


def print_config(config):
    print("\nFine-tuning configuration:")

    for key in sorted(config.keys()):
        print(f"{key}: {config[key]}")


def main():
    parser = argparse.ArgumentParser(
        description="Weak safety-critical class oversampling fine-tuning"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["unet", "deeplabv3plus", "segformer"],
    )

    parser.add_argument(
        "--config",
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
    )

    parser.add_argument(
        "--oversample_factor",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--target_class_names",
        nargs="+",
        default=DEFAULT_WEAK_SAFETY_CLASS_NAMES,
    )

    parser.add_argument(
        "--target_class_ids",
        nargs="+",
        type=int,
        default=[],
    )

    args = parser.parse_args()

    target_class_ids = resolve_target_class_ids(
        args.target_class_names,
        args.target_class_ids,
    )

    config, checkpoint_path, output_dir = prepare_config(
        args,
        target_class_ids,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_config(config)

    saved_config_path = output_dir / "finetune_config.yaml"
    save_yaml(config, saved_config_path)

    print(f"\nSaved fine-tune config: {saved_config_path}")

    train_transform = build_transforms(config, train=True)
    val_transform = build_transforms(config, train=False)

    train_dataset_base = build_iddaw_dataset(
        config,
        split="train",
        transform=train_transform,
    )

    val_dataset = build_iddaw_dataset(
        config,
        split="val",
        transform=val_transform,
    )

    train_dataset = WeakSafetyOversampledDataset(
        train_dataset_base,
        target_class_ids=target_class_ids,
        oversample_factor=args.oversample_factor,
        output_dir=output_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 2),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("val_batch_size", 2),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    model = build_model(args.model, config)

    model = load_model_weights(
        model,
        checkpoint_path,
    )

    display_name = MODEL_DEFAULTS[args.model]["display_name"]

    print(f"\nModel: {display_name}")

    if hasattr(model, "count_parameters"):
        print(f"Parameters: {model.count_parameters() / 1e6:.2f}M")
    else:
        num_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
        print(f"Parameters: {num_params / 1e6:.2f}M")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
    )

    trainer.train()


if __name__ == "__main__":
    main()