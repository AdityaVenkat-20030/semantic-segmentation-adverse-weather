from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from dissertation.configs.label_mapping import NUM_CLASSES


class IDDDataset(Dataset):
    """
    Indian Driving Dataset (IDD) semantic segmentation dataset.

    Expected structure:

    leftImg8bit/
        train/
        val/
        test/

    processed_masks/
        train/
        val/
        test/
    """

    def __init__(
        self,
        image_root: str,
        mask_root: str,
        split: str = "train",
        transform=None,
        image_size: Tuple[int, int] | None = (512, 1024),
    ):

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")

        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)

        self.split = split
        self.transform = transform
        self.image_size = image_size

        self.samples: List[Tuple[Path, Path]] = []

        self._build_file_list()

        if not self.samples:
            raise RuntimeError(
                f"No samples found for split='{split}'"
            )

    def _build_file_list(self) -> None:

        image_dir = self.image_root / self.split
        mask_dir = self.mask_root / self.split

        for img_path in sorted(
            image_dir.rglob("*_leftImg8bit.png")
        ):

            relative = img_path.relative_to(image_dir)

            mask_name = img_path.name.replace(
                "_leftImg8bit.png",
                "_gtFine_polygons.png",
            )

            mask_path = (
                mask_dir /
                relative.parent /
                mask_name
            )

            if mask_path.exists():
                self.samples.append(
                    (img_path, mask_path)
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):

        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.image_size is not None:

            h, w = self.image_size

            image = image.resize(
                (w, h),
                Image.BILINEAR,
            )

            mask = mask.resize(
                (w, h),
                Image.NEAREST,
            )

        if self.transform is not None:

            image, mask = self.transform(
                image,
                mask,
            )

        else:

            image = to_tensor(image)

            image = (
                image -
                torch.tensor(
                    [0.485, 0.456, 0.406]
                ).view(3, 1, 1)
            ) / torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(3, 1, 1)

            mask = torch.from_numpy(
                np.array(mask)
            ).long()

        return {
            "image": image,
            "mask": mask,
            "img_path": str(img_path),
        }

    def get_class_weights(self) -> torch.Tensor:

        counts = torch.zeros(NUM_CLASSES)

        for _, mask_path in self.samples:

            mask = np.array(
                Image.open(mask_path)
            )

            for cls_id in range(NUM_CLASSES):
                counts[cls_id] += (
                    mask == cls_id
                ).sum()

        counts = counts.clamp(min=1)

        weights = 1.0 / counts

        return (
            weights /
            weights.sum()
        ) * NUM_CLASSES

    def summary(self) -> str:

        return (
            f"IDDDataset | "
            f"split={self.split} | "
            f"samples={len(self.samples)}"
        )