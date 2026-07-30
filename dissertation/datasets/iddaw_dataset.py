from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dissertation.configs.label_mapping import NUM_CLASSES


WEATHER_CATEGORIES = ("fog", "lowlight", "rain", "snow")


class IDDAWDataset(Dataset):

    def __init__(
        self,
        image_root: str,
        mask_root: str,
        split: str = "train",
        weather: Optional[str] = None,
        transform=None,
        image_size=(512, 1024),
    ):

        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")

        if weather and weather.lower() not in WEATHER_CATEGORIES:
            raise ValueError(
                f"Invalid weather: {weather}. "
                f"Choose from {WEATHER_CATEGORIES}"
            )

        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)

        self.split = split
        self.weather = weather.lower() if weather else None

        self.transform = transform
        self.image_size = image_size

        self.samples = []
        self._build_file_list()

        if not self.samples:
            raise RuntimeError(
                f"No samples found for split='{split}', "
                f"weather='{weather}'"
            )

    def _build_file_list(self):

        weather_dirs = (
            [self.weather.upper()]
            if self.weather
            else [w.upper() for w in WEATHER_CATEGORIES]
        )

        for weather_name in weather_dirs:

            image_dir = (
                self.image_root /
                self.split /
                weather_name /
                "rgb"
            )

            mask_dir = (
                self.mask_root /
                self.split /
                weather_name /
                "gtSeg"
            )

            if not image_dir.exists():
                continue

            image_files = sorted(
                image_dir.rglob("*_rgb.png")
            )

            for img_path in image_files:

                relative = img_path.relative_to(image_dir)

                mask_name = img_path.name.replace(
                    "_rgb.png",
                    "_mask.png"
                )

                mask_path = (
                    mask_dir /
                    relative.parent /
                    mask_name
                )

                if mask_path.exists():

                    self.samples.append(
                        (
                            img_path,
                            mask_path,
                            weather_name.lower(),
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, mask_path, weather = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.image_size:

            h, w = self.image_size

            image = image.resize(
                (w, h),
                Image.BILINEAR,
            )

            mask = mask.resize(
                (w, h),
                Image.NEAREST,
            )

        if self.transform:
            image, mask = self.transform(image, mask)

        else:
            image = self._to_tensor(image)
            mask = torch.from_numpy(
                np.array(mask)
            ).long()

        return {
            "image": image,
            "mask": mask,
            "weather": weather,
            "img_path": str(img_path),
        }

    @staticmethod
    def _to_tensor(image):

        image = (
            np.array(image, dtype=np.float32)
            / 255.0
        )

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        image = (image - mean) / std

        return torch.from_numpy(
            image.transpose(2, 0, 1)
        )

    def get_class_weights(self):

        counts = torch.zeros(NUM_CLASSES)

        for _, mask_path, _ in self.samples:

            mask = np.array(
                Image.open(mask_path)
            )

            for class_id in range(NUM_CLASSES):
                counts[class_id] += (
                    mask == class_id
                ).sum()

        counts = counts.clamp(min=1)

        weights = 1.0 / counts
        weights = (
            weights / weights.sum()
        ) * NUM_CLASSES

        return weights

    def summary(self):

        weather_counts = {}

        for _, _, weather in self.samples:
            weather_counts[weather] = (
                weather_counts.get(weather, 0) + 1
            )

        summary = [
            f"IDDAWDataset | split={self.split} | samples={len(self.samples)}"
        ]

        summary.extend(
            f"  {weather:8s}: {count}"
            for weather, count in sorted(weather_counts.items())
        )

        return "\n".join(summary)