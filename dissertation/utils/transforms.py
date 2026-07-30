"""
utils/transforms.py — Shared image/mask augmentation pipeline
--------------------------------------------------------------
All three models must use IDENTICAL preprocessing so comparisons are fair.
Transforms operate on (PIL Image, PIL Image) pairs to keep image and mask
geometrically in sync.
"""

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import ColorJitter as TVColorJitter


# ImageNet normalization constants
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


class Compose:
    """Chain multiple (image, mask) transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Resize:
    """Resize image (bilinear) and mask (nearest) to (height, width)."""

    def __init__(self, height: int, width: int):
        self.size = (height, width)

    def __call__(self, image, mask):
        h, w = self.size
        image = image.resize((w, h), Image.BILINEAR)
        mask  = mask.resize((w, h), Image.NEAREST)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        return image, mask

class ColorJitter:
    """Apply random brightness, contrast, saturation to image only."""

    def __init__(
        self,
        brightness: float = 0.4,
        contrast:   float = 0.4,
        saturation: float = 0.3,
        hue:        float = 0.1,
        p:          float = 0.5,
    ):
        self.p = p
        self.jitter = TVColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = self.jitter(image)
        return image, mask


class ToTensor:
    """Convert PIL image to normalized float32 tensor and mask to int64 tensor."""

    def __call__(self, image: Image.Image, mask: Image.Image):
        img = np.array(image, dtype=np.float32) / 255.0
        mean = np.array(MEAN, dtype=np.float32)
        std  = np.array(STD,  dtype=np.float32)
        img  = (img - mean) / std
        image_tensor = torch.from_numpy(img.transpose(2, 0, 1))  # CHW

        mask_tensor = torch.from_numpy(np.array(mask)).long()
        return image_tensor, mask_tensor

class RandomGaussianBlur:

    def __init__(self, p=0.3):

        self.p = p

    def __call__(self, image, mask):

        if random.random() < self.p:

            image = TF.gaussian_blur(
                image,
                kernel_size=5
            )

        return image, mask

# ------------------------------------------------------------------
# Pre-built pipelines — import these in training scripts
# ------------------------------------------------------------------

def get_train_transforms(height=512,width=1024,):
    return Compose([
        Resize(height, width),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.1,
            p=0.5,
        ),
        RandomGaussianBlur(p=0.3),
        ToTensor(),
    ])


def get_val_transforms(height=512,width=1024,):
    return Compose([
        Resize(height, width),
        ToTensor(),
    ])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for visualization.
    Input:  (3, H, W) float tensor
    Output: (H, W, 3) uint8 numpy array
    """
    mean = np.array(MEAN, dtype=np.float32)
    std  = np.array(STD,  dtype=np.float32)
    img  = tensor.permute(1, 2, 0).cpu().numpy()
    img  = img * std + mean
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img
