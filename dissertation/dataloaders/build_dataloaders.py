from torch.utils.data import DataLoader

from dissertation.datasets.idd_dataset import IDDDataset
from dissertation.datasets.idd20kII_dataset import IDD20KIIDataset
from dissertation.datasets.iddaw_dataset import IDDAWDataset

from dissertation.utils.transforms import (
    get_train_transforms,
    get_val_transforms,
)


def create_dataloader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# ==========================================================
# IDD
# ==========================================================

def build_idd_dataloaders(
    batch_size=8,
    num_workers=4,
):

    train_dataset = IDDDataset(
        image_root="dissertation/data/idd/IDD_Segmentation/leftImg8bit",
        mask_root="dissertation/data/processed/idd_mask",
        split="train",
        transform=get_train_transforms(),
    )

    val_dataset = IDDDataset(
        image_root="dissertation/data/idd/IDD_Segmentation/leftImg8bit",
        mask_root="dissertation/data/processed/idd_mask",
        split="val",
        transform=get_val_transforms(),
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


# ==========================================================
# IDD20KII
# ==========================================================

def build_idd20kii_dataloaders(
    batch_size=8,
    num_workers=4,
):

    train_dataset = IDD20KIIDataset(
        image_root="dissertation/data/idd/idd20kII/leftImg8bit",
        mask_root="dissertation/data/processed/idd20kII_mask",
        split="train",
        transform=get_train_transforms(),
    )

    val_dataset = IDD20KIIDataset(
        image_root="dissertation/data/idd/idd20kII/leftImg8bit",
        mask_root="dissertation/data/processed/idd20kII_mask",
        split="val",
        transform=get_val_transforms(),
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


# ==========================================================
# IDD-AW
# ==========================================================

def build_iddaw_dataloaders(
    batch_size=8,
    num_workers=4,
    weather=None,
):

    train_dataset = IDDAWDataset(
        image_root="dissertation/data/idd_aw/IDDAW",
        mask_root="dissertation/data/processed/iddaw_masks",
        split="train",
        weather=weather,
        transform=get_train_transforms(),
    )

    val_dataset = IDDAWDataset(
        image_root="dissertation/data/idd_aw/IDDAW",
        mask_root="dissertation/data/processed/iddaw_masks",
        split="val",
        weather=weather,
        transform=get_val_transforms(),
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader