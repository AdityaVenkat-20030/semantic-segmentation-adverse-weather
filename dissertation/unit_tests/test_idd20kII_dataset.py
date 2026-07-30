from dissertation.datasets.idd20kII_dataset import IDD20KIIDataset
import torch

dataset = IDD20KIIDataset(
    image_root="dissertation/data/idd/idd20kII/leftImg8bit",
    mask_root="dissertation/data/processed/idd20kII_mask",
    split="train",
)

print(dataset.summary())

sample = dataset[0]

print("Image:", sample["image"].shape)
print("Mask :", sample["mask"].shape)

print("Mask min:", sample["mask"].min())
print("Mask max:", sample["mask"].max())

print("Unique mask values:")
print(torch.unique(sample["mask"]))

print("Path:", sample["img_path"])