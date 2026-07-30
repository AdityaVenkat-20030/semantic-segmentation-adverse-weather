# dissertation/scripts/test_idd_dataset.py

from dissertation.datasets.idd_dataset import IDDDataset

dataset = IDDDataset(
    image_root="dissertation/data/idd/IDD_Segmentation/leftImg8bit",
    mask_root="dissertation/data/processed/idd_mask",
    split="train",
)

print(dataset.summary())

sample = dataset[0]

print("Image:", sample["image"].shape)
print("Mask :", sample["mask"].shape)

print("Mask min:", sample["mask"].min())
print("Mask max:", sample["mask"].max())

print("Path:", sample["img_path"])