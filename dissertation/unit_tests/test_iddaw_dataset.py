import torch
from dissertation.datasets.iddaw_dataset import IDDAWDataset

dataset = IDDAWDataset(
    image_root="dissertation/data/idd_aw/IDDAW",
    mask_root="dissertation/data/processed/iddaw_masks",
    split="train",
)

print(dataset.summary())

# Check whether any masks contain IGNORE_INDEX (255)
found_ignore = False

for i in range(min(100, len(dataset))):

    sample = dataset[i]

    if sample["mask"].max() == 255:
        print(f"Found IGNORE_INDEX in sample {i}")
        found_ignore = True
        break

if not found_ignore:
    print("No IGNORE_INDEX found in first 100 samples")

ignore_count = 0

for i in range(len(dataset)):

    sample = dataset[i]

    if (sample["mask"] == 255).any():
        ignore_count += 1

print(f"Masks containing IGNORE_INDEX: {ignore_count}")
print(f"Total masks: {len(dataset)}")

sample = dataset[1]

print("Image:", sample["image"].shape)
print("Mask :", sample["mask"].shape)
print("Weather:", sample["weather"])

print("Mask min:", sample["mask"].min())
print("Mask max:", sample["mask"].max())

print("Unique mask values:")
print(torch.unique(sample["mask"]))

print("Path:", sample["img_path"])

all_classes = set()

for i in range(len(dataset)):
    all_classes.update(
        torch.unique(dataset[i]["mask"]).tolist()
    )

print(sorted(all_classes))