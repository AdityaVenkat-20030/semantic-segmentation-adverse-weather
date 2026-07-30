from dissertation.dataloaders.build_dataloaders import (
    build_idd_dataloaders,
)

train_loader, val_loader = build_idd_dataloaders(
    batch_size=4
)

batch = next(iter(train_loader))

print(batch["image"].shape)
print(batch["mask"].shape)