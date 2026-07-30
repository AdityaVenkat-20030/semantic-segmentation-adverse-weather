import torch

from dissertation.models.unet import UNet
from dissertation.configs.label_mapping import NUM_CLASSES

model = UNet(
    num_classes=NUM_CLASSES
)

x = torch.randn(
    2,
    3,
    512,
    1024,
)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)