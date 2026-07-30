"""
models/deeplabv3plus.py

DeepLabV3+ semantic segmentation model.

Backbone: ResNet-50 with dilated convolutions
Main module: Atrous Spatial Pyramid Pooling (ASPP)
Decoder: Low-level feature refinement + upsampling

References:
Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation", 2017.
Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic
Image Segmentation" DeepLabV3+, ECCV 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from dissertation.configs.label_mapping import NUM_CLASSES


class ASPPConv(nn.Module):
    """
    Atrous convolution branch used inside ASPP.
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPPPooling(nn.Module):
    """
    Global average pooling branch used inside ASPP.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]

        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

        return x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    It captures multi-scale context using:
    1. One 1x1 convolution branch
    2. Three atrous convolution branches with different dilation rates
    3. One global pooling branch
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        dilations=(6, 12, 18),
    ):
        super().__init__()

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
                ASPPConv(
                    in_channels,
                    out_channels,
                    dilations[0],
                ),
                ASPPConv(
                    in_channels,
                    out_channels,
                    dilations[1],
                ),
                ASPPConv(
                    in_channels,
                    out_channels,
                    dilations[2],
                ),
                ASPPPooling(
                    in_channels,
                    out_channels,
                ),
            ]
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels * 5,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        features = [
            branch(x)
            for branch in self.branches
        ]

        x = torch.cat(
            features,
            dim=1,
        )

        return self.project(x)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with ResNet-50 backbone.

    Args:
        num_classes:
            Number of semantic segmentation classes.

        pretrained:
            Whether to use ImageNet-pretrained ResNet-50 weights.

        output_stride:
            Controls output feature-map resolution.
            16 is faster and uses less memory.
            8 keeps more spatial detail but uses more memory.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        output_stride: int = 16,
    ):
        super().__init__()

        assert output_stride in (8, 16)

        self.name = "DeepLabV3Plus-ResNet50"
        self.num_classes = num_classes
        self.output_stride = output_stride

        backbone = models.resnet50(
            weights=(
                models.ResNet50_Weights.IMAGENET1K_V2
                if pretrained
                else None
            )
        )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        if output_stride == 16:
            self._set_dilation(
                backbone.layer4,
                dilation=2,
                stride=1,
            )
        else:
            self._set_dilation(
                backbone.layer3,
                dilation=2,
                stride=1,
            )
            self._set_dilation(
                backbone.layer4,
                dilation=4,
                stride=1,
            )

        self.layer4 = backbone.layer4

        aspp_dilations = (
            (6, 12, 18)
            if output_stride == 16
            else (12, 24, 36)
        )

        self.aspp = ASPP(
            in_channels=2048,
            out_channels=256,
            dilations=aspp_dilations,
        )

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(
                256,
                48,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                256 + 48,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(
            256,
            num_classes,
            kernel_size=1,
        )

    @staticmethod
    def _set_dilation(layer, dilation: int, stride: int):
        """
        Replace stride with dilation in ResNet layers.

        This increases the feature-map resolution while preserving
        a larger receptive field.
        """

        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                if module.kernel_size == (3, 3):
                    module.dilation = (
                        dilation,
                        dilation,
                    )
                    module.padding = (
                        dilation,
                        dilation,
                    )

                if module.stride == (2, 2):
                    module.stride = (
                        stride,
                        stride,
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        x = self.stem(x)

        low_level = self.layer1(x)

        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)

        x = F.interpolate(
            x,
            size=low_level.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        low_level = self.low_level_proj(low_level)

        x = torch.cat(
            [
                x,
                low_level,
            ],
            dim=1,
        )

        x = self.decoder(x)

        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        x = self.head(x)

        return x

    def get_1x_lr_params(self):
        return (
            list(self.stem.parameters())
            + list(self.layer1.parameters())
            + list(self.layer2.parameters())
            + list(self.layer3.parameters())
            + list(self.layer4.parameters())
        )

    def get_10x_lr_params(self):
        return (
            list(self.aspp.parameters())
            + list(self.low_level_proj.parameters())
            + list(self.decoder.parameters())
            + list(self.head.parameters())
        )

    def count_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )


def build_deeplabv3plus(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    output_stride: int = 16,
) -> DeepLabV3Plus:
    return DeepLabV3Plus(
        num_classes=num_classes,
        pretrained=pretrained,
        output_stride=output_stride,
    )


if __name__ == "__main__":
    model = build_deeplabv3plus(
        num_classes=NUM_CLASSES,
        pretrained=False,
        output_stride=16,
    )

    x = torch.randn(
        2,
        3,
        512,
        1024,
    )

    y = model(x)

    print("Input: ", x.shape)
    print("Output:", y.shape)
    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_parameters() / 1e6:.2f}M")