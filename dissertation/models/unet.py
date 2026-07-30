"""
models/unet.py
U-Net semantic segmentation model.

Encoder: ResNet-50 ImageNet pretrained backbone

Decoder: Progressive upsampling with skip connections

Datasets: IDD, IDD20KII, IDDAW
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from dissertation.configs.label_mapping import NUM_CLASSES

class DoubleConv(nn.Module):
    """
    Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(
                nn.Dropout2d(dropout)
            )

        self.block = nn.Sequential(*layers)

    def forward(self,x):

        return self.block(x)



class DecoderBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels, dropout=0.1)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([skip,x],dim=1)
        return self.conv(x)



class UNet(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):

        super().__init__()
        self.name = "UNet-ResNet50"
        backbone = models.resnet50(
            weights=(
                models.ResNet50_Weights.IMAGENET1K_V2
                if pretrained
                else None
            )
        )

        # Encoder
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        # Decoder
        self.dec4 = DecoderBlock(2048, 1024, 512)

        self.dec3 = DecoderBlock(512, 512, 256)

        self.dec2 = DecoderBlock(256, 256, 128)

        self.dec1 = DecoderBlock(128, 64, 64)

        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2,)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

        self.dropout = nn.Dropout2d(0.2)

    def forward(self,x):

        input_size = x.shape[2:]

        # Encoder
        s0 = self.enc0(x)
        p = self.pool(s0)
        s1 = self.enc1(p)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        s4 = self.dropout(s4)

        # Decoder
        x = self.dec4(s4, s3)

        x = self.dec3(x, s2)

        x = self.dec2(x, s1)

        x = self.dec1(x, s0)

        x = self.final_up(x)

        x = self.classifier(x)

        # safety
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        return x

    def get_encoder_params(self):
        return list(self.enc0.parameters()) + \
               list(self.enc1.parameters()) + \
               list(self.enc2.parameters()) + \
               list(self.enc3.parameters()) + \
               list(self.enc4.parameters())

    def get_decoder_params(self):
        return list(self.dec1.parameters()) + \
               list(self.dec2.parameters()) + \
               list(self.dec3.parameters()) + \
               list(self.dec4.parameters()) + \
               list(self.final_up.parameters()) + \
               list(self.classifier.parameters())

    def count_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
                if p.requires_grad
        )

def build_unet(
    num_classes=NUM_CLASSES,
    pretrained=True,
):

    return UNet(
        num_classes=num_classes,
        pretrained=pretrained,
    )

if __name__ == "__main__":

    model = build_unet()
    x = torch.randn(2, 3, 512, 1024)
    y = model(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
    print(f"Parameters: {model.count_parameters()/1e6:.2f}M")