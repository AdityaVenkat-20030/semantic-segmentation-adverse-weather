"""
models/segformer.py

SegFormer semantic segmentation model wrapper.

This file wraps HuggingFace SegFormer so that it behaves like the
other models in this codebase, namely U-Net and DeepLabV3+.

Backbone: MiT Transformer encoder
Decoder: Lightweight MLP segmentation head

Reference:
Xie et al., "SegFormer: Simple and Efficient Design for Semantic
Segmentation with Transformers", NeurIPS 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from dissertation.configs.label_mapping import NUM_CLASSES


class SegFormer(nn.Module):
    """
    SegFormer wrapper with output interface matching U-Net and DeepLabV3+.

    Args:
        num_classes:
            Number of output segmentation classes.

        pretrained:
            Whether to load pretrained HuggingFace SegFormer weights.

        backbone:
            Backbone variant. Supported values:
            "mit_b0", "mit_b2", "mit_b5", "b0", "b2", "b5".

        pretrained_checkpoint:
            Optional HuggingFace checkpoint name or local path.
            If provided, this overrides the default checkpoint selected
            from the backbone.
    """

    PRETRAINED_CHECKPOINTS = {
        "b0": "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
        "b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    }

    BACKBONE_CONFIGS = {
        "b0": {
            "depths": [2, 2, 2, 2],
            "hidden_sizes": [32, 64, 160, 256],
            "num_attention_heads": [1, 2, 5, 8],
            "decoder_hidden_size": 256,
        },
        "b2": {
            "depths": [3, 4, 6, 3],
            "hidden_sizes": [64, 128, 320, 512],
            "num_attention_heads": [1, 2, 5, 8],
            "decoder_hidden_size": 768,
        },
        "b5": {
            "depths": [3, 6, 40, 3],
            "hidden_sizes": [64, 128, 320, 512],
            "num_attention_heads": [1, 2, 5, 8],
            "decoder_hidden_size": 768,
        },
    }

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        backbone: str = "mit_b0",
        pretrained_checkpoint: str | None = None,
    ):
        super().__init__()

        backbone_key = self._normalize_backbone_name(backbone)

        self.name = f"SegFormer-{backbone_key.upper()}"
        self.num_classes = num_classes
        self.backbone = backbone_key

        if pretrained:
            checkpoint = (
                pretrained_checkpoint
                if pretrained_checkpoint is not None
                else self.PRETRAINED_CHECKPOINTS[backbone_key]
            )

            self.model = SegformerForSemanticSegmentation.from_pretrained(
                checkpoint,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )

        else:
            config = self._build_config_from_scratch(
                num_classes=num_classes,
                backbone_key=backbone_key,
            )

            self.model = SegformerForSemanticSegmentation(config)

    @staticmethod
    def _normalize_backbone_name(backbone: str) -> str:
        """
        Convert config-friendly names like 'mit_b0' to 'b0'.
        """

        backbone = backbone.lower().strip()

        if backbone.startswith("mit_"):
            backbone = backbone.replace("mit_", "", 1)

        if backbone not in ["b0", "b2", "b5"]:
            raise ValueError(
                "Unsupported SegFormer backbone. "
                "Use one of: mit_b0, mit_b2, mit_b5, b0, b2, b5."
            )

        return backbone

    def _build_config_from_scratch(
        self,
        num_classes: int,
        backbone_key: str,
    ) -> SegformerConfig:
        """
        Build SegFormer configuration without pretrained weights.
        """

        backbone_config = self.BACKBONE_CONFIGS[backbone_key]

        return SegformerConfig(
            num_labels=num_classes,
            num_channels=3,
            num_encoder_blocks=4,
            depths=backbone_config["depths"],
            hidden_sizes=backbone_config["hidden_sizes"],
            num_attention_heads=backbone_config["num_attention_heads"],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            mlp_ratios=[4, 4, 4, 4],
            hidden_act="gelu",
            decoder_hidden_size=backbone_config["decoder_hidden_size"],
            classifier_dropout_prob=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Input image tensor of shape [B, 3, H, W].

        Returns:
            logits:
                Segmentation logits of shape [B, num_classes, H, W].
        """

        input_size = x.shape[2:]

        outputs = self.model(
            pixel_values=x
        )

        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits

    def get_1x_lr_params(self):
        """
        Encoder parameters.
        """

        return list(
            self.model.segformer.parameters()
        )

    def get_10x_lr_params(self):
        """
        Decoder/head parameters.
        """

        return list(
            self.model.decode_head.parameters()
        )

    def count_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )


def build_segformer(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    backbone: str = "mit_b0",
    pretrained_checkpoint: str | None = None,
) -> SegFormer:
    return SegFormer(
        num_classes=num_classes,
        pretrained=pretrained,
        backbone=backbone,
        pretrained_checkpoint=pretrained_checkpoint,
    )


if __name__ == "__main__":
    model = build_segformer(
        num_classes=NUM_CLASSES,
        pretrained=False,
        backbone="mit_b0",
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