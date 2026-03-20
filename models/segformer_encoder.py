"""
SegFormer (Mix Transformer) Backbone Encoder

Replaces SAM2 with a highly-optimized Vision Transformer architectures designed
explicitly for semantic segmentation (natively extracting 4-stage feature maps).
Uses HuggingFace integrated `transformers` library.
"""

import logging
from typing import List

import torch
import torch.nn as nn

try:
    from transformers import SegformerModel, SegformerConfig
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")

logger = logging.getLogger(__name__)

# SegFormer B4 is heavily optimized for high-res (like 1024x1024) and achieves SOTA on Cityscapes
DEFAULT_SEGFORMER_MODEL = "nvidia/segformer-b4-finetuned-cityscapes-1024-1024"


class SegformerEncoder(nn.Module):
    """
    Wraps the HuggingFace SegFormer model to act as a drop-in replacement
    for our previous SAM2 encoder.

    Extracts 4 hierarchical feature maps equivalent to FPN levels.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SEGFORMER_MODEL,
        freeze: bool = False,
        load_pretrained: bool = True,
    ):
        super().__init__()

        logger.info(f"Initializing SegFormer Backbone: {model_name}")

        if load_pretrained:
            # We load the bare model (no classification head) to just get the multi-scale hidden states
            self.backbone = SegformerModel.from_pretrained(model_name)
        else:
            config = SegformerConfig.from_pretrained(model_name)
            self.backbone = SegformerModel(config)

        # Segformer B4 outputs these specific channel dimensions at the 4 stages
        self.feature_channels = [64, 128, 320, 512]

        if freeze:
            self.freeze()

    def freeze(self):
        """Freezes the backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("SegFormer backbone frozen")

    def unfreeze(self):
        """Unfreezes the backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("SegFormer backbone unfrozen")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input tensor (B, 3, H, W). Normalized to [0, 1] or ImageNet Z-score
               in the dataloader/inference script.

        Returns:
            List of 4 feature maps at different spatial resolutions (H/4, H/8, H/16, H/32)
        """
        # The transformers model outputs a dataclass, we want the `hidden_states` tuple
        outputs = self.backbone(
            pixel_values=x, output_hidden_states=True, return_dict=True
        )

        # SegFormer outputs exactly 4 hidden states corresponding to the 4 Mix Transformer blocks.
        # However, some versions include the initial embedding output as the first entry (total 5).
        hidden_states = list(outputs.hidden_states)

        if len(hidden_states) == 5:
            # Drop the initial embedding, keep the 4 hierarchical stage outputs
            hidden_states = hidden_states[1:]

        if len(hidden_states) != 4:
            raise RuntimeError(
                f"Expected 4 hidden states from SegFormer blocks, got {len(hidden_states)}"
            )

        return hidden_states
