"""
DIGITAL UNIVERSITY OF KERALA EXTRACTION MODEL (DUK-EM)
Unified segmentation model (SegFormer encoder + UPerNet/FPN decoder).

This model predicts all raster outputs directly:
- building_mask, roof_type_mask
- road_mask, road_centerline_mask
- waterbody_mask, waterbody_line_mask
- utility_line_mask

YOLO-based point detections (Wells, Transformers) are integrated at 
inference-time in `inference/predict.py`.
"""

import logging
from typing import Dict, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_encoder import DEFAULT_SEGFORMER_MODEL, SegformerEncoder
from .decoder import FPNDecoder
from .heads import create_all_heads

logger = logging.getLogger(__name__)


class EnsembleDUKModel(nn.Module):
    """
    Unified Production Architecture for DIGITAL UNIVERSITY OF KERALA EXTRACTION MODEL.

    Integrates:
    - SegFormer-B4 Backbone (Multi-scale Mix Transformer)
    - FPN Decoder with CBAM Attention
    - Specialized Task Heads (Building, Line, Detection)
    """

    def __init__(
        self,
        num_roof_classes: int = 5,
        pretrained: bool = True,
        model_name: str = DEFAULT_SEGFORMER_MODEL,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1. Backbone (SegFormer)
        self.encoder = SegformerEncoder(
            model_name=model_name, load_pretrained=pretrained, freeze=False
        )
        self.feature_channels = self.encoder.feature_channels

        # 2. Decoder (FPN + CBAM)
        in_ch_dict = {
            f"feat_s{i+1}": ch for i, ch in enumerate(self.feature_channels)
        }
        self.decoder = FPNDecoder(in_channels=in_ch_dict, out_channels=256)

        # 3. Heads (11 tasks)
        self.heads = create_all_heads(
            in_channels=256, num_roof_classes=num_roof_classes, dropout=dropout
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Zero-out task heads initially to stabilize multi-task training."""
        for name, m in self.heads.named_modules():
            if isinstance(m, nn.Conv2d) and "out" in name:
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, -4.0)  # low initial prob

    def forward(
        self, x: torch.Tensor, task: str = "all"
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the unified pipeline.

        Args:
            x: Input image tensor (B, 3, H, W)
            task: Task filter (e.g., "building", "road", or "all")
        """
        # Feature extraction
        backbone_feats_list = self.encoder(x)
        
        # Package into a dict mapping the exact names expected by the decoder
        backbone_feats = {
            f"feat_s{i+1}": backbone_feats_list[i]
            for i in range(len(backbone_feats_list))
        }

        # Multi-scale fusion
        fused_feat = self.decoder(backbone_feats)
        target_size = x.shape[2:]

        outputs = {}
        task_norm = task.lower().strip()
        run_all = task_norm in {"all", "*", "full"}

        # Buildings (Dual output)
        if run_all or task_norm in {"building", "buildings", "building_mask"}:
            mask, roof = self.heads["building"](fused_feat)
            outputs["building_mask"] = F.interpolate(
                mask, size=target_size, mode="bilinear", align_corners=False
            )
            outputs["roof_type_mask"] = F.interpolate(
                roof, size=target_size, mode="bilinear", align_corners=False
            )

        # Iterate through common binary/line tasks
        other_tasks = [
            "road", "road_centerline", "waterbody", "waterbody_line",
            "utility_line", "utility_poly"
        ]

        for t in other_tasks:
            if run_all or task_norm in {t, f"{t}_mask"}:
                logits = self.heads[t](fused_feat)
                outputs[f"{t}_mask"] = F.interpolate(
                    logits, size=target_size, mode="bilinear", align_corners=False
                )

        if not outputs:
            raise ValueError(f"Unknown task key: {task}")

        return outputs if run_all else outputs.get(f"{task_norm}_mask") or outputs

    def freeze_backbone(self):
        """Freeze SegFormer encoder for head-only training."""
        self.encoder.freeze()

    def unfreeze_backbone(self):
        """Unfreeze SegFormer encoder for full fine-tuning."""
        self.encoder.unfreeze()

    def get_param_groups(self, base_lr: float = 1e-4) -> list:
        """Categorize parameters for LR scaling."""
        backbone_params = list(self.encoder.parameters())
        head_params = list(self.decoder.parameters()) + list(self.heads.parameters())

        return [
            {"params": head_params, "lr": base_lr},
            {"params": backbone_params, "lr": base_lr * 0.1},  # Slower backbone LR
        ]


# Final rebranded Alias
DUKModel = EnsembleDUKModel
EnsembleSvamitvaModel = EnsembleDUKModel  # Backwards compatibility for legacy training scripts
