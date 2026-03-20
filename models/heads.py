"""
Task-specific prediction heads for 11 output tasks.

Each head takes the FPN feature map and produces a per-pixel prediction:
  - Binary heads: 1-channel sigmoid output
  - Roof-type head: 5-channel softmax output
"""

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Conv → BatchNorm → ReLU helper block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Binary Segmentation Head ─────────────────────────────────────────────────


class BinaryHead(nn.Module):
    """
    Generic binary segmentation head.
    Output: (B, 1, H, W) logits → sigmoid for probability.
    """

    def __init__(
        self,
        in_channels: int = 256,
        mid_channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.project = nn.Conv2d(in_channels, mid_channels, 1)
        self.refine = nn.Sequential(
            ConvBNReLU(in_channels, mid_channels),
            nn.Dropout2d(dropout),
            ConvBNReLU(mid_channels, mid_channels),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(mid_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.project(x)
        out = self.refine(x)
        return self.out(out + skip)  # residual


# ── Building Head (binary mask + roof-type classification) ───────────────────


class BuildingHead(nn.Module):
    """
    Dual-output head for buildings:
      - building_mask: (B, 1, H, W) binary segmentation
      - roof_type: (B, num_classes, H, W) per-pixel classification
    """

    def __init__(
        self,
        in_channels: int = 256,
        mid_channels: int = 128,
        num_roof_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            ConvBNReLU(in_channels, mid_channels),
            nn.Dropout2d(dropout),
            ConvBNReLU(mid_channels, mid_channels),
            nn.Conv2d(
                mid_channels, mid_channels, 3, padding=1, bias=False
            ),  # Refinement
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.mask_out = nn.Conv2d(mid_channels, 1, 1)
        self.roof_out = nn.Conv2d(mid_channels, num_roof_classes, 1)

    def forward(self, x: torch.Tensor):
        shared = self.shared(x)
        mask = self.mask_out(shared)
        roof = self.roof_out(shared)
        return mask, roof


# ── Line Feature Head (D-LinkNet Style) ──────────────────────────────────────


class DLinkBlock(nn.Module):
    """Dilated convolution block to expand receptive field for linear features."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.d1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, dilation=1, bias=False)
        self.d2 = nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False)
        self.d4 = nn.Conv2d(in_ch, out_ch, 3, padding=4, dilation=4, bias=False)
        self.d8 = nn.Conv2d(in_ch, out_ch, 3, padding=8, dilation=8, bias=False)
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.d1(x) + self.d2(x) + self.d4(x) + self.d8(x))


class LineHead(nn.Module):
    """
    Head optimized for thin linear features.
    Uses dilated convolutions for wider receptive field without resolution loss.
    """

    def __init__(
        self,
        in_channels: int = 256,
        mid_channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.project = nn.Conv2d(in_channels, mid_channels, 1)
        self.dlink = DLinkBlock(in_channels, mid_channels)
        self.refine = nn.Sequential(
            nn.Dropout2d(dropout),
            ConvBNReLU(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(mid_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.project(x)
        out = self.dlink(x)
        return self.out(self.refine(out) + skip)


# ── Detection Head (Detection-style features for sparse objects) ─────────────


class DetectionHead(nn.Module):
    """
    Head optimized for detection-like sparse objects (wells, transformers).
    Combines centripetal-shift style features with segmentation.
    """

    def __init__(
        self, in_channels: int = 256, mid_channels: int = 64, dropout: float = 0.1
    ):
        super().__init__()
        self.stem = ConvBNReLU(in_channels, mid_channels)
        # Multi-scale aggregation
        self.p3 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.p5 = nn.Conv2d(mid_channels, mid_channels, 5, padding=2)

        self.fuse = nn.Sequential(
            ConvBNReLU(mid_channels * 2, mid_channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        p3 = self.p3(x)
        p5 = self.p5(x)
        return self.fuse(torch.cat([p3, p5], dim=1))


# ── Point Feature Head (Legacy support) ──────────────────────────────────────


class PointHead(nn.Module):
    """
    Head optimized for sparse point-like features.
    Uses global context aggregation before predicting.
    """

    def __init__(
        self, in_channels: int = 256, mid_channels: int = 64, dropout: float = 0.1
    ):
        super().__init__()
        # Global context branch
        self.global_ctx = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.ReLU(inplace=True),
        )
        # Local detail branch
        self.local = ConvBNReLU(in_channels, mid_channels)
        # Fusion
        self.fuse = nn.Sequential(
            ConvBNReLU(mid_channels * 2, mid_channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(
                mid_channels, mid_channels, 3, padding=1, bias=False
            ),  # Refinement
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        g = self.global_ctx(x).expand(-1, -1, H, W)
        loc = self.local(x)
        return self.fuse(torch.cat([loc, g], dim=1))


# ── Head Registry ────────────────────────────────────────────────────────────


def create_all_heads(
    in_channels: int = 256,
    num_roof_classes: int = 5,
    dropout: float = 0.1,
) -> nn.ModuleDict:
    """
    Create all 11 task heads.

    Returns:
        ModuleDict with keys matching mask output names.
    """
    heads = nn.ModuleDict()

    # Building (dual output)
    heads["building"] = BuildingHead(in_channels, 128, num_roof_classes, dropout)

    # Polygon heads
    heads["road"] = BinaryHead(in_channels, 64, dropout)
    heads["waterbody"] = BinaryHead(in_channels, 64, dropout)
    heads["bridge"] = BinaryHead(in_channels, 64, dropout)

    # Line heads
    heads["road_centerline"] = LineHead(in_channels, 64, dropout)
    heads["waterbody_line"] = LineHead(in_channels, 64, dropout)
    heads["utility_line"] = LineHead(in_channels, 64, dropout)
    heads["railway"] = LineHead(in_channels, 64, dropout)

    # Point/Small Object heads (using DetectionHead for better IoU)
    heads["waterbody_point"] = DetectionHead(in_channels, 64, dropout)
    heads["utility_point"] = DetectionHead(in_channels, 64, dropout)

    return heads
