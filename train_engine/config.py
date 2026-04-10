"""
Training configuration dataclass for DUK-EM Ensemble SOTA Architecture (V1).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_tiny.yaml"


@dataclass
class TrainingConfig:
    """All hyperparameters and paths for ensemble V1 training."""

    # ── Paths ────────────────────────────────────────────────────────────────
    train_dirs: List[Path] = field(default_factory=lambda: [Path("data/MAP1")])
    val_dir: Optional[Path] = None
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    project_root: Path = field(default_factory=lambda: Path.cwd())

    # ── Model ────────────────────────────────────────────────────────────────
    backbone: str = "ensemble"
    pretrained: bool = True
    num_roof_classes: int = 5
    dropout: float = 0.1
    sam2_checkpoint: Optional[Path] = Path("checkpoints/sam2.1_hiera_tiny.pt")
    sam2_model_cfg: str = DEFAULT_SAM2_MODEL_CFG

    # ── Training ─────────────────────────────────────────────────────────────
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1.5e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # adamw | sgd
    scheduler: str = "cosine"  # cosine | step | plateau
    lr_min: float = 1e-6
    warmup_epochs: int = 5
    gradient_clip: float = 0.5
    mixed_precision: bool = True
    freeze_backbone_epochs: int = 5
    gradient_accumulation_steps: int = 1
    seed: int = 42
    force_cpu: bool = False
    one_epoch_only: bool = False

    # ── Feature Caching ──
    cache_features: bool = False

    # ── Data ─────────────────────────────────────────────────────────────────
    tile_size: int = 512
    tile_overlap: int = 96
    split_mode: str = "map"  # map | tile
    num_workers: int = 4
    pin_memory: bool = True
    val_split: float = 0.2

    # ── Loss Weights ─────────────────────────────────────────────────────────
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "building": 1.2,
            "road": 1.1,
            "road_centerline": 0.5,
            "waterbody": 1.0,
            "waterbody_line": 0.5,
            "utility_line": 0.4,
            "roof_type": 0.7,
        }
    )

    # ── Checkpointing ────────────────────────────────────────────────────────
    save_top_k: int = 3
    metric_for_best: str = "avg_iou"
    early_stopping: bool = True
    patience: int = 25
    eval_every_n_epochs: int = 1

    # ── Logging ──────────────────────────────────────────────────────────────
    log_every_n_steps: int = 50
    use_wandb: bool = False
    wandb_project: str = "duk-extraction-model-v1"
    experiment_name: str = "Vidhata-AI"
    enable_tensorboard: bool = True

    def __post_init__(self):
        # Ensure Paths are correctly typed
        if isinstance(self.train_dirs, list):
            self.train_dirs = [
                Path(d) if not isinstance(d, Path) else d for d in self.train_dirs
            ]
        if self.val_dir and not isinstance(self.val_dir, Path):
            self.val_dir = Path(self.val_dir)
        if not isinstance(self.checkpoint_dir, Path):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if not isinstance(self.log_dir, Path):
            self.log_dir = Path(self.log_dir)
        if self.sam2_checkpoint is not None and not isinstance(
            self.sam2_checkpoint, Path
        ):
            self.sam2_checkpoint = Path(self.sam2_checkpoint)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


def get_quick_test_config() -> TrainingConfig:
    """Small config for quick smoke-testing."""
    return TrainingConfig(
        batch_size=2,
        num_epochs=3,
        tile_size=256,
        num_workers=0,
        mixed_precision=False,
        early_stopping=False,
    )


def get_full_training_config() -> TrainingConfig:
    """Full production training config."""
    return TrainingConfig(
        batch_size=8,
        num_epochs=100,
        tile_size=512,
        mixed_precision=True,
        early_stopping=True,
        patience=25,
    )
