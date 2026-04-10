#!/usr/bin/env python3
"""
DGX-Optimized Training entry point for DUK-EM Ensemble (V1).
Automatically selects the GPU with most free memory and saves to check/best.pt.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SEGFORMER_MODEL = "nvidia/segformer-b4-finetuned-cityscapes-1024-1024"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-25s │ %(levelname)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dgx_train")


def parse_args():
    p = argparse.ArgumentParser(
        description="DGX Training: DUK-EM Ensemble Feature Extraction (V1)"
    )

    # Data
    p.add_argument(
        "--train_dirs",
        nargs="+",
        default=["data/MAP1"],
        help="Directories containing MAP*.tif + shapefiles",
    )
    p.add_argument("--val_dir", default=None, help="Separate validation directory")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--split_mode", default="map", choices=["map", "tile"])

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (8-16 per GPU recommended)",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tile_size", type=int, default=512)
    p.add_argument("--tile_overlap", type=int, default=96)
    p.add_argument("--num_workers", type=int, default=4)  # Halved to prevent OOM / CPU lock
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--freeze_epochs", type=int, default=5)
    p.add_argument("--one_epoch_only", action="store_true", help="Stop after 1 epoch")
    p.add_argument(
        "--model_name",
        default=DEFAULT_SEGFORMER_MODEL,
        help="HuggingFace model name for the SegFormer backbone",
    )

    # DGX Specifics
    p.add_argument("--checkpoint_dir", default="check", help="Requested 'check' dir")
    p.add_argument("--name", default="dgx_ensemble_v3", help="Experiment name")
    p.add_argument("--resume", action="store_true", help="Resume from latest.pt if found")
    p.add_argument("--checkpoint", default=None, help="Specific .pt file to resume from")
    p.add_argument(
        "--multi_gpu",
        action="store_true",
        default=True,
        help="Use DDP (primary) or DataParallel (fallback) on all GPUs. "
        "For max efficiency, launch with: "
        "torchrun --nproc_per_node=8 train_engine/train_segmentation.py",
    )

    p.add_argument("--quick_test", action="store_true", help="3-epoch smoke test")

    return p.parse_args()


def main():
    args = parse_args()

    # Ensure check directory exists
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Imports (deferred for faster CLI response)
    from data.dataset import create_dataloaders
    from models.losses import MultiTaskLoss
    from models.model import EnsembleDUKModel
    from train_engine.config import TrainingConfig, get_quick_test_config
    from train_engine.trainer import Trainer

    # Configuration
    if args.quick_test:
        config = get_quick_test_config()
        config.train_dirs = [Path(d) for d in args.train_dirs]
        config.checkpoint_dir = Path(args.checkpoint_dir)
        logger.info("⚡ Quick-test mode enabled")
    else:
        config = TrainingConfig(
            train_dirs=[Path(d) for d in args.train_dirs],
            val_dir=Path(args.val_dir) if args.val_dir else None,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            split_mode=args.split_mode,
            val_split=args.val_split,
            num_workers=args.num_workers,
            freeze_backbone_epochs=args.freeze_epochs,
            checkpoint_dir=Path(args.checkpoint_dir),
            experiment_name=args.name,
            dropout=args.dropout,
            sam2_checkpoint=None,
            mixed_precision=True,
            force_cpu=False,
            one_epoch_only=args.one_epoch_only,
            cache_features=False,
        )
        # Handle multi_gpu flag override if requested
        if not args.multi_gpu:
            logger.info("⚠️ Multi-GPU disabled by flag. Using single GPU.")
            # Just an example, we'd need a specific flag for single vs multi in
            # TrainingConfig if we wanted more control
            config.force_cpu = False

    import os
    import torch.distributed as dist

    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_distributed and not dist.is_initialized():
        from datetime import timedelta

        # Set device early to prevent NCCL mapping ambiguity
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=5),
        )

    # Data (Preprocessing happens here: tiling, normalization, etc.)
    logger.info(f"Loading datasets with tile_size={config.tile_size}...")
    train_loader, val_loader = create_dataloaders(
        train_dirs=config.train_dirs,
        val_dir=config.val_dir,
        image_size=config.tile_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
        split_mode=config.split_mode,
        distributed=is_distributed,
    )

    # Model
    logger.info("Building model...")
    model = EnsembleDUKModel(
        model_name=args.model_name,
        dropout=config.dropout,
    )

    # Loss
    loss_fn = MultiTaskLoss(
        num_roof_classes=config.num_roof_classes,
    )

    # Train
    trainer = Trainer(model, train_loader, val_loader, loss_fn, config)

    # Resume Logic (Refined for Robustness)
    if args.resume or args.checkpoint:
        success = False
        if args.checkpoint:
            res = trainer.load_checkpoint(args.checkpoint)
            success = res > 0
        else:
            # Look for checkpoints in order of preference
            for name in ["latest.pt", "best_latest.pt", "best.pt"]:
                potential = Path(args.checkpoint_dir) / name
                if potential.exists():
                    logger.info(f"🔍 Auto-detected checkpoint for resume: {name}")
                    try:
                        res = trainer.load_checkpoint(str(potential))
                        if res > 0:
                            success = True
                            break
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Corrupted or incompatible checkpoint detected ({name}): {e}. "
                            "Trying fallback..."
                        )

        if not success:
            if args.checkpoint:
                logger.critical(
                    f"❌ Failed to load specific checkpoint: {args.checkpoint}"
                )
            else:
                logger.warning(f"No usable checkpoints found in {args.checkpoint_dir}")
            logger.info("🚀 Starting training from scratch (Epoch 1)...")

    logger.info("Starting training on optimized GPU...")
    
    try:
        is_finished = trainer.fit()
        if is_finished:
            logger.info(f"🎉 Training finished. Checkpoints in: {args.checkpoint_dir}")
            sys.exit(0)
        else:
            logger.info("⏳ Epoch completed (one_epoch_only). Signaling continuation...")
            sys.exit(3)
    except Exception as e:
        logger.critical(f"💥 Training script failed with critical error: {e}")
        # If it crashed, we don't want to signal exit code 3 (continue)
        sys.exit(1)


if __name__ == "__main__":
    main()
