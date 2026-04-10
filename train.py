#!/usr/bin/env python3
"""
🚀 DIGITAL UNIVERSITY OF KERALA EXTRACTION MODEL Unified Training Pipeline (Hackathon V1)

Developed by Students of Digital University Kerala (DUK)

This script orchestrates the training of both the SegFormer-based segmentation
model and the YOLOv8 point-detection model simultaneously.

Usage:
    python train.py --train_dirs data/MAP1 data/MAP2 --epochs 100
"""

import argparse
import logging
import subprocess
import sys
import signal
import shutil
import time
import random
import torch
import numpy as np
from pathlib import Path
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-25s │ %(levelname)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("unified_train")


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified Training for DIGITAL UNIVERSITY OF KERALA EXTRACTION MODEL"
    )

    # Shared Data Path
    p.add_argument(
        "--train_dirs",
        nargs="+",
        default=[],
        help="Directories containing MAP*.tif + shapefiles",
    )

    # Training Hyperparameters
    p.add_argument("--epochs", type=int, default=100, help="Epochs for both models")
    p.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for segmentation"
    )
    p.add_argument("--yolo_batch", type=int, default=16, help="Batch size for YOLO")

    # Optimization
    p.add_argument("--resume", action="store_true", help="Resume training from latest.pt")
    p.add_argument(
        "--multi_gpu",
        action="store_true",
        default=True,
        help="Use 8x GPUs if available",
    )
    p.add_argument(
        "--skip_seg",
        action="store_true",
        help="Skip segmentation training, only run YOLO prep/train",
    )

    # Hardware Allocation
    p.add_argument(
        "--yolo_device",
        type=str,
        default="4",
        help="Specific GPU index for YOLO (e.g. '4')",
    )

    return p.parse_args()


def run_step(cmd: List[str], step_name: str):
    logger.info(f"▶️ Starting Step: {step_name}")
    start_time = time.time()
    try:
        # We use check=True to raise an error if the step fails
        subprocess.run(cmd, check=True, capture_output=False)
        elapsed = (time.time() - start_time) / 60
        logger.info(f"✅ Step '{step_name}' completed in {elapsed:.2f} mins")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Step '{step_name}' failed with error: {e}")
        sys.exit(1)


def main():
    args = parse_args()
    # Global Seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    project_root = Path(__file__).resolve().parent

    logger.info("=" * 60)
    logger.info("🌟 DIGITAL UNIVERSITY OF KERALA EXTRACTION MODEL STARTING 🌟")
    logger.info(f"  Training Dirs: {args.train_dirs}")
    logger.info(f"  Total Epochs:  {args.epochs}")
    logger.info("=" * 60)

    # Signal handling for graceful shutdown
    def signal_handler(sig, frame):
        logger.warning(f"🛑 Received signal {sig}. Attempting graceful shutdown...")
        # Since we are using subprocess.run with check=True, it will raise
        # CalledProcessError if killed. We let the system exit naturally.
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Disk space check (Require at least 10GB for dataset prep and checkpoints)
    total, used, free = shutil.disk_usage(project_root)
    free_gb = free // (2**30)
    if free_gb < 10:
        logger.error(f"❌ Critical: Low disk space ({free_gb} GB free). "
                     "Need at least 10 GB for training. Free up space and try again.")
        sys.exit(1)
    logger.info(f"💾 Disk space check passed: {free_gb} GB free.")

    # Step 1: Prepare YOLO Point Dataset
    # This converts shapefile points into YOLO .txt labels + image tiles
    logger.info("\n[STAGE 1/3] Preparing YOLO Dataset...")
    yolo_prep_cmd = (
        [
            sys.executable,
            str(project_root / "scripts" / "prepare_yolo_dataset.py"),
            "--map_dirs",
        ]
        + args.train_dirs
        + [
            "--output",
            "yolo_dataset",
            "--tile_size",
            "1024",
            "--overlap",
            "0.25",
            "--oversample_factor",
            "4",
        ]
    )
    run_step(yolo_prep_cmd, "YOLO Prep")

    # Step 2: Train Segmentation & Roof Model (Ensemble V1)
    # This handles buildings, roads, water, and roof classifications
    logger.info(
        "\n[STAGE 2/3] Training Segmentation & Roof Model (Multi-GPU Elastic)..."
    )
    
    # Check if run_ddp.sh exists
    ddp_script = project_root / "run_ddp.sh"
    if ddp_script.exists():
        logger.info(f"Delegating to optimized multi-GPU script: {ddp_script}")
        # Note: run_ddp.sh uses hardcoded paths by default right now, so we'll just run it.
        # Alternatively, we pass the parsed dirs:
        cmd_args = [str(x) for x in args.train_dirs]
        seg_cmd = ["bash", str(ddp_script)] + cmd_args
    else:
        logger.info("Using standard single-instance training...")
        seg_cmd = [
            sys.executable,
            str(project_root / "train_engine" / "train_segmentation.py"),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--train_dirs",
        ] + args.train_dirs + [
            "--checkpoint_dir",
            "check",
            "--name",
            "segmentation_v1",
        ]
        if args.resume:
            seg_cmd.append("--resume")

    # Step 3: Define YOLO Point Detection Command (will be run in parallel)
    # This handles Transformers, Wells, and Overhead Tanks
    yolo_train_cmd = [
        sys.executable,
        str(project_root / "scripts" / "train_yolo.py"),
        "--data",
        "yolo_dataset/duk_points.yaml",
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.yolo_batch),
        "--project",
        "check/yolo_runs",
        "--device",
        args.yolo_device,
    ]

    processes = []
    check_dir = project_root / "check"
    check_dir.mkdir(exist_ok=True)

    # Launch Segmentation (Background)
    if not args.skip_seg:
        logger.info("🚀 Launching Stage 2: Segmentation DDP Training (Background)")
        p_seg = subprocess.Popen(seg_cmd)
        processes.append((p_seg, "Segmentation"))
    else:
        logger.info("⏩ Skipping Segmentation Training as requested.")

    # Launch YOLO (Background)
    logger.info("🚀 Launching Stage 3: YOLO Point Detection Training (Background)")
    p_yolo = subprocess.Popen(yolo_train_cmd)
    processes.append((p_yolo, "YOLO"))

    logger.info("\n" + "="*60)
    logger.info("🌟 BOTH MODELS ARE NOW TRAINING SIMULTANEOUSLY")
    logger.info(f"   - Segmentation: Using GPUs [0,1,2,3,5,6,7]")
    logger.info(f"   - YOLO:         Using GPU [{args.yolo_device}]")
    logger.info("="*60 + "\n")

    # Wait for both to complete
    for p, name in processes:
        p.wait()
        if p.returncode == 0:
            logger.info(f"✅ {name} Training completed successfully.")
        else:
            logger.error(f"❌ {name} Training failed with exit code {p.returncode}.")

    # Final Consolidation: Move weights to 'check/' for easy access
    logger.info("\n[FINAL] Consolidating Best Weights...")
    
    # 1. Best Segmentation (from run_ddp.sh or local)
    seg_candidates = [
        check_dir / "best.pt",
        check_dir / "segmentation_v1" / "best.pt",
    ]
    for cand in seg_candidates:
        if cand.exists():
            shutil.copy2(cand, check_dir / "segmentation_best.pt")
            logger.info(f"📍 Segmentation weights: {check_dir / 'segmentation_best.pt'}")
            break

    # 2. Best YOLO
    yolo_best_src = check_dir / "yolo_runs" / "duk_points" / "weights" / "best.pt"
    if yolo_best_src.exists():
        shutil.copy2(yolo_best_src, check_dir / "yolo_best.pt")
        logger.info(f"📍 YOLO weights: {check_dir / 'yolo_best.pt'}")

    logger.info("\n" + "=" * 60)
    logger.info("🏆 ALL MODELS TRAINED SUCCESSFULLY! 🏆")
    logger.info(f"  Checkpoints in: {check_dir.resolve()}")
    logger.info("  1. Segmentation: segmentation_best.pt")
    logger.info("  2. Point Detector: yolo_best.pt")
    logger.info("\nRun 'streamlit run app.py' to launch the dashboard.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
