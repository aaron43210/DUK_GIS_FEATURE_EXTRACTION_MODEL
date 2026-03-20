"""
Training loop for DUK-EM SAM2 model.

Features:
    - Mixed-precision training (AMP)
    - Cosine annealing with linear warmup
    - Gradient clipping
    - Staged backbone unfreezing
    - Early stopping on avg IoU
    - Top-K checkpoint saving
    - Per-epoch metric logging
"""

import contextlib
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
<<<<<<< HEAD

import numpy as np
=======
>>>>>>> 48371e9 (Final production push: SegFormer-B4 architecture, memory optimizations, and robust resume mechanism)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .config import TrainingConfig
from .metrics import MetricsTracker

logger = logging.getLogger(__name__)


# ── Utilities ─────────────────────────────────────────────────────────────────


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_targets(batch: dict, device: torch.device) -> dict:
    """Move all tensor targets to device, skip metadata fields."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def get_best_gpu() -> int:
    """Find the GPU index with the most free memory."""
    if not torch.cuda.is_available():
        return 0

    n_devices = torch.cuda.device_count()
    if n_devices <= 1:
        return 0

    best_idx = 0
    max_free = 0

    for i in range(n_devices):
        try:
            free, total = torch.cuda.mem_get_info(i)
            if free > max_free:
                max_free = free
                best_idx = i
        except Exception:
            continue

    return best_idx


def get_device(config: TrainingConfig) -> torch.device:
    """Determine the best available device."""
    if config.force_cpu:
        return torch.device("cpu")
    
    # Priority 1: LOCAL_RANK (for DDP stability)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        device = torch.device(f"cuda:{local_rank}")
        # Explicitly set the device to resolve NCCL mapping ambiguity
        torch.cuda.set_device(device)
        return device

    # Priority 2: Use GPU with most free memory
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        best_gpu = get_best_gpu()
        logger.info(f"Total CUDA GPUs detected: {n_gpus}")
        logger.info(f"Auto-selected primary GPU:{best_gpu} (most free memory)")
        return torch.device(f"cuda:{best_gpu}")
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Learning Rate Schedule ────────────────────────────────────────────────────


class WarmupCosineScheduler:
    """Linear warmup → cosine annealing LR schedule."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        lr_min: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(base_lr * scale, self.lr_min)


# ── Checkpoint Manager ────────────────────────────────────────────────────────


class CheckpointManager:
    """Saves top-K checkpoints and supports early stopping."""

    def __init__(
        self,
        save_dir: Path,
        save_top_k: int = 3,
        metric_name: str = "avg_iou",
        patience: int = 25,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.metric_name = metric_name
        self.patience = patience

        self.best_score = -float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.top_k: list = []  # (score, path) sorted ascending

    def save_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: Dict[str, Any],
        rank: int = 0,
    ):
        """Save a 'best_latest.pt' checkpoint for crash recovery."""
        if rank != 0:
            return

        # Handle DataParallel/DDP state_dict
        if hasattr(model, "module"):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        state = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "metrics": metrics,
        }
        path = self.save_dir / "latest.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, path)
            logger.debug(f"Saved latest checkpoint to {path}")
        except Exception as e:
            logger.warning(f"Failed to save latest checkpoint: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
        
        # Periodic memory refresh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: Dict[str, Any],
        rank: int = 0,
    ) -> bool:
        """
        Save model to best.pt if it's the new best.
        """
        score = metrics.get(self.metric_name, 0)
        is_best = score > self.best_score

        # Always save latest for crash recovery (only rank 0)
        self.save_latest(model, optimizer, scheduler, epoch, metrics, rank)

        if is_best:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            
            if rank != 0:
                return is_best

            # Handle DataParallel state_dict
            if hasattr(model, "module"):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()

            state = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "metrics": metrics,
                "best_score": self.best_score,
            }

            best_path = self.save_dir / "best.pt"
            best_tmp = best_path.with_suffix(".pt.tmp")
            try:
                torch.save(state, best_tmp)
                os.replace(best_tmp, best_path)
                logger.info(
                    f"★ New best model saved to best.pt "
                    f"(epoch {epoch}, {self.metric_name}={score:.4f})"
                )
            except Exception as e:
                logger.warning(f"Failed to save best checkpoint: {e}")
                if best_tmp.exists():
                    best_tmp.unlink()
        else:
            self.epochs_no_improve += 1

        return is_best

    @property
    def should_stop(self) -> bool:
        return self.epochs_no_improve >= self.patience


# ── Training Engine ───────────────────────────────────────────────────────────


class Trainer:
    """
    Full training engine for DIGITAL UNIVERSITY OF KERALA EXTRACTION MODEL.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, loss_fn, config)
        trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        config: TrainingConfig,
        start_epoch: int = 1,
    ):
        self.config = config
        self.start_epoch = start_epoch
        self.device = get_device(config)
        logger.info(f"Using device: {self.device}")

        self.was_interrupted = False

        # Handle Distributed or DataParallel
        raw_model = model.to(self.device)
        self.is_multi_gpu = False
        self._ddp_initialized_here = False  # Track if we init'd dist

        # Check if DDP was already initialized externally (e.g. torchrun)
        self.is_distributed = dist.is_available() and dist.is_initialized()

        # Auto-initialize DDP if multiple GPUs available and not already init'd
        if (
            not self.is_distributed
            and not config.force_cpu
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
        ):
            n_gpus = torch.cuda.device_count()
            try:
                # Set env vars for single-node DDP (auto-init)
                os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                os.environ.setdefault("MASTER_PORT", "29500")
                os.environ.setdefault("RANK", "0")
                os.environ.setdefault("WORLD_SIZE", "1")
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                # Set 5-minute timeout instead of default 30-minute to allow faster recovery
                from datetime import timedelta
                dist.init_process_group(
                    backend=backend, 
                    timeout=timedelta(minutes=5)
                )
                self.is_distributed = True
                self._ddp_initialized_here = True
                
                # Assign local_rank for this process
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                self.device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(self.device)

                logger.info(
                    "🚀 Auto-initialized DDP process group " "(backend=%s, %d GPUs, rank=%d)",
                    backend,
                    n_gpus,
                    local_rank
                )
            except Exception as ddp_init_err:
                logger.warning(
                    "⚠️ DDP auto-init failed: %s. " "Will try DataParallel fallback.",
                    ddp_init_err,
                )

        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1

        if self.is_distributed:
            # Use the local rank for device assignment
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
            raw_model = raw_model.to(self.device)

            # Convert BatchNorm → SyncBatchNorm for multi-GPU consistency
            raw_model = nn.SyncBatchNorm.convert_sync_batchnorm(raw_model)

            logger.info(
                "🚀 Activating DDP on rank %d/%d (Device: %s)",
                self.rank,
                self.world_size,
                self.device,
            )
            self.model = DDP(
                raw_model,
                device_ids=[self.device.index],
                find_unused_parameters=True,  # Necessary for frozen backbones
            )
            self.is_multi_gpu = True
        elif (
            torch.cuda.is_available()
            and torch.cuda.device_count() > 1
            and not config.force_cpu
        ):
            # Fallback: DataParallel (if DDP init failed)
            n_gpus = torch.cuda.device_count()
            try:
                self.device = torch.device("cuda:0")
                raw_model = raw_model.to(self.device)
                logger.info(
                    "🚀 DDP unavailable, using DataParallel fallback "
                    "on %d GPUs (Master: %s)",
                    n_gpus,
                    self.device,
                )
                self.model = nn.DataParallel(raw_model)
                self.is_multi_gpu = True
            except RuntimeError as dp_err:
                logger.warning(
                    "⚠️ DataParallel also failed: %s. " "Falling back to single GPU.",
                    dp_err,
                )
                self.model = raw_model
                self.is_multi_gpu = False
        else:
            self.model = raw_model
            if not config.force_cpu:
                n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                if n_gpus <= 1:
                    logger.info("Multi-GPU skipped: %d GPU(s) visible.", n_gpus)
            else:
                logger.info("Multi-GPU skipped: force_cpu is True.")

        # Wrap data loaders with DistributedSampler for DDP
        if self.is_distributed and not isinstance(
            getattr(train_loader, "sampler", None),
            DistributedSampler,
        ):
            ddp_sampler = DistributedSampler(
                train_loader.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            self.train_loader = DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                sampler=ddp_sampler,
                num_workers=getattr(train_loader, "num_workers", 0),
                pin_memory=True,
                drop_last=True,
            )
            logger.info(
                "Wrapped train DataLoader with " "DistributedSampler (rank=%d/%d)",
                self.rank,
                self.world_size,
            )
        else:
            self.train_loader = train_loader

        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(self.device)

        # Optimizer
        if hasattr(model, "get_param_groups"):
            param_groups = getattr(model, "get_param_groups")(config.learning_rate)
        else:
            param_groups = model.parameters()

        self.optimizer: torch.optim.Optimizer
        if config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.SGD(
                param_groups,
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay,
            )

        # Scheduler
        self.scheduler: Any
        if config.optimizer == "adamw":
            # Using OneCycleLR for faster convergence
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.learning_rate,
                epochs=config.num_epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=0.1,  # 10% warmup, 90% decay for "perfect" cooling
                div_factor=25,
                final_div_factor=1e4,
            )
        else:
            self.scheduler = WarmupCosineScheduler(
                self.optimizer,
                config.warmup_epochs,
                config.num_epochs,
                config.lr_min,
            )
        self._scheduler_step_per_batch = isinstance(
            self.scheduler, torch.optim.lr_scheduler.OneCycleLR
        )

        # AMP
        self.use_amp = config.mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler(self.device.type, enabled=self.use_amp)
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # Metrics
        self.metrics = MetricsTracker(num_roof_classes=config.num_roof_classes)

        # Checkpoint manager
        self.ckpt_mgr = CheckpointManager(
            Path(config.checkpoint_dir),
            config.save_top_k,
            config.metric_for_best,
            config.patience,
        )

        # Feature Caching (Removed for SegFormer)
        self.feature_cache_enabled = False

        # TensorBoard writer (only on rank 0)
        self.tb_writer = None
        if self.rank == 0:
            try:
                # Check if logging is explicitly disabled to save file handles
                if getattr(config, "enable_tensorboard", True):
                    from torch.utils.tensorboard import SummaryWriter

                    self.tb_writer = SummaryWriter(log_dir=str(config.log_dir))
                    logger.info("TensorBoard logging enabled.")
                else:
                    logger.info("TensorBoard logging disabled by config.")
            except (ImportError, OSError) as e:
                logger.warning(f"TensorBoard logging disabled: {e}")

        # History Persistence
        self.history_path = self.config.log_dir / "training_history.json"
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "metrics": [],
            "train_breakdown": [],
            "val_breakdown": [],
        }
        if self.history_path.exists():
            try:
                with open(self.history_path, "r") as f:
                    old_history = json.load(f)
                    for k in self.history:
                        if k in old_history and isinstance(old_history[k], list):
                            self.history[k] = old_history[k]
                logger.info(f"📜 Loaded training history ({len(self.history['train_loss'])} entries)")
            except Exception as e:
                logger.warning(f"Could not load history: {e}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """
        Loads model, optimizer, and scheduler state from a checkpoint.
        Returns the next epoch to start from.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(f"Checkpoint not found at {path}. Starting from scratch.")
            return 1

        logger.info(f"📂 Loading checkpoint: {path}")
        # Map location to current device to avoid CUDA/CPU mismatches
        try:
            checkpoint = torch.load(path, map_location=self.device)
        except (EOFError, RuntimeError, Exception) as e:
            logger.warning(f"❌ Failed to load checkpoint {path}: {e}")
            # Identify specific corruption types for better logging
            if isinstance(e, EOFError):
                logger.error("Checkpoint file is truncated (EOFError).")
            return 0  # Signal failure so caller can try fallback

        # 1. Model State
        try:
            if hasattr(self.model, "module"):
                self.model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as e:
            logger.error(f"❌ State dict mismatch in {path}: {e}")
            return 0

        # ... rest of the logic ...
        # 2. Optimizer State
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}")

        # 3. Scheduler State
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                logger.warning(f"Could not load scheduler state: {e}")

        # 4. Starting Epoch
        # If this is 'latest.pt', it might be a mid-epoch save.
        # We check a custom flag 'epoch_completed' to decide if we increment.
        is_completed = checkpoint.get("metrics", {}).get("epoch_completed", True)
        epoch_val = checkpoint.get("epoch", 0)
        
        if is_completed:
            start_epoch = epoch_val + 1
            logger.info(f"✅ Finished epoch {epoch_val}. Resuming from epoch {start_epoch}")
        else:
            start_epoch = epoch_val
            logger.info(f"⚠️ Resuming unfinished/crashed epoch {start_epoch}")

        self.start_epoch = start_epoch
        return start_epoch

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """
        Loads model, optimizer, and scheduler state from a checkpoint.
        Returns the next epoch to start from.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(f"Checkpoint not found at {path}. Starting from scratch.")
            return 1

        logger.info(f"📂 Loading checkpoint: {path}")
        # Map location to current device to avoid CUDA/CPU mismatches
        checkpoint = torch.load(path, map_location=self.device)

        # 1. Model State
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # 2. Optimizer State
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 3. Scheduler State
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                logger.warning(f"Could not load scheduler state: {e}")

        # 4. Starting Epoch
        start_epoch = checkpoint.get("epoch", 0) + 1
        self.start_epoch = start_epoch

        logger.info(f"✅ Successfully resumed from epoch {start_epoch}")
        return start_epoch

    def fit(self):
        """Run full training loop with detailed logging and TensorBoard visualization."""
        set_seed(self.config.seed)
        self.was_interrupted = False
        logger.info(
            f"Starting training for {self.config.num_epochs} epochs "
            f"({len(self.train_loader)} train batches, "
            f"{len(self.val_loader)} val batches)"
        )
        completed_successfully = False

        try:
            for epoch in range(self.start_epoch, self.config.num_epochs + 1):
                # For DDP, we need to set the epoch on the sampler for correct shuffling
                if self.is_distributed and hasattr(self.train_loader, "sampler"):
                    if hasattr(self.train_loader.sampler, "set_epoch"):
                        self.train_loader.sampler.set_epoch(epoch)

                # ... unfreezing logic ...
                model_to_unfreeze = (
                    self.model.module if self.is_multi_gpu else self.model
                )
                if epoch == self.config.freeze_backbone_epochs + 1:
                    logger.info(f"Epoch {epoch}: Unfreezing backbone")
                    model_to_unfreeze.unfreeze_backbone()

                if epoch <= self.config.freeze_backbone_epochs:
                    model_to_unfreeze.freeze_backbone()

                # Train
                train_loss, train_breakdown = self._train_epoch(epoch)
                self.history["train_loss"].append(train_loss)
                self.history["train_breakdown"].append(train_breakdown)

                # Validate
                val_loss, val_metrics = self._validate_epoch(epoch)
                self.history["val_loss"].append(val_loss)
                self.history["metrics"].append(val_metrics)
                self.history["val_breakdown"].append(
                    {k: val_metrics.get(k, 0) for k in train_breakdown.keys()}
                )

                # LR schedule
                if not self._scheduler_step_per_batch:
                    if isinstance(self.scheduler, WarmupCosineScheduler):
                        # WarmupCosineScheduler expects a zero-based epoch index.
                        self.scheduler.step(epoch - 1)
                    else:
                        self.scheduler.step()
                current_lr = self.optimizer.param_groups[-1]["lr"]

                # Log
                if self.rank == 0:
                    avg_iou = val_metrics.get("avg_iou", 0)
                    avg_dice = val_metrics.get("avg_dice", 0)
                    logger.info(
                        f"Epoch {epoch}/{self.config.num_epochs} │ "
                        f"Train Loss: {train_loss:.4f} │ Val Loss: {val_loss:.4f} │ "
                        f"Avg IoU: {avg_iou:.4f} │ Avg Dice: {avg_dice:.4f} │ "
                        f"LR: {current_lr:.2e}"
                    )

                # Save checkpoint (includes best_latest.pt save)
                val_metrics["epoch_completed"] = True  # Mark as finished
                self.ckpt_mgr.save(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_metrics,
                    self.rank,
                )

                # Memory refresh at end of epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Early stopping
                if self.config.early_stopping and self.ckpt_mgr.should_stop:
                    logger.info(
                        f"Early stopping after {self.config.patience} epochs "
                        f"without improvement. Best epoch: {self.ckpt_mgr.best_epoch}"
                    )
                    break
            completed_successfully = True

        except (KeyboardInterrupt, Exception) as e:
            logger.error(f"Training interrupted or crashed: {e}")
            logger.info("Saving emergency checkpoint to best_latest.pt...")
            # Emergency save using current state (only rank 0)
            current_epoch = epoch if "epoch" in locals() else 0
            metrics = {"epoch_interrupted": True}
            self.ckpt_mgr.save_latest(
                self.model,
                self.optimizer,
                self.scheduler,
                current_epoch,
                metrics,
                self.rank,
            )
            if isinstance(e, KeyboardInterrupt):
                self.was_interrupted = True
                logger.info("Keyboard interrupt received. Exiting.")
            else:
                logger.warning("Crash handled by TrainEngine emergency checkpoint.")
        finally:
            # Final save if we completed all epochs successfully
            if (
                completed_successfully
                and "epoch" in locals()
                and epoch == self.config.num_epochs
            ):
                # Do not overwrite ckpt manager's best.pt with the last epoch.
                # Save final epoch weights separately for reproducibility/debug.
                final_path = self.ckpt_mgr.save_dir / "final_last_epoch.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": (
                            self.model.module.state_dict()
                            if self.is_multi_gpu
                            else self.model.state_dict()
                        ),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": (
                            self.scheduler.state_dict()
                            if self.scheduler is not None
                            else None
                        ),
                    },
                    final_path,
                )
                logger.info(
                    "Training completed successfully. "
                    f"Last-epoch checkpoint saved to {final_path} "
                    "(best.pt preserved from best validation epoch)."
                )

        # Save training history
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        logger.info(f"Training history saved to {self.history_path}")

        if self.tb_writer:
            self.tb_writer.close()
            logger.info("TensorBoard logs written.")

        logger.info(
            f"Training complete. Best {self.config.metric_for_best}: "
            f"{self.ckpt_mgr.best_score:.4f} at epoch "
            f"{self.ckpt_mgr.best_epoch}"
        )

        # Cleanup DDP process group if we initialized it
        if self._ddp_initialized_here and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("DDP process group destroyed.")

    def _train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        breakdown_sums: Dict[str, float] = {}
        n_batches = 0

        # For DDP, only show progress bar on rank 0
        train_iter: Any = self.train_loader
        if self.rank == 0:
            train_iter = tqdm(
                self.train_loader,
                desc=f"Train Epoch {epoch}",
                leave=False,
                dynamic_ncols=True,
            )

        accum_steps = getattr(self.config, "gradient_accumulation_steps", 1)
        
        for i, batch in enumerate(train_iter):
            batch = move_targets(batch, self.device)
            images = batch["image"]

            # Use dist.no_sync() to avoid unnecessary gradient synchronization during accumulation
            context = (
                self.model.no_sync() 
                if self.is_distributed and (i + 1) % accum_steps != 0 
                else contextlib.nullcontext()
            )

            with context:
                with torch.autocast(
                    device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
                ):
                    predictions = self.model(images)
                    loss, breakdown = self.loss_fn(predictions, batch)
                    
                    # Normalize loss for accumulation
                    loss = loss / accum_steps

                # Skip NaN/Inf loss batches
                if torch.isnan(loss) or torch.isinf(loss):
                    if self.rank == 0:
                        train_iter.set_postfix(loss="NaN-skip")
                    continue

                # Backward
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Optimizer Step (only every accum_steps)
            if (i + 1) % accum_steps == 0 or (i + 1) == len(self.train_loader):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)

                if self._scheduler_step_per_batch:
                    self.scheduler.step()

            # Metrics and Logging
            total_loss += loss.item() * accum_steps # Denormalize for logging
            n_batches += 1

            for k, v in breakdown.items():
                breakdown_sums[k] = breakdown_sums.get(k, 0) + v

            if self.rank == 0:
                train_iter.set_postfix(loss=f"{loss.item():.4f}")
                
                # Periodic logging for visibility in non-interactive terminals
                if n_batches % 20 == 0:
                    logger.info(
                        f"Epoch {epoch} │ Batch {n_batches}/{len(self.train_loader)} │ "
                        f"Loss: {loss.item():.4f}"
                    )

                # Save intra-epoch checkpoint every 500 batches
                if n_batches % 500 == 0:
                    try:
                        metrics = {
                            "epoch_interrupted": 1.0, 
                            "mid_epoch_batch": float(n_batches)
                        }
                        self.ckpt_mgr.save_latest(
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            epoch,
                            metrics,
                            self.rank,
                        )
                        logger.debug(f"Saved intra-epoch checkpoint at batch {n_batches}.")
                    except Exception as e:
                        logger.warning(f"Failed to save intra-epoch checkpoint: {e}")

        if self.is_distributed:
            dist.barrier()

        avg_loss = total_loss / max(n_batches, 1)
        avg_breakdown = {k: v / max(n_batches, 1) for k, v in breakdown_sums.items()}
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return avg_loss, avg_breakdown

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Run validation on all ranks and synchronize metrics."""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        n_batches = 0

        # Optional: only show progress bar on Rank 0
        disable_tqdm = (self.rank != 0)
        val_iter: Any = tqdm(
            self.val_loader,
            desc=f"Val Epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
            disable=disable_tqdm,
        )

        for batch in val_iter:
            # batch = move_targets(batch, self.device) # move_targets is defined in utils or somewhere? 
            # Wait, I should check where move_targets is. It's likely in this file or imported.
            # Looking at previous view_file, it was used on line 790.
            batch = move_targets(batch, self.device)
            images = batch["image"]

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                predictions = self.model(images)
                loss, _ = self.loss_fn(predictions, batch)

            total_loss += loss.item()
            n_batches += 1

            self.metrics.update(predictions, batch)

        # Synchronize metrics across all ranks
        if self.is_distributed:
            self.metrics.sync()
            
            # Sync validation loss
            dist_data = torch.tensor(
                [total_loss, float(n_batches)],
                dtype=torch.float64,
                device=self.device
            )
            torch.distributed.all_reduce(dist_data, op=torch.distributed.ReduceOp.SUM)
            global_total_loss, global_n_batches = dist_data.tolist()
            avg_loss = global_total_loss / max(global_n_batches, 1.0)
        else:
            avg_loss = total_loss / max(n_batches, 1)

        metrics = self.metrics.compute()
        metrics["val_loss"] = avg_loss
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return avg_loss, metrics
