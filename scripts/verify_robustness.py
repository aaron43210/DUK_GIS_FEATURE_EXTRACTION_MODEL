import os
import torch
import torch.nn as nn
from pathlib import Path
import logging
import sys

# Ensure REPO_ROOT is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_engine.trainer import CheckpointManager, Trainer
from train_engine.config import TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-15s │ %(levelname)s │ %(message)s"
)
logger = logging.getLogger("verify_dgx")

def test_atomic_save():
    save_dir = Path("test_check_atomic")
    save_dir.mkdir(exist_ok=True)
    
    mgr = CheckpointManager(save_dir)
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Mock some metrics
    metrics = {"avg_iou": 0.5}
    
    logger.info("Testing atomic save...")
    mgr.save_latest(model, optimizer, None, 1, metrics)
    
    latest_path = save_dir / "latest.pt"
    assert latest_path.exists(), "latest.pt should exist"
    
    # Check if tmp file is gone
    tmp_path = latest_path.with_suffix(".pt.tmp")
    assert not tmp_path.exists(), "tmp file should be removed"
    
    logger.info("✅ Atomic save test passed.")

def test_robust_load():
    save_dir = Path("test_check_robust")
    save_dir.mkdir(exist_ok=True)
    latest_path = save_dir / "latest.pt"
    
    # Corrupt the file (truncate it)
    logger.info("Creating corrupted checkpoint file...")
    with open(latest_path, "wb") as f:
        f.write(b"corrupted data (truncated)")
    
    # Setup trainer config
    config = TrainingConfig(checkpoint_dir=str(save_dir))
    model = nn.Linear(10, 2)
    
    # Mocking loaders and loss
    train_loader = None
    val_loader = None
    loss_fn = nn.CrossEntropyLoss()
    
    trainer = Trainer(model, train_loader, val_loader, loss_fn, config)
    
    logger.info("Testing robust load with corrupted file...")
    # This should return 0 instead of crashing with EOFError
    res = trainer.load_checkpoint(latest_path)
    
    assert res == 0, "load_checkpoint should return 0 for corrupted file"
    logger.info("✅ Robust load test passed (returned 0 as expected, no crash).")

if __name__ == "__main__":
    try:
        test_atomic_save()
        test_robust_load()
        logger.info("🎉 All robustness tests passed successfully!")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup test directories
        import shutil
        for p in ["test_check_atomic", "test_check_robust"]:
            if Path(p).exists():
                shutil.rmtree(p)
