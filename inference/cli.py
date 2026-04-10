"""
DUK Feature Extraction - Command Line Interface (CLI)
====================================================
Standalone entry point for QGIS and other integrations.
Usage: python cli.py --input map.tif --model best.pt --output ./results

9 extractable layers:
  SegFormer (Segmentation):
    building_mask, road_mask, road_centerline_mask,
    waterbody_mask, waterbody_line_mask, roof_type_mask
  YOLO (Point Detection):
    waterbody_point_mask  (class 0: Wells)
    utility_transformer_mask (class 1: Transformers)
    overhead_tank_mask    (class 2: Overhead Tanks)
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("DUK-CLI")

ALL_LAYERS = [
    # SegFormer segmentation
    "building_mask",
    "road_mask",
    "road_centerline_mask",
    "waterbody_mask",
    "waterbody_line_mask",
    "roof_type_mask",
    # YOLO point detection
    "waterbody_point_mask",  # Wells
    "utility_transformer_mask",  # Transformers
    "overhead_tank_mask",  # Overhead Tanks
]


def main():
    parser = argparse.ArgumentParser(
        description="DUK AI Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available layers:\n  " + "\n  ".join(ALL_LAYERS),
    )
    parser.add_argument("--input", required=True, help="Path to input GeoTIFF")
    parser.add_argument(
        "--model", required=True, help="Path to best.pt SegFormer model"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for GPKG files"
    )
    parser.add_argument(
        "--device", default="auto", help="Inference device: auto | cuda | mps | cpu"
    )
    parser.add_argument("--yolo", help="Path to YOLO .pt model for point detection")
    parser.add_argument(
        "--sam",
        action="store_true",
        help="Enable SAM 2.1 boundary refinement (buildings)",
    )
    parser.add_argument(
        "--sam-model",
        default="sam2_t.pt",
        help="Path to SAM weights (default: sam2_t.pt)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        metavar="LAYER",
        help=f"Layers to extract (default: all). One or more of:\n"
        + ", ".join(ALL_LAYERS),
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable 8-fold test-time augmentation (slower, more accurate)",
    )

    args = parser.parse_args()

    # ── Device resolution ──────────────────────────────────────────────
    device = args.device
    if device == "auto":
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Device: {device}")

    # ── Add project root to sys.path ──────────────────────────────────
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from inference.predict import load_ensemble_pipeline
        from inference.export import export_predictions
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        sys.exit(1)

    # ── Validate layer names ───────────────────────────────────────────
    selected_layers = None
    if args.layers:
        invalid = [l for l in args.layers if l not in ALL_LAYERS]
        if invalid:
            logger.error(
                f"Unknown layer(s): {invalid}\n" f"Valid choices: {ALL_LAYERS}"
            )
            sys.exit(1)
        selected_layers = list(args.layers)

        # Bidirectional auto-enable: roofs are attributes ON building polygons
        if "roof_type_mask" in selected_layers and "building_mask" not in selected_layers:
            selected_layers.append("building_mask")
            logger.info("Auto-added building_mask (required for roof classification)")
        if "building_mask" in selected_layers and "roof_type_mask" not in selected_layers:
            selected_layers.append("roof_type_mask")
            logger.info("Auto-added roof_type_mask for building extraction")

        logger.info(f"Extracting layers: {selected_layers}")
    else:
        logger.info("Extracting all layers")

    # ── Load model ────────────────────────────────────────────────────
    logger.info(f"Loading model: {args.model}")
    predictor = load_ensemble_pipeline(
        weights_path=args.model,
        yolo_path=args.yolo,
        device=device,
        use_tta=args.tta,
    )

    # ── Run inference (selected_masks passed inside for efficiency) ───
    logger.info(f"Running inference: {args.input}")
    results = predictor.predict_tif(
        Path(args.input),
        selected_masks=selected_layers,  # ← filters INSIDE inference engine
    )

    # ── Image Read for Spectral/SAM Refinement ──────────────────────────
    image_rgb = None
    if args.sam or "building_mask" in results:
        logger.info("Reading image RGB for professional refinement...")
        with rasterio.open(str(args.input)) as src:
            img_data = src.read([1, 2, 3])
            from predict import _percentile_stretch
            img_norm = _percentile_stretch(np.transpose(img_data, (1, 2, 0)))
            image_rgb = (img_norm * 255).astype(np.uint8)
            logger.info("  Image RGB loaded successfully.")

    # ── Export ────────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract roof classification mask for building attribute assignment
    roof_mask = results.pop("roof_type_mask", None)

    logger.info(f"Exporting to: {args.output} (SAM={args.sam})")
    exported = export_predictions(
        results,
        tif_path=Path(args.input),
        output_dir=out_dir,
        roof_mask=roof_mask,
        image_rgb=image_rgb,
        use_sam=args.sam,
        device=device,
    )

    logger.info(f"Exported {len(exported)} layer(s).")
    for key, path in exported.items():
        print(f"EXPORTED:{key}:{path}")


if __name__ == "__main__":
    main()
