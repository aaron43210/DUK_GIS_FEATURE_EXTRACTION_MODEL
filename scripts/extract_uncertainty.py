"""
Active Learning Tool: Uncertainty/Entropy Tiled Exporter.
Identifies "hard" regions where the AI is unsure and exports them for QGIS labeling.
"""

import argparse
import logging
import json
from pathlib import Path
import numpy as np
import rasterio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Uncertainty-Export")

def calculate_entropy(prob_map: np.ndarray) -> np.ndarray:
    """
    Shannon Entropy: H = -p*log2(p) - (1-p)*log2(1-p)
    Maximized at p=0.5 (highest uncertainty).
    """
    p = np.clip(prob_map, 1e-7, 1.0 - 1e-7)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def main():
    parser = argparse.ArgumentParser(description="Extract high-uncertainty tiles for active learning.")
    parser.add_argument("--input_tif", required=True, help="Path to raw drone GeoTIFF")
    parser.add_argument("--prob_tif", required=True, help="Path to matching AI probability GeoTIFF (e.g. building_mask.tif)")
    parser.add_argument("--output_dir", required=True, help="Where to save candidate tiles")
    parser.add_argument("--tile_size", type=int, default=512, help="Size of tiles to extract")
    parser.add_argument("--top_n", type=int, default=50, help="Number of highest-uncertainty tiles to export")
    parser.add_argument("--min_mask_presence", type=float, default=0.01, help="Minimum mask area (0-1) to consider a tile (avoids empty fields)")
    
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Analyzing uncertainty in: {args.prob_tif}")
    
    with rasterio.open(args.prob_tif) as prob_src, rasterio.open(args.input_tif) as raw_src:
        if prob_src.shape != raw_src.shape:
            logger.error("Probability and Raw images must have same dimensions!")
            return

        H, W = prob_src.shape
        tile_size = args.tile_size
        
        candidates = []
        
        # Grid processing for entropy calculation
        for y in tqdm(range(0, H, tile_size), desc="Scanning uncertainty"):
            for x in range(0, W, tile_size):
                window = rasterio.windows.Window(x, y, min(tile_size, W-x), min(tile_size, H-y))
                if window.width < tile_size // 2 or window.height < tile_size // 2:
                    continue
                
                # Read probability tile
                prob_tile = prob_src.read(1, window=window)
                
                # Filter: skip tiles with almost no feature detection (saves effort)
                presence = (prob_tile > 0.1).mean()
                if presence < args.min_mask_presence:
                    continue
                
                # Calculate entropy
                entropy_tile = calculate_entropy(prob_tile)
                avg_entropy = float(np.mean(entropy_tile))
                
                candidates.append({
                    "avg_entropy": avg_entropy,
                    "window": (x, y, window.width, window.height),
                    "presence": float(presence)
                })

        # Sort by highest uncertainty
        candidates.sort(key=lambda x: x["avg_entropy"], reverse=True)
        top_candidates = candidates[:args.top_n]
        
        logger.info(f"Exporting top {len(top_candidates)} uncertainty tiles to {out_dir}")
        
        metadata = []
        for i, cand in enumerate(top_candidates):
            x, y, w, h = cand["window"]
            win = rasterio.windows.Window(x, y, w, h)
            
            # Read and save raw image tile
            raw_tile = raw_src.read(window=win)
            out_name = f"uncertainty_tile_{i:03d}_{x}_{y}.tif"
            out_path = out_dir / out_name
            
            profile = raw_src.profile.copy()
            profile.update({
                "height": h,
                "width": w,
                "transform": raw_src.window_transform(win)
            })
            
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(raw_tile)
                
            metadata.append({
                "id": i,
                "tile_path": out_name,
                "entropy": cand["avg_entropy"],
                "presence": cand["presence"],
                "pixel_coords": [x, y, w, h]
            })

        # Save metadata for QGIS layer naming
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info("Done! Metadata saved to metadata.json")

if __name__ == "__main__":
    main()
