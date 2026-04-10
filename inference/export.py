"""
Professional-Grade GIS Vector Export Manager for DUK.

Restored and optimized version for high-precision building extraction.
Performs all geometric refinement in pixel space to ensure correct scaling.
Integrates Shadow-Aware refinement and Watershed Separation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon, LineString, shape
from shapely.ops import unary_union

from .postprocess import (
    refine_mask,
    separate_instances,
    refine_polygon,
    refine_line,
    snap_line_endpoints,
    get_threshold,
    SAM2Refiner,
    GlobalAligner,
    ShadowDetector,
    orthogonalize_polygon,
    ROOF_LABELS,
    ROOF_COLORS_HEX,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Feature Configuration — all 9 GIS layers
# ─────────────────────────────────────────────────────────────────────
FEATURE_CONFIG: Dict[str, dict] = {
    # ── SegFormer segmentation outputs ────────────────────────────
    "building_mask": {"name": "Buildings", "type": "Polygon"},
    "road_mask": {"name": "Roads", "type": "Polygon"},
    "road_centerline_mask": {"name": "Road_Centrelines", "type": "LineString"},
    "waterbody_mask": {"name": "Water_Polygons", "type": "Polygon"},
    "waterbody_line_mask": {"name": "Water_Lines", "type": "LineString"},
    # ── YOLO point detection outputs ──────────────────────────────
    "waterbody_point_mask": {"name": "Wells", "type": "Point"},
    "utility_transformer_mask": {"name": "Transformers", "type": "Point"},
    "overhead_tank_mask": {"name": "Overhead_Tanks", "type": "Point"},
}


# ROOF_LABELS and ROOF_COLORS_HEX are now imported from postprocess.py
# for single-source-of-truth consistency.


def _merge_point_blobs(
    geoms: List[Any],
    nms_radius_px: float,
    transform: rasterio.Affine,
) -> List[Any]:
    """
    Spatial NMS for point features: merge duplicate detections
    (multiple YOLO boxes from overlapping tiles) into one centroid.
    Operates in GIS coordinate space.
    """
    if len(geoms) <= 1:
        return geoms
    # Convert NMS radius from pixels to GIS units using pixel size
    pixel_size = abs(transform.a)  # metres per pixel
    radius_m = nms_radius_px * pixel_size

    kept: List[Any] = []
    suppressed = [False] * len(geoms)
    for i, g in enumerate(geoms):
        if suppressed[i]:
            continue
        cluster = [g]
        for j in range(i + 1, len(geoms)):
            if not suppressed[j] and g.distance(geoms[j]) <= radius_m:
                cluster.append(geoms[j])
                suppressed[j] = True
        # Centroid of cluster
        xs = [p.x for p in cluster]
        ys = [p.y for p in cluster]
        kept.append(Point(sum(xs) / len(xs), sum(ys) / len(ys)))
    return kept


class ExportManager:
    """Orchestrates the conversion from probability rasters to refined GIS vector files."""

    def __init__(
        self,
        output_dir: Path,
        crs: Any,
        transform: rasterio.Affine,
        export_format: str = "GPKG",
        use_sam: bool = False,
        device: str = "cpu",
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crs = crs
        self.transform = transform
        self.shapely_transform = (
            transform.a,
            transform.b,
            transform.d,
            transform.e,
            transform.xoff,
            transform.yoff,
        )
        self.export_format = export_format.upper()
        self.use_sam = use_sam
        self.sam_refiner = SAM2Refiner(device=device) if use_sam else None
        self.global_aligner = GlobalAligner()
        self.shadow_detector = ShadowDetector()

    def _mask_to_geometries(
        self,
        prob_map: np.ndarray,
        feature_key: str,
        image_rgb: Optional[np.ndarray] = None,
        shadow_mask: Optional[np.ndarray] = None,
    ) -> List[Any]:
        """Convert a probability map to refined Shapely geometries (Pixel-space logic)."""
        threshold = get_threshold(feature_key)
        binary_mask = (prob_map >= threshold).astype(np.uint8)

        # 1. Mask-level cleanup
        refined_mask = refine_mask(binary_mask, feature_key, prob_map=prob_map)
        if refined_mask.sum() == 0:
            return []

        geoms = []
        if FEATURE_CONFIG[feature_key]["type"] == "Point":
            labeled = separate_instances(refined_mask, feature_key)
            for i in range(1, labeled.max() + 1):
                rows, cols = np.where(labeled == i)
                if len(rows) > 0:
                    cy, cx = np.mean(rows), np.mean(cols)
                    geoms.append(Point(self.transform * (cx, cy)))
        else:
            # 2. Instance-level Vectorization & SAM Snapping
            labeled = separate_instances(refined_mask, feature_key)
            num_instances = labeled.max()

            pixel_geoms = []
            for i in range(1, num_instances + 1):
                inst_mask = (labeled == i).astype(np.uint8)

                # BUILDING FOCUS: Skip tiny noise (smaller than 15sqm approx)
                if feature_key == "building_mask" and inst_mask.sum() < 100:
                    continue

                # Professional Snapping (SAM 2.1)
                if (
                    self.use_sam
                    and feature_key == "building_mask"
                    and image_rgb is not None
                ):
                    inst_mask = self.sam_refiner.refine_building(
                        image_rgb, inst_mask, shadow_mask=shadow_mask
                    )

                # Vectorize in PIXELS
                inst_shapes = shapes(inst_mask, mask=(inst_mask > 0))
                for s, v in inst_shapes:
                    if v > 0:
                        pixel_geoms.append(shape(s))

            # 3. Geometry-level refinement
            if FEATURE_CONFIG[feature_key]["type"] == "Polygon":
                if feature_key == "building_mask" and len(pixel_geoms) > 1:
                    # 3.1 Adaptive Merging (STOPS STRIPS)
                    # Fuse polygons that are nearly touching and part of the same structure
                    from shapely.ops import unary_union

                    # Buffer 2px to bridge gaps, union, then unbuffer
                    merged = unary_union(
                        [p.buffer(2) for p in pixel_geoms if isinstance(p, Polygon)]
                    )
                    if isinstance(merged, Polygon):
                        pixel_geoms = [merged.buffer(-1.5)]  # Shrink slightly back
                    else:
                        # Multi-polygon: split back to individual pieces
                        pixel_geoms = [
                            p.buffer(-1.5)
                            for p in list(merged.geoms)
                            if not p.buffer(-1.5).is_empty
                        ]

                    # GIS Engineering: Hough-Guided Visual Locking
                    # Get dominant village angle for fallback
                    village_angles = self.global_aligner.find_dominant_village_angles(
                        pixel_geoms
                    )
                    dominant_rad = (
                        np.radians(village_angles[0]) if village_angles else None
                    )

                    for g in pixel_geoms:
                        if isinstance(g, Polygon):
                            # Extract local RGB tile for this specific building
                            tile = None
                            if image_rgb is not None:
                                bounds = g.bounds
                                y1, x1, y2, x2 = map(
                                    int,
                                    [
                                        bounds[1] - 10,
                                        bounds[0] - 10,
                                        bounds[3] + 10,
                                        bounds[2] + 10,
                                    ],
                                )
                                h, w = image_rgb.shape[:2]
                                tile = image_rgb[
                                    max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)
                                ]

                            # Force building to snap to its own visual lines (Hough)
                            # or fallback to village grid
                            refined = orthogonalize_polygon(
                                g, dominant_angle_override=dominant_rad, image_tile=tile
                            )
                            if not refined.is_empty:
                                geoms.append(
                                    affine_transform(refined, self.shapely_transform)
                                )
                else:
                    for g in pixel_geoms:
                        if isinstance(g, Polygon):
                            refined = refine_polygon(g, feature_key)
                            if not refined.is_empty:
                                geoms.append(
                                    affine_transform(refined, self.shapely_transform)
                                )
            else:
                for g in pixel_geoms:
                    if isinstance(g, LineString):
                        refined = refine_line(g, feature_key)
                        if not refined.is_empty:
                            geoms.append(
                                affine_transform(refined, self.shapely_transform)
                            )
                geoms = snap_line_endpoints(geoms, feature_key)

        return geoms

    def export(
        self,
        results: Dict[str, np.ndarray],
        roof_mask: Optional[np.ndarray] = None,
        image_rgb: Optional[np.ndarray] = None,
    ) -> Dict[str, Path]:
        """Process all results and write to disk."""
        exported_files = {}

        # Pre-detect shadows if image is available
        shadow_mask = None
        if image_rgb is not None:
            logger.info("Detecting shadow regions for boundary pruning...")
            shadow_mask = self.shadow_detector.get_shadow_mask(image_rgb)

        # Pre-calculate road union for clipping buildings (GIS Engineering)
        road_union = None
        if "road_mask" in results:
            road_prob = results["road_mask"]
            road_bin = (road_prob >= get_threshold("road_mask")).astype(np.uint8)
            road_shapes = shapes(road_bin, mask=(road_bin > 0))
            road_geoms = [shape(s) for s, v in road_shapes if v > 0]
            if road_geoms:
                road_union = unary_union(road_geoms)

        for key, prob in results.items():
            if key == "detections" or key == "roof_type_mask":
                continue
            if key not in FEATURE_CONFIG:
                continue

            config = FEATURE_CONFIG[key]
            logger.info(f"Exporting professional layer: {config['name']}")

            geoms = self._mask_to_geometries(
                prob, key, image_rgb=image_rgb, shadow_mask=shadow_mask
            )

            # REAL WORLD: Clip buildings by roads to prevent overlaps
            if key == "building_mask" and road_union is not None:
                clipped_geoms = []
                for g in geoms:
                    # Work in GIS space for clipping
                    if g.intersects(road_union):
                        g = g.difference(road_union)
                    if not g.is_empty:
                        clipped_geoms.append(g)
                geoms = clipped_geoms

            if not geoms:
                continue

            records = []
            for i, g in enumerate(geoms):
                data = {"geometry": g, "id": i, "feature": config["name"]}

                if key == "building_mask":
                    inv_transform = ~self.transform
                    roof_label = self._classify_building_roof(
                        g, inv_transform, roof_mask, image_rgb
                    )
                    data["roof_type"] = roof_label
                    data["roof_color"] = ROOF_COLORS_HEX.get(
                        roof_label, "#CCCCCC"
                    )

                records.append(data)

            gdf = gpd.GeoDataFrame(records, crs=self.crs)
            if not gdf.empty:
                if config["type"] == "Polygon":
                    gdf["Area_SqM"] = gdf.geometry.area
                elif config["type"] == "LineString":
                    gdf["Length_M"] = gdf.geometry.length

            ext = "shp" if self.export_format == "SHP" else "gpkg"
            file_name = f"{config['name']}.{ext}"
            out_path = self.output_dir / file_name

            if self.export_format == "SHP":
                gdf.to_file(out_path)
            else:
                gdf.to_file(out_path, driver="GPKG", layer=config["name"])

            exported_files[key] = out_path
            logger.info(f"  Saved {len(gdf)} features to {file_name}")

            # Generate QML categorized style for buildings
            if key == "building_mask" and roof_mask is not None:
                qml_path = self.output_dir / f"{config['name']}.qml"
                _write_building_qml(qml_path)
                logger.info(f"  Style file: {qml_path.name}")

        return exported_files

    def _classify_building_roof(
        self,
        geom: Polygon,
        inv_transform,
        roof_mask: Optional[np.ndarray],
        image_rgb: Optional[np.ndarray],
    ) -> str:
        """
        Classify a building's roof using high-fidelity majority voting from the NN predictions.
        (Spectral HSV analysis removed for increased robustness to lighting variations).
        """
        if roof_mask is None:
            return ROOF_LABELS.get(0, "Incomplete")
            
        return self._get_majority_nn_roof(geom, roof_mask, inv_transform)

    def _get_majority_nn_roof(self, geom, roof_mask, inv_transform) -> str:
        """Internal helper for NN majority vote."""
        try:
            from rasterio.features import rasterize
            from shapely.affinity import affine_transform as sat
            bounds = geom.bounds
            px_min_x, px_min_y = inv_transform * (bounds[0], bounds[3])
            px_max_x, px_max_y = inv_transform * (bounds[2], bounds[1])
            r0, r1 = max(0, int(px_min_y)), int(px_max_y) + 1
            c0, c1 = max(0, int(px_min_x)), int(px_max_x) + 1
            
            roi = roof_mask[r0:r1, c0:c1]
            px_geom = sat(geom, [
                inv_transform.a, inv_transform.b,
                inv_transform.d, inv_transform.e,
                inv_transform.xoff - c0, inv_transform.yoff - r0,
            ])
            mask = rasterize([(px_geom, 1)], out_shape=roi.shape, fill=0, dtype=np.uint8)
            values = roi[mask > 0]
            if len(values) == 0: return "Unknown"
            unique, counts = np.unique(values, return_counts=True)
            return ROOF_LABELS.get(int(unique[np.argmax(counts)]), "Unknown")
        except: return "Unknown"



def _write_building_qml(qml_path: Path) -> None:
    """Generate a QGIS categorized renderer style file for roof types."""
    categories = []
    for label_name, hex_color in ROOF_COLORS_HEX.items():
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        categories.append(
            f'      <category symbol="{label_name}" '
            f'value="{label_name}" label="{label_name}" render="true"/>'
        )

    symbols = []
    for i, (label_name, hex_color) in enumerate(ROOF_COLORS_HEX.items()):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        symbols.append(f'''      <symbol name="{label_name}" type="fill" alpha="0.75">
        <layer class="SimpleFill" enabled="1">
          <prop k="color" v="{r},{g},{b},190"/>
          <prop k="outline_color" v="35,35,35,255"/>
          <prop k="outline_width" v="0.26"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>''')

    qml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<qgis version="3.28">
  <renderer-v2 type="categorizedSymbol" attr="roof_type"
               enableorderby="0" symbollevels="0">
    <categories>
{chr(10).join(categories)}
    </categories>
    <symbols>
{chr(10).join(symbols)}
    </symbols>
  </renderer-v2>
  <labeling type="simple">
    <settings>
      <text-style fieldName="roof_type" fontSize="8"
                  fontFamily="Arial" fontWeight="50"/>
    </settings>
  </labeling>
</qgis>
'''
    qml_path.write_text(qml_content, encoding="utf-8")


def export_predictions(
    results: Dict[str, np.ndarray],
    tif_path: Union[str, Path],
    output_dir: Union[str, Path],
    roof_mask: Optional[np.ndarray] = None,
    export_format: str = "GPKG",
    image_rgb: Optional[np.ndarray] = None,
    use_sam: bool = False,
    device: str = "cpu",
) -> Dict[str, Path]:
    """Top-level entry point for QGIS integration."""
    with rasterio.open(str(tif_path)) as src:
        manager = ExportManager(
            output_dir=Path(output_dir),
            crs=src.crs,
            transform=src.transform,
            export_format=export_format,
            use_sam=use_sam,
            device=device,
        )
        return manager.export(results, roof_mask=roof_mask, image_rgb=image_rgb)
