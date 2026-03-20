"""
Mask & Detection → GIS Vector (GeoPackage) export module.

V3 Improvements:
- Integrated YOLOv8 detection box to GIS point conversion.
- Consolidated feature mapping for ensemble outputs.
- Robust geometry cleaning and attribute assignment.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import LineString, Point, Polygon, MultiPolygon, shape

from inference.postprocess import (
    get_threshold,
    prune_skeleton,
    refine_line,
    refine_mask,
    refine_polygon,
    snap_line_endpoints,
)

logger = logging.getLogger(__name__)

# GIS Attribute Labels
ROOF_LABELS = ["Background", "RCC", "Tiled", "Tin", "Others"]

# Feature Optimization Settings
FEATURE_CONFIG: Dict[str, Dict[str, Any]] = {
    "building_mask": {
        "name": "Built-up_Area",
        "type": "Polygon",
        "min_area": 15,
        "simplify": 0.5,
    },
    "roof_type_mask": {
        "name": "Roof_Classification",
        "type": "Polygon",
        "min_area": 10,
        "simplify": 0.5,
    },
    "road_mask": {"name": "Road", "type": "Polygon", "min_area": 30, "simplify": 0.8},
    "road_centerline_mask": {
        "name": "Road_Centre_Line",
        "type": "LineString",
        "min_length": 5,
        "simplify": 0.5,
    },
    "waterbody_mask": {
        "name": "Water_Body",
        "type": "Polygon",
        "min_area": 25,
        "simplify": 1.0,
    },
    "waterbody_line_mask": {
        "name": "Water_Body_Line",
        "type": "LineString",
        "min_length": 5,
        "simplify": 0.5,
    },
    "waterbody_point_mask": {"name": "Well", "type": "Point"},
    "utility_line_mask": {
        "name": "Utility_Pipeline_Wires",
        "type": "LineString",
        "min_length": 5,
        "simplify": 0.5,
    },
    "utility_point_mask": {"name": "Utility_Point_Generic", "type": "Point"},
    "utility_transformer_mask": {"name": "Transformer", "type": "Point"},
    "bridge_mask": {
        "name": "Bridge",
        "type": "Polygon",
        "min_area": 20,
        "simplify": 0.5,
    },
    "railway_mask": {
        "name": "Railway",
        "type": "LineString",
        "min_length": 10,
        "simplify": 1.0,
    },
}

# YOLO Class Mapping (used for point categorization)
YOLO_CLASS_INFO = {
    0: {"name": "Well", "key": "waterbody_point_mask"},
    1: {"name": "Transformer", "key": "utility_transformer_mask"},
}


def _mask_to_geometries(
    mask: np.ndarray,
    transform: rasterio.Affine,
    threshold: float = 0.5,
    geom_type: str = "Polygon",
    min_val: float = 20.0,
    simplify_tol: float = 0.5,
    feature_key: str = "",
) -> list:
    """Convert mask to cleaned shapely geometries with post-processing."""
    binary = (mask > threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

    # ── Stage 1: Mask-level morphological refinement ──
    binary = refine_mask(binary, feature_key)
    if binary.sum() == 0:
        return []

    if geom_type == "LineString":
        from skimage.morphology import skeletonize

        skeleton = skeletonize(binary)

        # ── Stage 2: Skeleton pruning (remove spurious branches) ──
        skeleton = prune_skeleton(skeleton, feature_key)

        shapes = list(
            rasterio_shapes(
                skeleton.astype(np.uint8), mask=skeleton > 0, transform=transform
            )
        )
        geoms = [shape(g) for g, v in shapes if v > 0]
        geoms = [g for g in geoms if g.length > min_val]

        # ── Stage 3: Chaikin smoothing on each line ──
        geoms = [
            refine_line(g, feature_key) if isinstance(g, LineString) else g
            for g in geoms
        ]

        # ── Stage 4: Dead-end snapping (connect broken endpoints) ──
        geoms = snap_line_endpoints(geoms, feature_key)

        return geoms

    elif geom_type == "Polygon":
        shapes = list(rasterio_shapes(binary, mask=binary > 0, transform=transform))
        geoms = []
        for g, v in shapes:
            poly = shape(g)
            if poly.area < min_val:
                continue
            if simplify_tol > 0:
                poly = poly.simplify(simplify_tol, preserve_topology=True)
            if not poly.is_empty and poly.is_valid:
                # ── Stage 3: Polygon refinement (orthogonalize / smooth) ──
                poly = refine_polygon(poly, feature_key)
                if not poly.is_empty and poly.is_valid:
                    geoms.append(poly)
        return geoms

    elif geom_type == "Point":
        from skimage.measure import label, regionprops

        labels = label(binary)
        # centroids are (row, col) i.e (y, x), needs to be (x, y) for transform
        return [
            Point(transform * (prop.centroid[1], prop.centroid[0]))
            for prop in regionprops(labels)
        ]

    return []


def _roof_mask_to_records(
    roof_mask: np.ndarray,
    transform: rasterio.Affine,
    min_area: float = 10.0,
    simplify_tol: float = 0.5,
) -> List[Dict[str, Any]]:
    """Convert class-index roof mask to polygon records with roof_type labels."""
    records: List[Dict[str, Any]] = []
    for class_id in range(1, min(len(ROOF_LABELS), 5)):
        binary = (roof_mask == class_id).astype(np.uint8)
        if binary.sum() == 0:
            continue
        shapes = list(rasterio_shapes(binary, mask=binary > 0, transform=transform))
        for geom_raw, val in shapes:
            if val <= 0:
                continue
            poly = shape(geom_raw)
            if poly.area < min_area:
                continue
            if simplify_tol > 0:
                poly = poly.simplify(simplify_tol, preserve_topology=True)
            if poly.is_empty or not poly.is_valid:
                continue
            records.append(
                {
                    "geometry": poly,
                    "class": "Roof_Types",
                    "roof_type": ROOF_LABELS[class_id],
                }
            )
    return records


class GISExporter:
    """Production exporter for ensemble V3 predictions."""

    def __init__(
        self,
        output_dir: Path,
        crs: Any,
        default_threshold: float = 0.5,
        task_thresholds: Optional[Dict[str, float]] = None,
        export_format: str = "GPKG",  # 'GPKG' or 'SHP'
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crs = crs
        self.default_threshold = float(default_threshold)
        self.task_thresholds = task_thresholds or {}
        self.export_format = export_format.upper()

    def _get_threshold(self, key: str) -> float:
        return float(self.task_thresholds.get(key, self.default_threshold))

    def export(
        self,
        results: Dict[str, Any],
        roof_mask: Optional[np.ndarray] = None,
        transform: Optional[rasterio.Affine] = None,
    ) -> Dict[str, Path]:
        """Main export loop handling both segmentation and detection."""
        exported_paths = {}
        if transform is None:
            logger.error("Transform is required for export.")
            return {}

        # Auto-detect roof mask in results if possible
        if roof_mask is None and "roof_type_mask" in results:
            roof_mask = results["roof_type_mask"]

        # 1. Process Segmentation Masks
        for key, config in FEATURE_CONFIG.items():
            if key not in results or not isinstance(results[key], np.ndarray):
                continue

            logger.info("Vectorizing %s...", config["name"])
            mask = results[key]

            if key == "roof_type_mask":
                records = _roof_mask_to_records(
                    mask,
                    transform,
                    min_area=float(config.get("min_area", 10)),
                    simplify_tol=float(config.get("simplify", 0.5)),
                )
                if records:
                    exported_paths[key] = self._write_records(
                        records, config["name"]
                    )
                continue

            # Use per-class adaptive threshold from postprocess config
            adaptive_thresh = get_threshold(key)
            fallback_thresh = self._get_threshold(key)
            threshold = adaptive_thresh if adaptive_thresh != 0.5 else fallback_thresh

            geoms = _mask_to_geometries(
                mask,
                transform,
                threshold=threshold,
                geom_type=str(config["type"]),
                min_val=float(config.get("min_area", config.get("min_length", 0))),
                simplify_tol=float(config.get("simplify", 0)),
                feature_key=key,
            )

            if geoms:
                exported_paths[key] = self._write_gpkg(
                    geoms, config["name"], key, transform, roof_mask
                )

        # 2. Process YOLO Detections
        if "detections" in results:
            logger.info("Processing YOLOv8 detections...")
            det_by_key: Dict[str, List[Dict[str, Any]]] = {}
            for det in results["detections"]:
                info = YOLO_CLASS_INFO.get(det["class"])
                if info:
                    key = info["key"]
                    if key not in det_by_key:
                        det_by_key[key] = []

                    x1, y1, x2, y2 = det["box"]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    world_pt = Point(transform * (cx, cy))

                    det_by_key[key].append(
                        {
                            "geometry": world_pt,
                            "label": det.get("label", info["name"]),
                            "confidence": det.get("conf", 0.0),
                        }
                    )

            for key, items in det_by_key.items():
                name = FEATURE_CONFIG[key]["name"]
                exported_paths[f"det_{key}"] = self._write_records(items, name)

        return exported_paths

    def _write_records(self, records: List[Dict[str, Any]], layer_name: str) -> Path:
        """Write pre-built records to a GeoPackage or Shapefile layer."""
        for i, row in enumerate(records):
            if "id" not in row:
                row["id"] = i

        gdf = gpd.GeoDataFrame(records, crs=self.crs)

        # Standard attribute enrichment
        if not gdf.empty:
            # Add feature class if missing
            if "feature" not in gdf.columns:
                gdf["feature"] = layer_name

            # Geometry units
            if "Polygon" in str(gdf.geometry.iloc[0].geom_type):
                gdf["Area_SqM"] = gdf.geometry.area
            elif "LineString" in str(gdf.geometry.iloc[0].geom_type):
                gdf["Length_M"] = gdf.geometry.length

        # Sanitize layer name for safe file path
        safe_name = "".join(
            c for c in layer_name if c.isalnum() or c in ("_", "-", ".")
        )

        if self.export_format == "SHP":
            out_path = self.output_dir / f"{safe_name}.shp"
            gdf.to_file(out_path)
        else:
            out_path = self.output_dir / f"{safe_name}.gpkg"
            gdf.to_file(out_path, driver="GPKG")

        logger.info("  Saved %d features to %s", len(gdf), out_path.name)
        return out_path

    def _write_gpkg(
        self,
        geoms: list,
        name: str,
        key: str,
        transform: rasterio.Affine,
        roof_mask: Optional[np.ndarray] = None,
    ) -> Path:
        """Helper to write GDF to GeoPackage."""
        records = []
        for i, g in enumerate(geoms):
            # Polygon orientation/topology cleaning
            if isinstance(g, (Polygon, MultiPolygon)) and not g.is_valid:
                g = g.buffer(0)

            data = {"geometry": g, "id": i, "class": name}

            # Roof Attribute logic (Specific to buildings)
            if key == "building_mask" and roof_mask is not None:
                cp = g.representative_point()
                inv_transform = ~transform
                px, py = inv_transform * (cp.x, cp.y)
                r, c = int(py), int(px)
                if 0 <= r < roof_mask.shape[0] and 0 <= c < roof_mask.shape[1]:
                    idx = int(roof_mask[r, c])
                    data["roof_type"] = ROOF_LABELS[min(idx, 4)]

            records.append(data)

        gdf = gpd.GeoDataFrame(records, crs=self.crs)

        # Standard attribute enrichment
        if not gdf.empty:
            if "feature" not in gdf.columns:
                gdf["feature"] = name
            if "Polygon" in str(gdf.geometry.iloc[0].geom_type):
                gdf["Area_SqM"] = gdf.geometry.area
            elif "LineString" in str(gdf.geometry.iloc[0].geom_type):
                gdf["Length_M"] = gdf.geometry.length

        if self.export_format == "SHP":
            out_path = self.output_dir / f"{name}.shp"
            gdf.to_file(out_path)
        else:
            out_path = self.output_dir / f"{name}.gpkg"
            gdf.to_file(out_path, driver="GPKG")

        logger.info("  Saved %d features to %s", len(gdf), out_path.name)
        return out_path


def export_predictions(
    results: Dict[str, Any],
    tif_path: Path,
    output_dir: Path,
    threshold: Union[float, Dict[str, float]] = 0.5,
    roof_type_mask: Optional[np.ndarray] = None,
    export_format: str = "GPKG",
) -> Dict[str, Path]:
    """Compatibility wrapper."""
    if isinstance(threshold, dict):
        default_threshold = 0.5
        task_thresholds = threshold
    else:
        default_threshold = float(threshold)
        task_thresholds = None

    with rasterio.open(tif_path) as src:
        exporter = GISExporter(
            output_dir,
            src.crs,
            default_threshold=default_threshold,
            task_thresholds=task_thresholds,
            export_format=export_format,
        )
        return exporter.export(
            results, roof_mask=roof_type_mask, transform=src.transform
        )
