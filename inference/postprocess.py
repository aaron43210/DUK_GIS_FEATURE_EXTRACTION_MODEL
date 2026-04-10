"""
Advanced Post-Processing for GIS Vector Export.

Refines raw AI probability maps and vectorized geometries to produce
survey-grade shapefiles. Techniques are research-backed (ISPRS 2024,
IEEE GRSL) and tailored per feature type.

Pipeline order:
  1. Probability map refinement (CRF, adaptive threshold)
  2. Mask-level morphological cleanup (closing, hole-filling)
  3. Skeleton pruning (skan branch removal for LineString layers)
  4. Vectorization (handled in export.py)
  5. Geometry-level refinement (orthogonalization, smoothing, snapping)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Any, Union

import cv2
import numpy as np
from .topology import RoadNetworkCleaner
import torch
from scipy import ndimage
from shapely.geometry import LineString, Polygon, MultiPolygon
from skimage.morphology import closing, disk, skeletonize
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Roof Classification Labels — Indian Survey Conventions
# ─────────────────────────────────────────────────────────────────────
ROOF_LABELS = {
    0: "Incomplete",
    1: "RCC (Flat)",
    2: "Tiled (Sloped)",
    3: "Tin/Sheet",
    4: "Others (Kutcha)",
}

ROOF_COLORS_HEX = {
    "Incomplete":      "#95a5a6", # Grey
    "RCC (Flat)":      "#3498db", # Blue
    "Tiled (Sloped)":  "#e67e22", # Orange
    "Tin/Sheet":       "#1abc9c", # Cyan
    "Others (Kutcha)": "#8e44ad", # Purple
}

# ─────────────────────────────────────────────────────────────────────
# Professional Refinement System (SAM 2.1 + Global Alignment)
# ─────────────────────────────────────────────────────────────────────


class SAM2Refiner:
    """
    Uses Segment Anything Model 2.1 (Tiny) to snap AI predictions
    to high-resolution structural edges (roof lines, eaves).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize SAM2Refiner with optional model path.
        
        Args:
            model_path: Path to SAM 2.1 model. If None, uses models/sam2.1_t.pt
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        if model_path is None:
            # Use relative path from project root
            model_path = str(Path(__file__).parent.parent / "models" / "sam2.1_t.pt")
        
        self.model_path = Path(model_path)
        self.device = device
        self._model = None

        # Ensure writable directory for model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        if self._model is None:
            try:
                from ultralytics import SAM

                # Note: This will download if not present, but only to our writable path
                self._model = SAM(str(self.model_path))
                self._model.to(self.device)
                logger.info(f"SAM 2.1 loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load SAM 2.1: {e}")
                return None
        return self._model

    def refine_building(
        self,
        image_rgb: np.ndarray,
        initial_mask: np.ndarray,
        shadow_mask: Optional[np.ndarray] = None,
        padding: int = 20,
    ) -> np.ndarray:
        """
        Snap a single building instance mask to visual edges using SAM 2.1.
        Uses Centroid + Distance Peaks as positive prompts.
        """
        if self.model is None or initial_mask.sum() == 0:
            return initial_mask

        # 1. Multi-point prompt: Centroid + Highest Peaks
        distance = ndimage.distance_transform_edt(initial_mask)
        from skimage.feature import peak_local_max

        peaks = peak_local_max(
            distance, min_distance=10, num_peaks=3, labels=initial_mask
        )

        points = []
        labels = []

        # Add peaks as positive prompts
        for py, px in peaks:
            points.append([px, py])
            labels.append(1)

        # Ensure we have at least the centroid if no peaks found
        if not points:
            cy, cx = np.argwhere(initial_mask > 0).mean(axis=0).astype(int)
            points.append([cx, cy])
            labels.append(1)

        # 2. BBox prompt
        coords = np.argwhere(initial_mask > 0)
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        h, w = initial_mask.shape
        bbox = [max(0, x1 - 5), max(0, y1 - 5), min(w - 1, x2 + 5), min(h - 1, y2 + 5)]

        # 3. Negative prompts (from shadow mask near the building)
        if shadow_mask is not None:
            kernel = np.ones((10, 10), np.uint8)
            buffer = cv2.dilate(initial_mask, kernel) - initial_mask
            shadow_points = np.argwhere((buffer > 0) & (shadow_mask > 0))
            if len(shadow_points) > 5:
                idx = np.random.choice(
                    len(shadow_points), min(5, len(shadow_points)), replace=False
                )
                for sy, sx in shadow_points[idx]:
                    points.append([sx, sy])
                    labels.append(0)

        try:
            results = self.model.predict(
                image_rgb,
                bboxes=[bbox],
                points=points,
                labels=labels,
                device=self.device,
                verbose=False,
            )

            if results and len(results) > 0:
                ref_mask = results[0].masks.data[0].cpu().numpy()
                ref_mask = (ref_mask > 0).astype(np.uint8)

                # Constrain to dilated original to prevent runaway leakage
                kernel = np.ones((padding, padding), np.uint8)
                constraint = cv2.dilate(initial_mask, kernel, iterations=1)

                final_mask = np.logical_and(ref_mask, constraint).astype(np.uint8)
                return final_mask
        except Exception as e:
            logger.debug(f"SAM snapping failed: {e}")

        return initial_mask


class GlobalAligner:
    """
    Ensures a "cadastral" look by snapping building orientations to
    shared village-scale dominant angles.
    """

    def __init__(self, bin_width_deg: float = 2.0):
        self.bin_width = bin_width_deg

    def find_dominant_village_angles(self, polygons: List[Polygon]) -> List[float]:
        """Collect orientations from all buildings to find the village 'grid'."""
        angles = []
        for poly in polygons:
            # Skip tiny noise (threshold in pixels)
            if poly.area < 100:
                continue
            coords = np.array(poly.exterior.coords)
            angle = _dominant_angle(coords) % (np.pi / 2)
            angles.append(np.degrees(angle))

        if not angles:
            return [0.0, 90.0]

        # Histogram-based peak detection
        counts, bins = np.histogram(
            angles, bins=int(90 / self.bin_width), range=(0, 90)
        )
        # Top 2 peaks (usually parallel and perpendicular streets)
        peaks = bins[np.argsort(counts)[-2:]]
        return sorted(list(peaks))

    def snap_to_grid(
        self, poly: Polygon, grid_angles_deg: List[float], tolerance_deg: float = 15.0
    ) -> Polygon:
        """Snap a building to the closest grid angle if it's roughly aligned."""
        coords = np.array(poly.exterior.coords)
        current_angle = np.degrees(_dominant_angle(coords) % (np.pi / 2))

        for target in grid_angles_deg:
            if abs(current_angle - target) < tolerance_deg:
                dominant_rad = np.radians(target)
                return orthogonalize_polygon(
                    poly,
                    snap_tol_deg=tolerance_deg,
                    dominant_angle_override=dominant_rad,
                )

        return poly


class HoughRefiner:
    """
    Industry-Standard: Uses the Hough Transform to find the REAL straight lines
    in the drone photo and 'locks' the AI footprints to them.
    Matches survey-grade precision by following visual eaves.
    """

    def __init__(self, threshold_val: int = 50, min_line_len: int = 30):
        self.thresh = threshold_val
        self.min_len = min_line_len

    def refine_geometry(self, image_rgb_tile: np.ndarray, poly: Polygon) -> Polygon:
        """Lock polygon segments to the strongest visual architectural lines."""
        if poly.is_empty:
            return poly

        # 1. Edge detection on the RGB tile
        gray = cv2.cvtColor(image_rgb_tile, cv2.COLOR_RGB2GRAY)
        # Use Bilateral filter to preserve edges while removing noise (e.g. tile joints)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(gray, 50, 150)

        # 2. Hough Transform (Probabilistic)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=self.thresh,
            minLineLength=self.min_len,
            maxLineGap=20,
        )

        if lines is None:
            return poly

        # 3. Filter lines near the polygon and parallel to its axis
        from shapely.geometry import LineString

        coords = np.array(poly.exterior.coords)
        new_coords = coords.copy()

        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            seg = LineString([p1, p2])
            seg_vec = p2 - p1
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 5:
                continue

            seg_angle = np.arctan2(seg_vec[1], seg_vec[0])

            # Find best matching Hough line for this segment
            best_match = None
            min_dist = 8.0  # Snap tolerance in pixels

            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                l_vec = np.array([lx2 - lx1, ly2 - ly1])
                l_len = np.linalg.norm(l_vec)
                l_angle = np.arctan2(l_vec[1], l_vec[0])

                # Parallelism check (within 10 degrees)
                angle_diff = abs(
                    ((seg_angle - l_angle + np.pi / 2) % np.pi) - np.pi / 2
                )
                if angle_diff > np.radians(10):
                    continue

                # Distance check (segment midpoint to line)
                mid = (p1 + p2) / 2
                dist = np.abs(np.cross(l_vec, mid - [lx1, ly1])) / l_len

                if dist < min_dist:
                    min_dist = dist
                    best_match = line[0]

            if best_match is not None:
                # Snap segment to the Hough line
                lx1, ly1, lx2, ly2 = best_match
                l_vec = np.array([lx2 - lx1, ly2 - ly1])
                l_unit = l_vec / np.linalg.norm(l_vec)

                # Project p1, p2 onto the line
                v1 = p1 - [lx1, ly1]
                v2 = p2 - [lx1, ly1]
                p1_snapped = np.array([lx1, ly1]) + np.dot(v1, l_unit) * l_unit
                p2_snapped = np.array([lx1, ly1]) + np.dot(v2, l_unit) * l_unit

                new_coords[i] = p1_snapped
                new_coords[i + 1] = p2_snapped

        try:
            snapped_poly = Polygon(new_coords)
            if snapped_poly.is_valid and not snapped_poly.is_empty:
                return snapped_poly
        except:
            pass
        return poly


def bridge_tile_seams(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Gently weld 1-2 pixel tile gaps without fusing distinct buildings."""
    if mask.sum() == 0:
        return mask
    kernel = np.ones((radius, radius), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


class GeoAIRegularizer:
    """
    Industry-Standard GeoAI Regularizer.
    Uses Manhattan-frame recursive partitioning to square complex building footprints.
    Supported by Hough-guided visual alignment.
    """

    def __init__(self, eps: float = 2.0):
        self.eps = eps

    def regularize(
        self,
        poly: Polygon,
        dominant_angle: float,
        image_tile: Optional[np.ndarray] = None,
    ) -> Polygon:
        """Recursive Manhattan Squaring."""
        from shapely.affinity import rotate

        if poly.area < 10:
            return poly

        # 1. Rotate to global Manhattan frame
        angle_deg = np.degrees(dominant_angle)
        center = poly.centroid
        p_rot = rotate(poly, -angle_deg, origin=center)

        # 2. Extract and Snap Coordinates (Recursive Grid Alignment)
        coords = np.array(p_rot.exterior.coords)
        snapped = []
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            # SOTA Logic: Use larger delta to prevent 'stair-stepping'
            if abs(dx) > abs(dy):  # Horizontal dominance
                snapped.append([p2[0], p1[1]])
            else:  # Vertical dominance
                snapped.append([p1[0], p2[1]])

        if not snapped:
            return poly
        snapped.append(snapped[0])
        try:
            p_sq = Polygon(snapped)
            if not p_sq.is_valid:
                p_sq = p_sq.buffer(0)

            # 3. Post-Squaring Simplification (Clean up micro-jags)
            p_sq = p_sq.simplify(1.5, preserve_topology=True)

            # 4. Safety Guard: Don't lose too much area
            if p_sq.area < poly.area * 0.5:
                return poly

            # 5. Final Manhattan Rotation back
            return rotate(p_sq, angle_deg, origin=center)
        except:
            return poly


class ShadowDetector:
    """
    Identifies shadow regions in drone imagery using HSV chromaticity analysis.
    Real-world shadows are dark (low V) and shifted towards blue (sky illumination).
    """

    def __init__(
        self, brightness_threshold: int = 70, blue_min: int = 90, blue_max: int = 130
    ):
        self.v_thresh = brightness_threshold
        self.h_min = blue_min
        self.h_max = blue_max

    def get_shadow_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """Create a binary mask of potential shadow areas."""
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # 1. Darkness criteria
        dark = v < self.v_thresh

        # 2. Blue-shift criteria (Sky illumination in shadows)
        # Hue for Blue is typically around 120 in OpenCV (0-179 range)
        blue_ish = (h > self.h_min) & (h < self.h_max) & (s > 20)

        shadow_mask = (dark & blue_ish).astype(np.uint8)
        return shadow_mask


# ─────────────────────────────────────────────────────────────────────
# Per-feature configuration for post-processing parameters
# ─────────────────────────────────────────────────────────────────────
POSTPROCESS_CONFIG: Dict[str, dict] = {
    # ── Segmentation: Building ──────────────────────────────────────
    "building_mask": {
        "closing_radius": 3,
        "fill_holes": True,
        "orthogonalize": True,
        "min_rect_area": 50.0,  # m² — below this, use minAreaRect
        "angle_snap_deg": 5.0,  # snap edges within ±5° of dominant angle
        "threshold": 0.45,  # slightly lower for better recall
        "min_area_px": 50,
    },
    # ── Segmentation: Road ─────────────────────────────────────────
    "road_mask": {
        "closing_radius": 7,
        "fill_holes": True,
        "orthogonalize": False,
        "threshold": 0.50,
        "min_area_px": 100,
    },
    "road_centerline_mask": {
        "closing_radius": 5,
        "fill_holes": False,
        "skeletonize": True,
        "prune_branches": True,
        "min_branch_px": 15,
        "chaikin_iters": 3,
        "snap_distance": 30.0,  # snap line dead-ends up to 30px apart
        "threshold": 0.35,  # lowered for faint road lines under canopy
        "min_area_px": 15,
        "gap_fill_px": 20,  # bridge culvert / tree-canopy gaps
    },
    # ── Segmentation: Water Body ───────────────────────────────────
    "waterbody_mask": {
        # Large closing radius handles internal no-data holes in ponds
        "closing_radius": 11,
        "fill_holes": True,
        "orthogonalize": False,
        # Small ponds (< 500 px²) get convex_hull fill to close scalloped edges
        "convex_hull_area": 500.0,
        "threshold": 0.65,  # Aggressive 0.65 to block field noise
        "min_area_px": 150,  # Larger minimum area
    },
    "waterbody_line_mask": {
        # Canals/streams: thin lines — low branch pruning threshold
        "closing_radius": 2,
        "fill_holes": False,
        "skeletonize": True,
        "prune_branches": True,
        "min_branch_px": 5,
        "chaikin_iters": 3,
        "snap_distance": 30.0,
        "threshold": 0.32,  # Lower threshold ALLOWED because linear_ratio is strict
        "min_area_px": 25,
        "gap_fill_px": 25,
        "linear_ratio": 4.0, # Stricter ratio: length must be 4x width
    },
    # ── YOLO Point Detection: Wells ────────────────────────────────
    "waterbody_point_mask": {
        "threshold": 0.38,
        "min_area_px": 4,
        # Spatial NMS radius (px): blobs closer than this → one centroid
        "point_nms_radius_px": 15,
    },
    # ── YOLO Point Detection: Transformers ────────────────────────
    "utility_transformer_mask": {
        # Transformers are compact, bright objects on poles
        "threshold": 0.45,
        "min_area_px": 4,
        "point_nms_radius_px": 12,  # tighter NMS for dense pole lines
    },
    # ── YOLO Point Detection: Overhead Tanks ──────────────────────
    "overhead_tank_mask": {
        # Overhead tanks are large elevated structures — more distinct
        "threshold": 0.40,
        "min_area_px": 9,
        "point_nms_radius_px": 20,  # larger NMS to avoid double-counting
    },
    # ── Segmentation: Utility Lines ───────────────────────────────
    "utility_line_mask": {
        "closing_radius": 5,
        "fill_holes": False,
        "skeletonize": True,
        "prune_branches": True,
        "min_branch_px": 10,
        "chaikin_iters": 3,
        "snap_distance": 8.0,
        "threshold": 0.45,
        "min_area_px": 20,
    },
}


def get_threshold(feature_key: str) -> float:
    """Return the optimal per-class threshold for a feature."""
    cfg = POSTPROCESS_CONFIG.get(feature_key, {})
    return float(cfg.get("threshold", 0.5))


# ═══════════════════════════════════════════════════════════════════
# 1. MASK-LEVEL REFINEMENT  (operates on binary masks)
# ═══════════════════════════════════════════════════════════════════


def refine_mask(
    mask: np.ndarray, feature_key: str, prob_map: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply morphological closing and hole-filling to a binary mask
    based on the feature type.
    """
    cfg = POSTPROCESS_CONFIG.get(feature_key, {})
    closing_radius = cfg.get("closing_radius", 0)
    fill_holes = cfg.get("fill_holes", False)

    if closing_radius > 0:
        selem = disk(closing_radius)
        mask = closing(mask.astype(bool), selem).astype(np.uint8)

    if fill_holes:
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    # ── Stage 1.5: Skeletonization (for line features) ──
    if cfg.get("skeletonize", False):
        gap_fill = cfg.get("gap_fill_px", 0)
        if gap_fill > 0:
            mask = fill_road_gaps(mask, gap_fill, prob_map=prob_map)

        # Professional Topology Cleanup (Phase 1)
        cleaner = RoadNetworkCleaner(min_branch_length=cfg.get("min_branch_px", 15.0))
        mask = cleaner.clean_mask(mask)

    # ── Stage 2: Connected Component Area & Geometry Filtering ──
    min_area = cfg.get("min_area_px", 0)
    linear_ratio = cfg.get("linear_ratio", 0)
    
    if (min_area > 0 or linear_ratio > 0) and mask.sum() > 0:
        labeled = label(mask)
        refined = np.zeros_like(mask)
        for prop in regionprops(labeled):
            # Area constraint
            if min_area > 0 and prop.area < min_area:
                continue
            
            # Linearity constraint (Major/Minor axis ratio)
            if linear_ratio > 0:
                # Use axis lengths if available, else approximate via bbox
                major = prop.major_axis_length
                minor = prop.minor_axis_length
                if minor > 0:
                    ratio = major / minor
                else:
                    # Fallback to bbox ratio for vertical/horizontal lines
                    minr, minc, maxr, maxc = prop.bbox
                    h, w = maxr - minr, maxc - minc
                    ratio = max(h, w) / max(1, min(h, w))
                
                if ratio < linear_ratio:
                    continue

            refined[labeled == prop.label] = 1
        mask = refined

    return mask


def fill_road_gaps(
    mask: np.ndarray, radius: int = 5, prob_map: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Fill small gaps in linear features using morphological techniques.
    """
    if radius <= 0:
        return mask

    selem = disk(radius).astype(np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), selem, iterations=1)

    if prob_map is not None:
        prob_valid = prob_map > 0.20
        new_pixels = (dilated > 0) & (mask == 0)
        dilated[new_pixels & ~prob_valid] = 0

    closed = closing(dilated.astype(bool), selem).astype(np.uint8)
    refined = cv2.medianBlur(closed, 3)
    return refined


# ═══════════════════════════════════════════════════════════════════
# 2. SKELETON PRUNING  (removes spurious branches)
# ═══════════════════════════════════════════════════════════════════


def prune_skeleton(skeleton: np.ndarray, feature_key: str) -> np.ndarray:
    """
    Remove short spurious branches from a skeleton.
    """
    cfg = POSTPROCESS_CONFIG.get(feature_key, {})
    if not cfg.get("prune_branches", False):
        return skeleton

    min_branch_px = cfg.get("min_branch_px", 10)

    try:
        from skan import Skeleton, summarize
    except ImportError:
        logger.debug("skan not available; skipping skeleton pruning")
        return skeleton

    if skeleton.sum() == 0:
        return skeleton

    try:
        skel_obj = Skeleton(skeleton.astype(bool))
        stats = summarize(skel_obj, find_main_branch=False)

        keep_mask = np.zeros_like(skeleton, dtype=bool)
        for idx, row in stats.iterrows():
            branch_type = row.get("branch-type", 2)
            branch_dist = row.get("branch-distance", 0)
            if branch_type == 2 or branch_dist >= min_branch_px:
                path = skel_obj.path_coordinates(idx)
                for r, c in path.astype(int):
                    if 0 <= r < skeleton.shape[0] and 0 <= c < skeleton.shape[1]:
                        keep_mask[r, c] = True
        return keep_mask.astype(np.uint8)
    except Exception as e:
        logger.warning("Skeleton pruning failed: %s; using original", e)
        return skeleton


# ═══════════════════════════════════════════════════════════════════
# 3. POLYGON REFINEMENT
# ═══════════════════════════════════════════════════════════════════


def _dominant_angle(coords: np.ndarray) -> float:
    if len(coords) < 3:
        return 0.0
    edges = np.diff(coords, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    mask = lengths > 1e-6
    edges = edges[mask]
    lengths = lengths[mask]
    if len(edges) == 0:
        return 0.0
    angles = np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2)
    n_bins = 90
    bins = np.linspace(0, np.pi / 2, n_bins + 1)
    hist, _ = np.histogram(angles, bins=bins, weights=lengths)
    dominant_bin = np.argmax(hist)
    return (bins[dominant_bin] + bins[dominant_bin + 1]) / 2


def _snap_edges_to_angle(
    coords: np.ndarray, dominant: float, snap_tol_deg: float = 5.0
) -> np.ndarray:
    snap_tol = np.radians(snap_tol_deg)
    result = [coords[0].copy()]
    for i in range(1, len(coords)):
        edge = coords[i] - result[-1]
        length = np.linalg.norm(edge)
        if length < 1e-6:
            continue
        angle = np.arctan2(edge[1], edge[0])
        for target in [
            dominant,
            dominant + np.pi / 2,
            dominant + np.pi,
            dominant + 3 * np.pi / 2,
        ]:
            diff = abs(((angle - target + np.pi) % (2 * np.pi)) - np.pi)
            if diff < snap_tol:
                new_edge = np.array([np.cos(target), np.sin(target)]) * length
                result.append(result[-1] + new_edge)
                break
        else:
            result.append(coords[i].copy())
    return np.array(result)


def orthogonalize_polygon(
    poly: Polygon,
    min_rect_area: float = 50.0,
    snap_tol_deg: float = 10.0,
    dominant_angle_override: Optional[float] = None,
    image_tile: Optional[np.ndarray] = None,
) -> Polygon:
    """
    GIS Engineering: SOTA GeoAI Regularization.
    Combines Manhattan partitioning with visual Hough evidence.
    """
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return poly

    # 1. Recursive GeoAI Pass
    # Use OBB for orientation
    if dominant_angle_override is not None:
        angle_rad = dominant_angle_override
    else:
        obb = poly.minimum_rotated_rectangle
        obb_coords = np.array(obb.exterior.coords)
        angle_rad = _dominant_angle(obb_coords)

    geoai = GeoAIRegularizer()
    poly = geoai.regularize(poly, angle_rad, image_tile=image_tile)

    # 2. Hough-Guided Correction (Visual Edge Locking)
    if image_tile is not None:
        hough = HoughRefiner()
        poly = hough.refine_geometry(image_tile, poly)

    return poly
    if poly.area < min_rect_area or len(coords) <= 6:
        try:
            cnt = coords.astype(np.float32).reshape(-1, 1, 2)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            result = Polygon(box)
            if result.is_valid and not result.is_empty:
                return result
        except Exception:
            pass
        return poly
    try:
        dominant = (
            dominant_angle_override
            if dominant_angle_override is not None
            else _dominant_angle(coords)
        )
        snapped = _snap_edges_to_angle(coords, dominant, snap_tol_deg)
        result = Polygon(snapped)
        if result.is_valid and not result.is_empty and result.area > 0:
            return result
    except Exception as e:
        logger.debug("Orthogonalization failed: %s", e)
    return poly


def refine_polygon(
    poly: Polygon, feature_key: str, dominant_angle: Optional[float] = None
) -> Polygon:
    cfg = POSTPROCESS_CONFIG.get(feature_key, {})
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return poly
    if cfg.get("orthogonalize", False):
        try:
            from inference.fer import regularize_polygon_shapely

            poly = regularize_polygon_shapely(poly)
        except Exception as e:
            logger.debug("Advanced FER failed: %s, falling back", e)
            poly = orthogonalize_polygon(
                poly,
                min_rect_area=cfg.get("min_rect_area", 50.0),
                snap_tol_deg=cfg.get("angle_snap_deg", 5.0),
            )
    convex_area = cfg.get("convex_hull_area", 0)
    if convex_area > 0 and poly.area < convex_area:
        hull = poly.convex_hull
        if isinstance(hull, Polygon) and hull.is_valid:
            poly = hull
    return poly


# ═══════════════════════════════════════════════════════════════════
# 4. LINE REFINEMENT (Chaikin smoothing + dead-end snapping)
# ═══════════════════════════════════════════════════════════════════


def _chaikin_smooth(
    coords: np.ndarray, iters: int = 3, keep_ends: bool = True
) -> np.ndarray:
    if len(coords) < 3:
        return coords
    for _ in range(iters):
        new_coords = []
        if keep_ends:
            new_coords.append(coords[0])
        for i in range(len(coords) - 1):
            a, b = coords[i], coords[i + 1]
            q, r = 0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b
            new_coords.extend([q, r])
        if keep_ends:
            new_coords.append(coords[-1])
        coords = np.array(new_coords)
    return coords


def refine_line(geom: LineString, feature_key: str) -> LineString:
    cfg = POSTPROCESS_CONFIG.get(feature_key, {})
    iters = cfg.get("chaikin_iters", 0)
    if iters <= 0:
        return geom
    coords = np.array(geom.coords)
    if len(coords) < 3:
        return geom
    smoothed = _chaikin_smooth(coords, iters=iters, keep_ends=True)
    try:
        result = LineString(smoothed)
        if result.is_valid and not result.is_empty:
            return result
    except Exception:
        pass
    return geom


def snap_line_endpoints(lines: list, feature_key: str) -> list:
    cfg = POSTPROCESS_CONFIG.get(feature_key, {})
    snap_dist = cfg.get("snap_distance", 0)
    if snap_dist <= 0 or len(lines) < 2:
        return lines

    # Professional Road Bridging (Phase 1)
    cleaner = RoadNetworkCleaner(bridge_gap_px=snap_dist)
    return cleaner.bridge_endpoints(lines)


# ═══════════════════════════════════════════════════════════════════
# 5. CRF & ROOF REFINEMENT
# ═══════════════════════════════════════════════════════════════════


def crf_refine(
    prob_map: np.ndarray,
    image_rgb: Optional[np.ndarray] = None,
    n_iters: int = 5,
    pos_w: float = 3.0,
    pos_xy_std: float = 3.0,
    bi_w: float = 5.0,
    bi_xy_std: float = 50.0,
    bi_rgb_std: float = 5.0,
) -> np.ndarray:
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        return prob_map
    h, w = prob_map.shape[:2]
    probs = np.stack([1.0 - prob_map, prob_map], axis=0).astype(np.float32)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    U = unary_from_softmax(probs)
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(
        sxy=pos_xy_std,
        compat=pos_w,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    if image_rgb is not None:
        img = image_rgb.astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        d.addPairwiseBilateral(
            sxy=bi_xy_std,
            srgb=bi_rgb_std,
            rgbim=img,
            compat=bi_w,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC,
        )
    Q = d.inference(n_iters)
    result = np.array(Q).reshape(2, h, w)
    return result[1].astype(np.float32)


def refine_roof_types(
    building_mask: np.ndarray,
    roof_logits: np.ndarray,
    image_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Refine roof predictions per building instance with majority vote + spectral cues."""
    if hasattr(building_mask, "cpu"):
        building_mask = building_mask.cpu().numpy()
    if hasattr(roof_logits, "cpu"):
        roof_logits = roof_logits.cpu().numpy()
    roof_preds = (
        np.argmax(roof_logits, axis=0) if roof_logits.ndim == 3 else roof_logits
    )
    labeled_buildings = label(building_mask > 0.5)
    refined_roofs = np.zeros_like(roof_preds)

    for prop in regionprops(labeled_buildings):
        coords = prop.coords
        rr, cc = coords[:, 0], coords[:, 1]
        building_roofs = roof_preds[rr, cc]
        unique, counts = np.unique(building_roofs, return_counts=True)

        if len(unique) == 0:
            continue

        # Majority vote from NN predictions
        majority_class = unique[np.argmax(counts)]
        majority_pct = counts.max() / counts.sum()

        # Spectral override: if NN confidence is low, use color cues
        if image_rgb is not None and majority_pct < 0.65:
            spectral_class = _classify_roof_by_color(image_rgb, rr, cc)
            if spectral_class is not None:
                majority_class = spectral_class

        refined_roofs[rr, cc] = majority_class

    return refined_roofs


def _classify_roof_by_color(
    image_rgb: np.ndarray, rr: np.ndarray, cc: np.ndarray
) -> Optional[int]:
    """
    Classify roof type using HSV color features of the building pixels.
    Returns class index or None if no confident spectral match.

    Spectral rules (Indian drone orthophotos):
      - Blue/teal hue + medium saturation → Tin/Sheet (class 3)
      - Red/brown hue + medium-high saturation → Tiled (class 2)
      - Low saturation + high brightness → RCC Flat (class 1)
      - Low saturation + low brightness → Incomplete (class 0)
    """
    h, w = image_rgb.shape[:2]
    # Bounds check
    valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    rr, cc = rr[valid], cc[valid]
    if len(rr) < 10:
        return None

    pixels = image_rgb[rr, cc]  # (N, 3)
    if pixels.max() <= 1.0:
        pixels = (pixels * 255).astype(np.uint8)
    else:
        pixels = pixels.astype(np.uint8)

    # Convert to HSV
    pixels_bgr = pixels[:, ::-1].reshape(-1, 1, 3)
    hsv = cv2.cvtColor(pixels_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h_vals = hsv[:, 0].astype(float)  # 0-179
    s_vals = hsv[:, 1].astype(float)  # 0-255
    v_vals = hsv[:, 2].astype(float)  # 0-255

    med_h = np.median(h_vals)
    med_s = np.median(s_vals)
    med_v = np.median(v_vals)

    # Rule 1: Blue/teal/green metal sheets (H: 80-130, S > 40)
    blue_frac = np.mean((h_vals > 80) & (h_vals < 130) & (s_vals > 40))
    if blue_frac > 0.35:
        return 3  # Tin/Sheet

    # Rule 2: Red/brown tiles (H: 0-20 or 160-179, S > 50)
    red_frac = np.mean(((h_vals < 20) | (h_vals > 160)) & (s_vals > 50))
    if red_frac > 0.35:
        return 2  # Tiled (Sloped)

    # Rule 3: Grey/white → RCC Flat (low saturation, high brightness)
    if med_s < 40 and med_v > 140:
        return 1  # RCC (Flat)

    # Rule 4: Dark, low saturation → Incomplete
    if med_s < 35 and med_v < 100:
        return 0  # Incomplete

    return None  # No confident spectral match — keep NN prediction


# ─────────────────────────────────────────────────────────────────────
# Missing Utility Functions for Export Compatibility
# ─────────────────────────────────────────────────────────────────────


def separate_instances(mask: np.ndarray, feature_key: str) -> np.ndarray:
    """
    Improved instance separation using Watershed + gentle seam bridging.
    """
    from skimage.measure import label
    from skimage.segmentation import watershed
    from scipy.ndimage import distance_transform_edt

    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.int32)

    # 1. Gentle weld (2px) - bridges tiles without fusing houses
    mask = bridge_tile_seams(mask, radius=2)
    binary = (mask > 0.5).astype(np.uint8)

    if feature_key != "building_mask":
        return label(binary).astype(np.int32)

    # 2. SMOOTHING: Extinguish micro-peaks from roof texture/rust
    # This ensures a long shed has ONE main peak instead of dozens of strips.
    distance = distance_transform_edt(binary)
    distance = ndimage.gaussian_filter(distance, sigma=3.0)

    from skimage.feature import peak_local_max

    # Find roof peaks (8px = ~0.8m GSD)
    coords = peak_local_max(distance, min_distance=15, labels=binary)
    if len(coords) == 0:
        return label(binary).astype(np.int32)

    mask_seeds = np.zeros(distance.shape, dtype=bool)
    mask_seeds[tuple(coords.T)] = True
    markers, _ = label(mask_seeds), None

    labels = watershed(-distance, markers, mask=binary)
    return labels.astype(np.int32)


def get_global_orientation(mask: np.ndarray) -> float:
    return 0.0
