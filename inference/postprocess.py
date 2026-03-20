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
from typing import Dict, Optional

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Polygon
from skimage.morphology import binary_closing, disk

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Per-feature configuration for post-processing parameters
# ─────────────────────────────────────────────────────────────────────
POSTPROCESS_CONFIG: Dict[str, dict] = {
    "building_mask": {
        "closing_radius": 3,
        "fill_holes": True,
        "orthogonalize": True,
        "min_rect_area": 50.0,  # m² — below this, use minAreaRect
        "angle_snap_deg": 5.0,  # snap edges within ±5° of dominant angle
        "threshold": 0.45,  # slightly lower for better recall
        "min_area_px": 50,  # minimum pixels to keep
    },
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
        "prune_branches": True,
        "min_branch_px": 15,
        "chaikin_iters": 3,
        "snap_distance": 10.0,  # pixels
        "threshold": 0.45,
        "min_area_px": 30,  # small segments of noise
    },
    "waterbody_mask": {
        "closing_radius": 9,
        "fill_holes": True,
        "orthogonalize": False,
        "convex_hull_area": 100.0,  # small ponds → convex hull
        "threshold": 0.50,
        "min_area_px": 80,
    },
    "waterbody_line_mask": {
        "closing_radius": 5,
        "fill_holes": False,
        "prune_branches": True,
        "min_branch_px": 10,
        "chaikin_iters": 3,
        "snap_distance": 8.0,
        "threshold": 0.45,
        "min_area_px": 20,
    },
    "waterbody_point_mask": {
        "threshold": 0.40,
        "min_area_px": 5,  # points are tiny, but should have some mass
    },
    "utility_line_mask": {
        "closing_radius": 5,
        "fill_holes": False,
        "prune_branches": True,
        "min_branch_px": 10,
        "chaikin_iters": 3,
        "snap_distance": 8.0,
        "threshold": 0.45,
        "min_area_px": 20,
    },
    "utility_point_mask": {
        "threshold": 0.40,
        "min_area_px": 5,
    },
    "bridge_mask": {
        "closing_radius": 3,
        "fill_holes": True,
        "orthogonalize": True,
        "min_rect_area": 30.0,
        "angle_snap_deg": 5.0,
        "threshold": 0.50,
        "min_area_px": 40,
    },
    "railway_mask": {
        "closing_radius": 5,
        "fill_holes": False,
        "prune_branches": True,
        "min_branch_px": 20,
        "chaikin_iters": 3,
        "snap_distance": 15.0,
        "threshold": 0.50,
        "min_area_px": 50,
    },
}


def get_threshold(feature_key: str) -> float:
    """Return the optimal per-class threshold for a feature."""
    cfg = POSTPROCESS_CONFIG.get(feature_key, {})
    return float(cfg.get("threshold", 0.5))


# ═══════════════════════════════════════════════════════════════════
# 1. MASK-LEVEL REFINEMENT  (operates on binary masks)
# ═══════════════════════════════════════════════════════════════════


def refine_mask(mask: np.ndarray, feature_key: str) -> np.ndarray:
    """
    Apply morphological closing and hole-filling to a binary mask
    based on the feature type.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (uint8, values 0/1).
    feature_key : str
        Key from FEATURE_CONFIG, e.g. 'building_mask'.

    Returns
    -------
    np.ndarray
        Refined binary mask.
    """
    cfg = POSTPROCESS_CONFIG.get(feature_key, {})
    closing_radius = cfg.get("closing_radius", 0)
    fill_holes = cfg.get("fill_holes", False)

    if closing_radius > 0:
        selem = disk(closing_radius)
        mask = binary_closing(mask.astype(bool), selem).astype(np.uint8)

    if fill_holes:
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    # ── Stage 2: Connected Component Area Filtering ──
    # Eliminates "salt and pepper" noise
    min_area = cfg.get("min_area_px", 0)
    if min_area > 0 and mask.sum() > 0:
        from skimage.measure import label, regionprops
        
        labeled = label(mask)
        refined = np.zeros_like(mask)
        for prop in regionprops(labeled):
            if prop.area >= min_area:
                refined[labeled == prop.label] = 1
        mask = refined

    return mask


# ═══════════════════════════════════════════════════════════════════
# 2. SKELETON PRUNING  (removes spurious branches)
# ═══════════════════════════════════════════════════════════════════


def prune_skeleton(skeleton: np.ndarray, feature_key: str) -> np.ndarray:
    """
    Remove short spurious branches from a skeleton image using
    skan branch statistics. Falls back to no-op if skan unavailable.

    Parameters
    ----------
    skeleton : np.ndarray
        Binary skeleton image (output of skimage.morphology.skeletonize).
    feature_key : str
        Feature key to look up min_branch_px.

    Returns
    -------
    np.ndarray
        Pruned skeleton.
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

        # Keep branches longer than threshold, or junction-junction branches
        keep_mask = np.zeros_like(skeleton, dtype=bool)
        for idx, row in stats.iterrows():
            branch_type = row.get("branch-type", 2)
            branch_dist = row.get("branch-distance", 0)

            # branch-type: 0 = endpoint-endpoint, 1 = junction-endpoint,
            #              2 = junction-junction, 3 = isolated
            # Keep junction-junction always; prune short tip branches
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
    """
    Find the dominant edge orientation angle in a polygon using
    edge-length-weighted angle histogram.
    Returns angle in radians (0 to π/2).
    """
    if len(coords) < 3:
        return 0.0

    edges = np.diff(coords, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    mask = lengths > 1e-6
    edges = edges[mask]
    lengths = lengths[mask]

    if len(edges) == 0:
        return 0.0

    # Angles mod π/2 (we only care about the main axis)
    angles = np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2)

    # Weighted histogram with 180 bins (1-degree resolution in π/2 range)
    n_bins = 90
    bins = np.linspace(0, np.pi / 2, n_bins + 1)
    hist, _ = np.histogram(angles, bins=bins, weights=lengths)

    dominant_bin = np.argmax(hist)
    return (bins[dominant_bin] + bins[dominant_bin + 1]) / 2


def _snap_edges_to_angle(
    coords: np.ndarray, dominant: float, snap_tol_deg: float = 5.0
) -> np.ndarray:
    """
    Snap polygon edges to be aligned with the dominant angle or
    perpendicular to it (±snap_tol_deg tolerance).
    """
    snap_tol = np.radians(snap_tol_deg)
    result = [coords[0].copy()]

    for i in range(1, len(coords)):
        edge = coords[i] - result[-1]
        length = np.linalg.norm(edge)
        if length < 1e-6:
            continue

        angle = np.arctan2(edge[1], edge[0])

        # Check if close to dominant or dominant + π/2
        for target in [
            dominant,
            dominant + np.pi / 2,
            dominant + np.pi,
            dominant + 3 * np.pi / 2,
        ]:
            diff = abs(((angle - target + np.pi) % (2 * np.pi)) - np.pi)
            if diff < snap_tol:
                # Snap to target angle
                new_edge = np.array([np.cos(target), np.sin(target)]) * length
                result.append(result[-1] + new_edge)
                break
        else:
            result.append(coords[i].copy())

    return np.array(result)


def orthogonalize_polygon(
    poly: Polygon, min_rect_area: float = 50.0, snap_tol_deg: float = 5.0
) -> Polygon:
    """
    Orthogonalize a building/bridge polygon:
    1. For small polygons: use minimum rotated rectangle.
    2. For larger polygons: detect dominant angle and snap edges.

    This implements the approach from Schuegraf et al. 2024 (ISPRS)
    in a simplified form suitable for real-time export.
    """
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return poly

    coords = np.array(poly.exterior.coords)

    # Small polygon → clean rectangle
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

    # Larger polygon → dominant-angle snapping
    try:
        dominant = _dominant_angle(coords)
        snapped = _snap_edges_to_angle(coords, dominant, snap_tol_deg)
        result = Polygon(snapped)
        if result.is_valid and not result.is_empty and result.area > 0:
            return result
    except Exception as e:
        logger.debug("Orthogonalization failed: %s", e)

    return poly


def refine_polygon(poly: Polygon, feature_key: str) -> Polygon:
    """
    Apply feature-specific polygon refinement.

    - Buildings/Bridges: orthogonalize
    - Waterbodies: smooth + convex hull for tiny ponds
    - Roads: no extra polygon refinement (DP simplify already in export.py)
    """
    cfg = POSTPROCESS_CONFIG.get(feature_key, {})

    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return poly

    # Orthogonalization for man-made structures
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

    # Convex hull fallback for tiny natural features
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
    """
    Chaikin corner-cutting algorithm for line smoothing.
    Each iteration replaces each segment A→B with two points
    at 1/4 and 3/4 along the segment.
    """
    if len(coords) < 3:
        return coords

    for _ in range(iters):
        new_coords = []
        if keep_ends:
            new_coords.append(coords[0])

        for i in range(len(coords) - 1):
            a, b = coords[i], coords[i + 1]
            q = 0.75 * a + 0.25 * b
            r = 0.25 * a + 0.75 * b
            new_coords.extend([q, r])

        if keep_ends:
            new_coords.append(coords[-1])

        coords = np.array(new_coords)

    return coords


def refine_line(geom: LineString, feature_key: str) -> LineString:
    """
    Apply Chaikin smoothing to a LineString geometry.
    """
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
    """
    Connect dangling endpoints (dead ends) of LineString geometries
    that are within snap_distance of each other.

    Uses a simple nearest-endpoint merge: for each degree-1 endpoint,
    find the closest degree-1 endpoint from a different line and
    connect them if within threshold.
    """
    cfg = POSTPROCESS_CONFIG.get(feature_key, {})
    snap_dist = cfg.get("snap_distance", 0)

    if snap_dist <= 0 or len(lines) < 2:
        return lines

    # Collect all endpoints
    endpoints = []  # (line_idx, end: 'start'|'end', point)
    for i, line in enumerate(lines):
        if not isinstance(line, LineString) or line.is_empty:
            continue
        coords = list(line.coords)
        if len(coords) >= 2:
            endpoints.append((i, "start", np.array(coords[0])))
            endpoints.append((i, "end", np.array(coords[-1])))

    if len(endpoints) < 2:
        return lines

    # For each endpoint, find nearest endpoint from a different line
    connected = set()
    new_lines = list(lines)

    for idx_a in range(len(endpoints)):
        if idx_a in connected:
            continue

        line_a, end_a, pt_a = endpoints[idx_a]
        best_dist = snap_dist
        best_idx = -1

        for idx_b in range(idx_a + 1, len(endpoints)):
            if idx_b in connected:
                continue
            line_b, end_b, pt_b = endpoints[idx_b]
            if line_b == line_a:
                continue

            dist = np.linalg.norm(pt_a - pt_b)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx_b

        if best_idx >= 0:
            _, _, pt_b = endpoints[best_idx]
            # Create a short connecting line
            connector = LineString([pt_a, pt_b])
            new_lines.append(connector)
            connected.add(idx_a)
            connected.add(best_idx)

    return new_lines


# ═══════════════════════════════════════════════════════════════════
# 5. CRF PROBABILITY MAP REFINEMENT  (optional, requires pydensecrf)
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
    """
    Apply Dense CRF to refine a probability map using image appearance
    as a guide. Sharpens fuzzy boundaries to align with visual edges.

    Falls back to returning the original prob_map if pydensecrf is
    not installed.

    Parameters
    ----------
    prob_map : np.ndarray
        Float32 probability map, shape (H, W), values in [0, 1].
    image_rgb : np.ndarray, optional
        Original image tile as uint8 RGB, shape (H, W, 3).
        If None, only position-based CRF is applied.
    n_iters : int
        Number of CRF inference iterations.

    Returns
    -------
    np.ndarray
        Refined probability map, shape (H, W), float32 in [0, 1].
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        return prob_map

    h, w = prob_map.shape[:2]

    # Build 2-class softmax: [bg, fg]
    probs = np.stack([1.0 - prob_map, prob_map], axis=0).astype(np.float32)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

    U = unary_from_softmax(probs)
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(U)

    # Position-based pairwise (smoothness)
    d.addPairwiseGaussian(
        sxy=pos_xy_std,
        compat=pos_w,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    # Appearance-based pairwise (if image available)
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
    return result[1].astype(np.float32)  # foreground probability
