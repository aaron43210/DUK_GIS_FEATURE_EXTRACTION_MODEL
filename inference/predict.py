"""
Tiled inference engine for DIGITAL UNIVERSITY OF KERALA EXTRACTION MODEL outputs.

Backbone segmentation model:
- SAM2 encoder + multi-head decoder for all raster tasks

Detection model:
- YOLOv8 for sparse point objects (wells/transformers),
  fused into point masks.
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio.windows import Window
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Training-time normalization constants (A.Normalize in data/augmentation.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _repo_search_roots() -> List[Path]:
    roots: List[Path] = []
    for cand in [Path.cwd(), Path(__file__).resolve().parents[1]]:
        if cand.exists() and cand.is_dir() and cand not in roots:
            roots.append(cand)
    return roots


def _discover_local_ultralytics_roots() -> List[Path]:
    """Discover local YOLO/Ultralytics source trees in the workspace."""
    candidates: List[Path] = []
    for root in _repo_search_roots():
        try:
            children = [root] + [d for d in root.iterdir() if d.is_dir()]
        except Exception:
            children = [root]

        for base in children:
            if (
                base / "ultralytics" / "__init__.py"
            ).exists() and base not in candidates:
                candidates.append(base)

            # One extra depth level is enough for typical `*-main` extracts.
            try:
                grand_children = [d for d in base.iterdir() if d.is_dir()]
            except Exception:
                grand_children = []
            for child in grand_children:
                if (child / "ultralytics" / "__init__.py").exists():
                    if child not in candidates:
                        candidates.append(child)

    # Prefer folder names likely to be uploaded sources.
    def _score(path: Path) -> int:
        name = path.name.lower()
        if "yolo8-main" in name or "yolov8-main" in name:
            return 0
        if "ultralytics" in name or "yolo" in name:
            return 1
        return 2

    return sorted(candidates, key=_score)


def _load_yolo_class():
    """Import YOLO class from installed package or a local uploaded source tree."""
    try:
        from ultralytics import YOLO as yolo_cls

        return yolo_cls
    except Exception:
        pass

    for candidate in _discover_local_ultralytics_roots():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        try:
            from ultralytics import YOLO as yolo_cls

            logger.info("Loaded ultralytics from local source tree: %s", candidate)
            return yolo_cls
        except Exception:
            continue

    return None


YOLO = _load_yolo_class()
if YOLO is None:
    warnings.warn(
        "ultralytics not installed and no local ultralytics source tree found. "
        "YOLOv8 features will be disabled."
    )


def _percentile_stretch(
    image: np.ndarray, limits: Tuple[float, float] = (2, 98)
) -> np.ndarray:
    """Robust percentile stretching to match training data/dataset.py."""
    image = image.astype(np.float32)
    vmin, vmax = np.percentile(image, limits)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    image = np.clip(image, vmin, vmax)
    image = (image - vmin) / (vmax - vmin)
    return image.clip(0.0, 1.0)


def _to_rgb(tile: np.ndarray) -> np.ndarray:
    if tile.ndim != 3:
        raise ValueError(f"Expected HxWxC tile, got shape {tile.shape}")
    if tile.shape[2] == 1:
        return np.repeat(tile, 3, axis=2)
    if tile.shape[2] == 2:
        return np.concatenate([tile, tile[:, :, :1]], axis=2)
    if tile.shape[2] > 3:
        return tile[:, :, :3]
    return tile


def _to_yolo_uint8(tile: np.ndarray) -> np.ndarray:
    """
    Convert raw tile pixels (often uint16 remote-sensing) into YOLO-friendly uint8 RGB.
    """
    tile_rgb = _to_rgb(tile)
    tile_norm = _percentile_stretch(tile_rgb)
    return np.clip(tile_norm * 255.0, 0, 255).astype(np.uint8)


def _gaussian_kernel_2d(size: int, sigma: float = 0.0) -> np.ndarray:
    """Create a 2D Gaussian weighting kernel for tile blending."""
    if sigma <= 0:
        sigma = size / 4
    x = np.arange(size) - size / 2 + 0.5
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.max()


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax_np(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(np.clip(x, -50, 50))
    return ex / np.maximum(ex.sum(axis=axis, keepdims=True), 1e-8)


def _box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box and N boxes, format [x1,y1,x2,y2]."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, ix2 - ix1)
    inter_h = np.maximum(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    box_area = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = np.maximum(box_area + boxes_area - inter, 1e-8)
    return inter / union


def _nms_detections(
    detections: List[Dict[str, Any]], iou_threshold: float
) -> List[Dict[str, Any]]:
    """Class-wise NMS over merged tile detections."""
    if not detections:
        return []

    kept: List[Dict[str, Any]] = []
    classes = sorted({int(d.get("class", -1)) for d in detections})
    for cls_id in classes:
        cls_dets = [d for d in detections if int(d.get("class", -1)) == cls_id]
        if not cls_dets:
            continue

        boxes = np.array([d["box"] for d in cls_dets], dtype=np.float32)
        scores = np.array(
            [float(d.get("conf", 0.0)) for d in cls_dets], dtype=np.float32
        )
        order = scores.argsort()[::-1]

        while order.size > 0:
            i = int(order[0])
            kept.append(cls_dets[i])
            if order.size == 1:
                break

            rest = order[1:]
            ious = _box_iou_xyxy(boxes[i], boxes[rest])
            order = rest[ious <= iou_threshold]

    kept.sort(key=lambda d: float(d.get("conf", 0.0)), reverse=True)
    return kept


def _extract_state_dict_from_checkpoint(ckpt: Any) -> Dict[str, Any]:
    """Extract model state dict from common checkpoint layouts."""
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            maybe_state = ckpt.get(key)
            if isinstance(maybe_state, dict):
                return dict(maybe_state)
        return dict(ckpt)
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def _strip_common_state_dict_prefixes(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize state_dict keys to this repo's model naming.
    Handles DataParallel and wrapper prefixes.
    """
    fixed = state_dict
    for prefix in ("module.", "model."):
        if fixed and all(k.startswith(prefix) for k in fixed.keys()):
            fixed = {k[len(prefix) :]: v for k, v in fixed.items()}
    return fixed


class TiledPredictor:
    """
    End-to-end tiled predictor for segmentation + point detection.
    """

    BINARY_MODEL_KEYS: List[str] = [
        "building_mask",
        "road_mask",
        "road_centerline_mask",
        "waterbody_mask",
        "waterbody_line_mask",
        "waterbody_point_mask",
        "utility_line_mask",
        "utility_point_mask",
        "bridge_mask",
        "railway_mask",
    ]
    ROOF_KEY = "roof_type_mask"
    POINT_KEYS: Set[str] = {
        "waterbody_point_mask",  # Wells
        "utility_point_mask",
        "utility_transformer_mask",
    }
    ALL_OUTPUT_KEYS: List[str] = BINARY_MODEL_KEYS + [
        ROOF_KEY,
        "utility_transformer_mask",
    ]

    # YOLO class IDs -> target point masks
    YOLO_CLASS_TO_MASK = {
        0: "waterbody_point_mask",  # Wells
        1: "utility_transformer_mask",  # Transformers
    }
    YOLO_LABELS = {
        0: "Well",
        1: "Transformer",
    }

    def __init__(
        self,
        model: nn.Module,
        yolo_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        tile_size: int = 512,
        overlap: int = 128,
        threshold: float = 0.5,
        yolo_conf: float = 0.25,
        point_radius_px: int = 5,
        yolo_iou: float = 0.45,
        yolo_min_area: float = 9.0,
        use_tta: bool = False,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.threshold = threshold
        self.yolo_conf = yolo_conf
        self.point_radius_px = point_radius_px
        self.yolo_iou = yolo_iou
        self.yolo_min_area = yolo_min_area
        self.use_tta = use_tta

        self.yolo = None
        if yolo_path and YOLO:
            try:
                self.yolo = YOLO(yolo_path)
                self.yolo.to(device)
                logger.info("Loaded YOLO detector from %s", yolo_path)
            except Exception as e:
                logger.warning("Failed to load YOLO model at %s: %s", yolo_path, e)

        self.blend_kernel = _gaussian_kernel_2d(tile_size).astype(np.float32)
        self._tta_flip_dims: Sequence[Tuple[int, ...]] = (
            [(), (3,), (2,), (2, 3)] if self.use_tta else [()]
        )

    def _get_valid_mask(self, tif_path: Path) -> np.ndarray:
        """Build a coarse valid-data mask from a raster thumbnail."""
        try:
            with rasterio.open(str(tif_path)) as src:
                h, w = src.height, src.width
                scale = min(1024.0 / max(h, w), 1.0)
                th = max(1, int(h * scale))
                tw = max(1, int(w * scale))
                thumb = src.read(
                    out_shape=(src.count, th, tw),
                    resampling=rasterio.enums.Resampling.bilinear,
                )
                return np.any(thumb > 0, axis=0)
        except Exception as e:
            logger.warning("Thumbnail scan failed: %s", e)
            return np.ones((1, 1), dtype=bool)

    def _normalize_tile(self, tile: np.ndarray) -> torch.Tensor:
        """
        Convert tile to normalized RGB float32 tensor input.
        Matches training preprocessing:
        percentile stretch to [0,1] -> ImageNet mean/std normalization.
        """
        image = _to_rgb(tile)

        # 1) Robust percentile stretch (matches dataset.py)
        image = _percentile_stretch(image)
        # 2) ImageNet normalization (matches albumentations A.Normalize)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # (H, W, C) -> (C, H, W)
        img_t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        return img_t.unsqueeze(0).to(self.device)

    def _forward_model_tta(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        merged: Dict[str, torch.Tensor] = {}
        n_augs = len(self._tta_flip_dims)

        for dims in self._tta_flip_dims:
            aug_in = torch.flip(tensor, dims=dims) if dims else tensor
            out = self.model(aug_in, task="all")
            for key, val in out.items():
                if not isinstance(val, torch.Tensor):
                    continue
                restored = torch.flip(val, dims=dims) if dims else val
                if key in merged:
                    merged[key] = merged[key] + restored
                else:
                    merged[key] = restored.clone()

        for key in list(merged.keys()):
            merged[key] = merged[key] / float(n_augs)
        return merged

    def _predict_tile_model(self, tile_img: np.ndarray) -> Dict[str, np.ndarray]:
        """Run the segmentation model for one tile and return numpy outputs."""
        tensor = self._normalize_tile(tile_img)
        with torch.no_grad():
            outputs = self._forward_model_tta(tensor)

        out_np: Dict[str, np.ndarray] = {}
        for key, val in outputs.items():
            arr = val.detach().cpu().numpy()
            if arr.ndim >= 3:
                arr = arr[0]  # drop batch dim
            out_np[key] = arr
        return out_np

    def _run_yolo_tile(
        self,
        tile_img: np.ndarray,
        x0: int,
        y0: int,
        tw_act: int,
        th_act: int,
        selected_point_keys: Set[str],
    ) -> List[Dict[str, Any]]:
        if self.yolo is None or not selected_point_keys:
            return []

        tile_crop = tile_img[:th_act, :tw_act]
        if tile_crop.size == 0:
            return []

        tile_u8 = _to_yolo_uint8(tile_crop)
        detections: List[Dict[str, Any]] = []

        try:
            yolo_results = self.yolo.predict(
                tile_u8,
                conf=self.yolo_conf,
                iou=self.yolo_iou,
                imgsz=max(tw_act, th_act),
                verbose=False,
            )
        except Exception as e:
            logger.warning("YOLO tile inference failed at (%d,%d): %s", x0, y0, e)
            return []

        for res in yolo_results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                try:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().tolist()
                except Exception:
                    continue

                mask_key = self.YOLO_CLASS_TO_MASK.get(cls_id)
                if mask_key not in selected_point_keys:
                    continue

                # Constrain to current crop bounds before global mapping.
                bx1 = float(np.clip(bx1, 0, tw_act))
                bx2 = float(np.clip(bx2, 0, tw_act))
                by1 = float(np.clip(by1, 0, th_act))
                by2 = float(np.clip(by2, 0, th_act))
                if bx2 <= bx1 or by2 <= by1:
                    continue

                area = (bx2 - bx1) * (by2 - by1)
                if area < self.yolo_min_area:
                    continue

                gx1, gy1 = bx1 + x0, by1 + y0
                gx2, gy2 = bx2 + x0, by2 + y0
                detections.append(
                    {
                        "box": [gx1, gy1, gx2, gy2],
                        "class": cls_id,
                        "label": self.YOLO_LABELS.get(cls_id, "Unknown"),
                        "conf": conf,
                        "mask_key": mask_key,
                    }
                )
        return detections

    def _detections_to_point_masks(
        self,
        detections: List[Dict[str, Any]],
        h: int,
        w: int,
        selected: Set[str],
    ) -> Dict[str, np.ndarray]:
        point_masks = {
            key: np.zeros((h, w), dtype=np.float32)
            for key in self.POINT_KEYS
            if key in selected
        }
        if not point_masks:
            return point_masks

        try:
            import cv2
        except Exception:
            cv2 = None  # type: ignore

        for det in detections:
            mask_key = det.get("mask_key")
            if mask_key not in point_masks:
                continue
            x1, y1, x2, y2 = det["box"]
            conf = float(np.clip(det.get("conf", 1.0), 0.0, 1.0))
            cx = int(round((x1 + x2) * 0.5))
            cy = int(round((y1 + y2) * 0.5))
            radius = max(self.point_radius_px, int(max(x2 - x1, y2 - y1) * 0.25))

            if cv2 is not None:
                tmp = np.zeros_like(point_masks[mask_key])
                cv2.circle(tmp, (cx, cy), radius, color=conf, thickness=-1)
                point_masks[mask_key] = np.maximum(point_masks[mask_key], tmp)
            else:
                if 0 <= cy < h and 0 <= cx < w:
                    point_masks[mask_key][cy, cx] = max(
                        point_masks[mask_key][cy, cx], conf
                    )

        return point_masks

    @torch.no_grad()
    def predict_tif(
        self,
        tif_path: Path,
        selected_masks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run full tiled inference for all requested outputs."""
        selected = set(selected_masks or self.ALL_OUTPUT_KEYS)
        selected = {key for key in selected if key in set(self.ALL_OUTPUT_KEYS)}
        if not selected:
            selected = set(self.ALL_OUTPUT_KEYS)

        selected_point_keys = self.POINT_KEYS & selected
        valid_thumb = self._get_valid_mask(tif_path)

        with rasterio.open(str(tif_path)) as src:
            h, w = src.height, src.width
            logger.info("Predicting %s (%dx%d)", tif_path.name, w, h)

            th_h, th_w = valid_thumb.shape
            scale_y, scale_x = th_h / h, th_w / w

            model_accum = {
                key: np.zeros((h, w), dtype=np.float32)
                for key in self.BINARY_MODEL_KEYS
                if key in selected
            }
            roof_accum = (
                np.zeros((5, h, w), dtype=np.float32)
                if self.ROOF_KEY in selected
                else None
            )
            raw_detections: List[Dict[str, Any]] = []
            weight_map = np.zeros((h, w), dtype=np.float32)

            stride = max(1, self.tile_size - self.overlap)
            windows = []
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    tw = min(self.tile_size, w - x)
                    th = min(self.tile_size, h - y)
                    tx, ty = int(x * scale_x), int(y * scale_y)
                    if not valid_thumb[min(ty, th_h - 1), min(tx, th_w - 1)]:
                        continue
                    windows.append((x, y, tw, th))

            for x0, y0, tw_act, th_act in tqdm(windows, desc="Inference", leave=False):
                win = Window(x0, y0, self.tile_size, self.tile_size)
                part = src.read(
                    window=win,
                    boundless=True,
                    fill_value=0,
                )
                tile_img = np.transpose(part, (1, 2, 0))
                tile_valid = np.any(tile_img[:th_act, :tw_act] > 0, axis=2)
                if tile_valid.size == 0 or float(tile_valid.mean()) < 0.01:
                    continue
                blend = self.blend_kernel[:th_act, :tw_act]
                weight_map[y0 : y0 + th_act, x0 : x0 + tw_act] += blend

                # SAM2 multi-task segmentation
                model_outputs = self._predict_tile_model(tile_img)

                for key in list(model_accum.keys()):
                    if key not in model_outputs:
                        continue
                    logits = model_outputs[key]
                    if logits.ndim == 3 and logits.shape[0] == 1:
                        logits2d = logits[0]
                    elif logits.ndim == 2:
                        logits2d = logits
                    else:
                        continue
                    prob = _sigmoid_np(logits2d[:th_act, :tw_act])
                    model_accum[key][y0 : y0 + th_act, x0 : x0 + tw_act] += prob * blend

                if roof_accum is not None and self.ROOF_KEY in model_outputs:
                    roof_logits = model_outputs[self.ROOF_KEY]
                    if roof_logits.ndim == 3 and roof_logits.shape[0] >= 2:
                        roof_probs = _softmax_np(
                            roof_logits[:, :th_act, :tw_act], axis=0
                        )
                        roof_accum[:, y0 : y0 + th_act, x0 : x0 + tw_act] += (
                            roof_probs * blend[None]
                        )

                # YOLO point detection (wells / transformers)
                if self.yolo is not None and selected_point_keys:
                    raw_detections.extend(
                        self._run_yolo_tile(
                            tile_img=tile_img,
                            x0=x0,
                            y0=y0,
                            tw_act=tw_act,
                            th_act=th_act,
                            selected_point_keys=selected_point_keys,
                        )
                    )

        weight_map = np.maximum(weight_map, 1e-8)
        final_results: Dict[str, Any] = {}

        # Finalize binary probability masks
        for key, accum in model_accum.items():
            final_results[key] = accum / weight_map

        # Finalize roof class map
        if roof_accum is not None:
            roof_probs = roof_accum / weight_map[None]
            roof_mask = np.argmax(roof_probs, axis=0).astype(np.uint8)
            if "building_mask" in final_results:
                roof_mask[final_results["building_mask"] <= self.threshold] = 0
            final_results[self.ROOF_KEY] = roof_mask

        detections = _nms_detections(raw_detections, iou_threshold=self.yolo_iou)
        det_point_masks = self._detections_to_point_masks(detections, h, w, selected)

        # Fuse YOLO points with segmentation point masks
        for point_key in self.POINT_KEYS:
            if point_key not in selected:
                continue
            seg_prob = final_results.get(point_key)
            det_prob = det_point_masks.get(point_key)
            if det_prob is not None and seg_prob is not None:
                final_results[point_key] = np.maximum(seg_prob, det_prob)
            elif det_prob is not None:
                final_results[point_key] = det_prob
            elif seg_prob is None:
                final_results[point_key] = np.zeros((h, w), dtype=np.float32)

        # Fill any missing requested output key with zeros
        for key in selected:
            if key not in final_results:
                dtype = np.uint8 if key == self.ROOF_KEY else np.float32
                final_results[key] = np.zeros((h, w), dtype=dtype)

        final_results["detections"] = detections
        return final_results

    @torch.no_grad()
    def predict_image(
        self,
        image_path: Path,
        selected_masks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run inference on a standard image (JPG, PNG, JPEG, BMP).

        Falls back to predict_tif for GeoTIFF formats.
        Works by loading via PIL, tiling, and running the
        same model pipeline.
        """
        ext = image_path.suffix.lower()
        if ext in {".tif", ".tiff"}:
            return self.predict_tif(image_path, selected_masks)

        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "PIL (Pillow) is required for JPG/PNG support. "
                "Install with: pip install Pillow"
            )

        selected = set(selected_masks or self.ALL_OUTPUT_KEYS)
        selected = {k for k in selected if k in set(self.ALL_OUTPUT_KEYS)}
        if not selected:
            selected = set(self.ALL_OUTPUT_KEYS)

        selected_point_keys = self.POINT_KEYS & selected

        # Load image as numpy array
        pil_img = Image.open(image_path).convert("RGB")
        full_image = np.array(pil_img, dtype=np.float32)
        # Normalize to [0, 1] if uint8
        if full_image.max() > 1.0:
            full_image = full_image / 255.0

        h, w = full_image.shape[:2]
        logger.info(
            "Predicting %s (%dx%d, format=%s)",
            image_path.name,
            w,
            h,
            ext,
        )

        model_accum: Dict[str, np.ndarray] = {
            key: np.zeros((h, w), dtype=np.float32)
            for key in self.BINARY_MODEL_KEYS
            if key in selected
        }
        roof_accum = (
            np.zeros((5, h, w), dtype=np.float32) if self.ROOF_KEY in selected else None
        )
        raw_detections: List[Dict[str, Any]] = []
        weight_map = np.zeros((h, w), dtype=np.float32)

        stride = max(1, self.tile_size - self.overlap)
        windows = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                tw = min(self.tile_size, w - x)
                th_act = min(self.tile_size, h - y)
                windows.append((x, y, tw, th_act))

        for x0, y0, tw_act, th_act in tqdm(
            windows,
            desc="Inference",
            leave=False,
        ):
            # Crop tile
            tile_img = np.zeros(
                (self.tile_size, self.tile_size, 3),
                dtype=np.float32,
            )
            crop = full_image[y0 : y0 + th_act, x0 : x0 + tw_act]
            tile_img[:th_act, :tw_act] = crop

            # Skip empty tiles
            if float(crop.mean()) < 0.01:
                continue

            blend = self.blend_kernel[:th_act, :tw_act]
            weight_map[y0 : y0 + th_act, x0 : x0 + tw_act] += blend

            # Model prediction
            model_outputs = self._predict_tile_model(tile_img)

            for key in list(model_accum.keys()):
                if key not in model_outputs:
                    continue
                logits = model_outputs[key]
                if logits.ndim == 3 and logits.shape[0] == 1:
                    logits2d = logits[0]
                elif logits.ndim == 2:
                    logits2d = logits
                else:
                    continue
                prob = _sigmoid_np(logits2d[:th_act, :tw_act])
                model_accum[key][y0 : y0 + th_act, x0 : x0 + tw_act] += (
                    prob * blend
                )

            if roof_accum is not None:
                roof_key = self.ROOF_KEY
                if roof_key in model_outputs:
                    roof_logits = model_outputs[roof_key]
                    for c in range(min(5, roof_logits.shape[0])):
                        roof_accum[c][
                            y0 : y0 + th_act,
                            x0 : x0 + tw_act,
                        ] += (
                            roof_logits[c][:th_act, :tw_act] * blend
                        )

            # YOLO point detection
            if self.yolo is not None and selected_point_keys:
                raw_detections.extend(
                    self._run_yolo_tile(
                        tile_img=tile_img,
                        x0=x0,
                        y0=y0,
                        tw_act=tw_act,
                        th_act=th_act,
                        selected_point_keys=selected_point_keys,
                    )
                )

        # Normalize by weight map
        safe_w = np.maximum(weight_map, 1e-6)
        final_results: Dict[str, Any] = {}

        for key, accum in model_accum.items():
            final_results[key] = (accum / safe_w >= self.threshold).astype(np.uint8)

        if roof_accum is not None:
            for c in range(5):
                roof_accum[c] /= safe_w
            roof_mask = np.argmax(_softmax_np(roof_accum, axis=0), axis=0).astype(
                np.uint8
            )
            if "building_mask" in final_results:
                roof_mask[final_results["building_mask"] <= self.threshold] = 0
            final_results[self.ROOF_KEY] = roof_mask

        detections = _nms_detections(raw_detections, iou_threshold=self.yolo_iou)
        det_point_masks = self._detections_to_point_masks(
            detections,
            h,
            w,
            selected,
        )

        for point_key in self.POINT_KEYS:
            if point_key not in selected:
                continue
            seg_prob = final_results.get(point_key)
            det_prob = det_point_masks.get(point_key)
            if det_prob is not None and seg_prob is not None:
                final_results[point_key] = np.maximum(
                    seg_prob,
                    det_prob,
                )
            elif det_prob is not None:
                final_results[point_key] = det_prob
            elif seg_prob is None:
                final_results[point_key] = np.zeros(
                    (h, w),
                    dtype=np.float32,
                )

        for key in selected:
            if key not in final_results:
                dtype = np.uint8 if key == self.ROOF_KEY else np.float32
                final_results[key] = np.zeros(
                    (h, w),
                    dtype=dtype,
                )

        final_results["detections"] = detections
        return final_results


def _resolve_weights_path(weights_path: str) -> Optional[Path]:
    p = Path(weights_path)
    if p.exists():
        return p
    fallback_candidates = [
        Path("checkpoints/best.pt"),
        Path("checkpoints/best_latest.pt"),
    ]
    for cand in fallback_candidates:
        if cand.exists():
            logger.warning("Weights not found at %s; using fallback %s", p, cand)
            return cand
    logger.warning("No ensemble weights found at %s or fallback paths.", p)
    return None


def _resolve_yolo_path(yolo_path: Optional[str]) -> Optional[str]:
    if yolo_path:
        p = Path(yolo_path)
        if p.exists():
            return str(p)
        logger.warning("YOLO weights not found at %s", p)

    fallback_candidates = [
        Path("checkpoints/yolov8s.pt"),
        Path("checkpoints/yolov8n.pt"),
    ]
    for cand in fallback_candidates:
        if cand.exists():
            logger.info("Using fallback YOLO weights: %s", cand)
            return str(cand)

    # If no local files found, return the model name to trigger auto-download by ultralytics
    logger.info(
        "YOLO weights not found locally; triggering auto-download for yolov8s.pt"
    )
    return "yolov8s.pt"


def load_ensemble_pipeline(
    weights_path: str,
    yolo_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
    use_tta: bool = False,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.45,
    tile_size: int = 512,
    overlap: int = 128,
) -> TiledPredictor:
    from models.model import EnsembleDUKModel

    model = EnsembleDUKModel(pretrained=True)
    
    # Safely check if this is the SAM2 encoder
    has_trunk = False
    if hasattr(model.encoder, "encoder") and hasattr(model.encoder.encoder, "trunk"):
        has_trunk = True
        
    if not has_trunk:
        logger.warning(
            "SAM2 encoder trunk not detected. "
            "Assuming SegFormer or fallback backbone is active."
        )

    resolved_weights = _resolve_weights_path(weights_path)
    if resolved_weights is None:
        raise FileNotFoundError(
            "No segmentation checkpoint found. "
            "Set `ckpt_path` to a valid trained weights file (for example "
            "`checkpoints/best.pt`)."
        )

    ckpt = torch.load(resolved_weights, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict_from_checkpoint(ckpt)
    state_dict = _strip_common_state_dict_prefixes(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)

    total_model_keys = len(model.state_dict())
    missing_count = len(incompatible.missing_keys)
    loaded_count = max(total_model_keys - missing_count, 0)
    loaded_ratio = loaded_count / max(total_model_keys, 1)

    if loaded_ratio < 0.80:
        missing_preview = ", ".join(incompatible.missing_keys[:5]) or "none"
        unexpected_preview = ", ".join(incompatible.unexpected_keys[:5]) or "none"
        raise RuntimeError(
            "Checkpoint is incompatible with current model architecture. "
            f"Loaded {loaded_count}/{total_model_keys} keys ({loaded_ratio:.1%}). "
            "This often happens when SAM2 is not installed or when checkpoint/model "
            "configs differ. "
            f"Missing sample: [{missing_preview}] | "
            f"Unexpected sample: [{unexpected_preview}]"
        )

    if incompatible.missing_keys or incompatible.unexpected_keys:
        logger.warning(
            "Loaded checkpoint with partial key mismatch: "
            "%d missing, %d unexpected (loaded %.1f%% of model keys).",
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
            loaded_ratio * 100.0,
        )
    logger.info("Loaded ensemble weights from %s", resolved_weights)

    resolved_yolo = _resolve_yolo_path(yolo_path) if YOLO is not None else None
    return TiledPredictor(
        model=model,
        yolo_path=resolved_yolo,
        device=device,
        use_tta=use_tta,
        yolo_conf=yolo_conf,
        yolo_iou=yolo_iou,
        tile_size=tile_size,
        overlap=overlap,
    )
