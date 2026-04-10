"""
Tiled inference engine for DIGITAL UNIVERSITY KERALA FEATURE EXTRACTION MODEL V1 outputs.

Backbone segmentation model:
- SegFormer encoder + multi-head decoder for all raster tasks

Detection model:
- YOLOv8 for sparse point objects (wells/transformers),
  fused into point masks.
"""

import logging
import sys
import warnings
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio.windows import Window
from tqdm import tqdm

# from inference.zero_shot import ZeroShotAssistant

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

            try:
                grand_children = [d for d in base.iterdir() if d.is_dir()]
            except Exception:
                grand_children = []
            for child in grand_children:
                if (child / "ultralytics" / "__init__.py").exists():
                    if child not in candidates:
                        candidates.append(child)

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


def _create_spline_window(size: int, overlap: int, power: int = 2) -> np.ndarray:
    """GeoAI Industry Standard: Raised-cosine spline window for seamless blending."""
    if overlap <= 0: return np.ones((size, size), dtype=np.float32)
    
    # 1D Spline
    window = np.ones(size, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, overlap + 1, endpoint=True)[1:]
    ramp = (0.5 * (1.0 - np.cos(np.pi * ramp))) ** power
    window[:overlap] = ramp
    window[-overlap:] = ramp[::-1]
    
    # 2D Spline
    return np.outer(window, window)

def _d4_forward(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Apply all 8 D4 dihedral transforms (Rotations + Flips)."""
    return [
        tensor,
        torch.rot90(tensor, k=1, dims=[-2, -1]),
        torch.rot90(tensor, k=2, dims=[-2, -1]),
        torch.rot90(tensor, k=3, dims=[-2, -1]),
        torch.flip(tensor, dims=[-1]),
        torch.flip(tensor, dims=[-2]),
        torch.flip(torch.rot90(tensor, k=1, dims=[-2, -1]), dims=[-1]),
        torch.flip(torch.rot90(tensor, k=1, dims=[-2, -1]), dims=[-2]),
    ]

def _d4_inverse(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """Invert the 8 D4 transforms to original orientation."""
    return [
        tensors[0],
        torch.rot90(tensors[1], k=3, dims=[-2, -1]),
        torch.rot90(tensors[2], k=2, dims=[-2, -1]),
        torch.rot90(tensors[3], k=1, dims=[-2, -1]),
        torch.flip(tensors[4], dims=[-1]),
        torch.flip(tensors[5], dims=[-2]),
        torch.rot90(torch.flip(tensors[6], dims=[-1]), k=3, dims=[-2, -1]),
        torch.rot90(torch.flip(tensors[7], dims=[-2]), k=3, dims=[-2, -1]),
    ]

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerical sigmoid for prediction probability conversion."""
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
    """
    fixed = state_dict
    for prefix in ("module.", "model."):
        if fixed and all(k.startswith(prefix) for k in fixed.keys()):
            fixed = {k[len(prefix) :]: v for k, v in fixed.items()}
    return fixed


class TileLoader:
    """Asynchronous tile loader to overlap I/O and GPU compute."""
    def __init__(self, src, windows, tile_size, num_workers=2):
        self.src = src
        self.windows = windows
        self.tile_size = tile_size
        self.queue = Queue(maxsize=num_workers * 2)
        self.stopped = False

    def start(self):
        t = Thread(target=self._run, daemon=True)
        t.start()
        return self

    def _run(self):
        for x0, y0, tw, th in self.windows:
            if self.stopped: break
            win = Window(x0, y0, self.tile_size, self.tile_size)
            part = self.src.read(window=win, boundless=True, fill_value=0)
            tile_img = np.transpose(part, (1, 2, 0))
            self.queue.put((x0, y0, tw, th, tile_img))
        self.queue.put(None)

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None: break
            yield item

    def stop(self):
        self.stopped = True


class TiledPredictor:
    """
    End-to-end tiled predictor for segmentation + point detection.
    """

    BINARY_MODEL_KEYS: List[str] = [
        "building_mask",
        "road_mask",
        "road_centerline_mask",
        "waterbody_mask",
        "waterbody_line_mask",   # Canal / stream centreline (SegFormer)
        "utility_line_mask",
        "utility_point_mask",
    ]
    ROOF_KEY = "roof_type_mask"
    # All point outputs come from YOLO detection, not SegFormer
    POINT_KEYS: Set[str] = {
        "waterbody_point_mask",       # Class 0 — Wells
        "utility_transformer_mask",   # Class 1 — Transformers
        "overhead_tank_mask",         # Class 2 — Overhead Tanks
    }
    ALL_OUTPUT_KEYS: List[str] = BINARY_MODEL_KEYS + [
        ROOF_KEY,
        "utility_transformer_mask",
        "overhead_tank_mask",
    ]

    # YOLO class IDs -> target point masks (must match YOLO training labels)
    YOLO_CLASS_TO_MASK = {
        0: "waterbody_point_mask",    # Well
        1: "utility_transformer_mask", # Transformer
        2: "overhead_tank_mask",       # Overhead Tank
    }
    YOLO_LABELS = {
        0: "Well",
        1: "Transformer",
        2: "Overhead_Tank",
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
        minority_classes: Optional[List[str]] = None,
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
        self.minority_classes = minority_classes or ["bridge_mask", "railway_mask", "utility_line_mask"]
        self.blend_kernel = _create_spline_window(tile_size, overlap)
        
        # self.zero_shot_assistant = ZeroShotAssistant(device=device)

        self.yolo = None
        if yolo_path and YOLO:
            try:
                self.yolo = YOLO(yolo_path)
                self.yolo.to(device)
                logger.info("Loaded YOLO detector from %s", yolo_path)
            except Exception as e:
                logger.warning("Failed to load YOLO model at %s: %s", yolo_path, e)


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
        image = _to_rgb(tile)
        image = _percentile_stretch(image)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        img_t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        return img_t.unsqueeze(0).to(self.device)

    def _predict_tile_model(self, tile_img: np.ndarray) -> Dict[str, np.ndarray]:
        tensor = self._normalize_tile(tile_img)
        with torch.no_grad():
            outputs = self.model(tensor)

        out_np: Dict[str, np.ndarray] = {}
        for key, val in outputs.items():
            arr = val.detach().cpu().numpy()
            if arr.ndim >= 3:
                arr = arr[0]
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
            if boxes is None: continue
            for box in boxes:
                try:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().tolist()
                except Exception: continue

                mask_key = self.YOLO_CLASS_TO_MASK.get(cls_id)
                if mask_key not in selected_point_keys: continue

                bx1, bx2 = float(np.clip(bx1, 0, tw_act)), float(np.clip(bx2, 0, tw_act))
                by1, by2 = float(np.clip(by1, 0, th_act)), float(np.clip(by2, 0, th_act))
                if bx2 <= bx1 or by2 <= by1: continue

                area = (bx2 - bx1) * (by2 - by1)
                if area < self.yolo_min_area: continue

                gx1, gy1 = bx1 + x0, by1 + y0
                gx2, gy2 = bx2 + x0, by2 + y0
                detections.append({
                    "box": [gx1, gy1, gx2, gy2],
                    "class": cls_id,
                    "label": self.YOLO_LABELS.get(cls_id, "Unknown"),
                    "conf": conf,
                    "mask_key": mask_key,
                })
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
        if not point_masks: return point_masks

        try:
            import cv2
        except ImportError: cv2 = None

        for det in detections:
            mask_key = det.get("mask_key")
            if mask_key not in point_masks: continue
            x1, y1, x2, y2 = det["box"]
            conf = float(np.clip(det.get("conf", 1.0), 0.0, 1.0))
            cx, cy = int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5))
            radius = max(self.point_radius_px, int(max(x2 - x1, y2 - y1) * 0.25))

            if cv2 is not None:
                tmp = np.zeros_like(point_masks[mask_key])
                cv2.circle(tmp, (cx, cy), radius, color=conf, thickness=-1)
                point_masks[mask_key] = np.maximum(point_masks[mask_key], tmp)
            else:
                if 0 <= cy < h and 0 <= cx < w:
                    point_masks[mask_key][cy, cx] = max(point_masks[mask_key][cy, cx], conf)
        return point_masks

    @torch.no_grad()
    def predict_tif(
        self,
        tif_path: Path,
        selected_masks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        selected = set(selected_masks or self.ALL_OUTPUT_KEYS)
        selected = {key for key in selected if key in set(self.ALL_OUTPUT_KEYS)}
        if not selected: selected = set(self.ALL_OUTPUT_KEYS)

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

            # Async Loader (Phase 3 Optimization)
            loader = TileLoader(src, windows, self.tile_size).start()
            
            for x0, y0, tw_act, th_act, tile_img in tqdm(loader, total=len(windows), desc="Inference", leave=False):
                tile_valid = np.any(tile_img[:th_act, :tw_act] > 0, axis=2)
                if tile_valid.size == 0 or float(tile_valid.mean()) < 0.001:
                    continue
                blend = self.blend_kernel[:th_act, :tw_act]
                weight_map[y0 : y0 + th_act, x0 : x0 + tw_act] += blend

                if self.use_tta:
                    # 8-fold Dihedral TTA (GeoAI Premium)
                    t_tile = torch.from_numpy(tile_img.transpose(2,0,1)).unsqueeze(0).to(self.device).float()
                    if t_tile.max() > 1.5: t_tile /= 255.0
                    
                    augmented = _d4_forward(t_tile)
                    all_batch_logits = []
                    for aug in augmented:
                        with torch.no_grad():
                            outputs = self.model(aug)
                            # Handle dict or tensor return
                            if isinstance(outputs, dict):
                                all_batch_logits.append({k: v.cpu().numpy() for k, v in outputs.items()})
                            else:
                                all_batch_logits.append(outputs.cpu().numpy())
                    
                    # Inverse and mean
                    # (Simplified: just for the primary binary and roof keys)
                    model_outputs = {}
                    for key in self.ALL_OUTPUT_KEYS:
                        key_tensors = []
                        for i, batch_res in enumerate(all_batch_logits):
                            if isinstance(batch_res, dict) and key in batch_res:
                                t = torch.from_numpy(batch_res[key])
                                key_tensors.append(t)
                        if key_tensors:
                            restored = _d4_inverse(key_tensors)
                            model_outputs[key] = torch.stack(restored).mean(dim=0).numpy()
                else:
                    model_outputs = self._predict_tile_model(tile_img)

                for key in list(model_accum.keys()):
                    if key not in model_outputs: continue
                    logits = model_outputs[key]
                    logits2d = logits[0] if (logits.ndim == 3 and logits.shape[0] == 1) else logits
                    if logits2d.ndim != 2: continue
                    
                    prob = _sigmoid_np(logits2d[:th_act, :tw_act])
                    # if key in self.minority_classes:
                    #     zs_mask = self.zero_shot_assistant.extract_feature(tile_img[:th_act, :tw_act], key)
                    #     prob = np.maximum(prob, zs_mask)
                    model_accum[key][y0 : y0 + th_act, x0 : x0 + tw_act] += prob * blend

                if roof_accum is not None and self.ROOF_KEY in model_outputs:
                    roof_logits = model_outputs[self.ROOF_KEY]
                    if roof_logits.ndim == 3 and roof_logits.shape[0] >= 2:
                        roof_probs = _softmax_np(roof_logits[:, :th_act, :tw_act], axis=0)
                        roof_accum[:, y0 : y0 + th_act, x0 : x0 + tw_act] += (roof_probs * blend[None])

                if self.yolo is not None and selected_point_keys:
                    raw_detections.extend(self._run_yolo_tile(tile_img, x0, y0, tw_act, th_act, selected_point_keys))

        weight_map = np.maximum(weight_map, 1e-8)
        final_results: Dict[str, Any] = {}
        for key, accum in model_accum.items():
            prob = accum / weight_map
            
            # Seam reconciliation is now handled in post-processing (GeoAI layer)
            final_results[key] = prob

        if roof_accum is not None:
            roof_probs = roof_accum / weight_map[None]
            roof_mask = np.argmax(roof_probs, axis=0).astype(np.uint8)
            if "building_mask" in final_results:
                from inference.postprocess import refine_roof_types
                building_binary = (final_results["building_mask"] > self.threshold).astype(np.uint8)
                roof_mask = refine_roof_types(building_binary, roof_mask)
                roof_mask[final_results["building_mask"] <= self.threshold] = 0
            final_results[self.ROOF_KEY] = roof_mask

        detections = _nms_detections(raw_detections, iou_threshold=self.yolo_iou)
        det_point_masks = self._detections_to_point_masks(detections, h, w, selected)

        for point_key in self.POINT_KEYS:
            if point_key not in selected: continue
            seg_prob = final_results.get(point_key)
            det_prob = det_point_masks.get(point_key)
            if det_prob is not None and seg_prob is not None:
                final_results[point_key] = np.maximum(seg_prob, det_prob)
            elif det_prob is not None:
                final_results[point_key] = det_prob
            elif seg_prob is None:
                final_results[point_key] = np.zeros((h, w), dtype=np.float32)

        for key in selected:
            if key not in final_results:
                final_results[key] = np.zeros((h, w), dtype=np.uint8 if key == self.ROOF_KEY else np.float32)

        final_results["detections"] = detections
        return final_results

    @torch.no_grad()
    def predict_image(
        self,
        image_path: Path,
        selected_masks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        ext = image_path.suffix.lower()
        if ext in {".tif", ".tiff"}: return self.predict_tif(image_path, selected_masks)

        try: from PIL import Image
        except ImportError: raise ImportError("PIL (Pillow) is required for JPG/PNG support.")

        selected = set(selected_masks or self.ALL_OUTPUT_KEYS)
        selected = {k for k in selected if k in set(self.ALL_OUTPUT_KEYS)}
        if not selected: selected = set(self.ALL_OUTPUT_KEYS)

        selected_point_keys = self.POINT_KEYS & selected
        pil_img = Image.open(image_path).convert("RGB")
        full_image = np.array(pil_img, dtype=np.float32)
        if full_image.max() > 1.0: full_image = full_image / 255.0

        h, w = full_image.shape[:2]
        logger.info("Predicting %s (%dx%d)", image_path.name, w, h)

        model_accum = {key: np.zeros((h, w), dtype=np.float32) for key in self.BINARY_MODEL_KEYS if key in selected}
        roof_accum = np.zeros((5, h, w), dtype=np.float32) if self.ROOF_KEY in selected else None
        raw_detections, weight_map = [], np.zeros((h, w), dtype=np.float32)

        stride = max(1, self.tile_size - self.overlap)
        windows = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                tw = min(self.tile_size, w - x)
                th = min(self.tile_size, h - y)
                windows.append((x, y, tw, th))

        for x0, y0, tw_act, th_act in tqdm(windows, desc="Inference", leave=False):
            tile_img = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.float32)
            crop = full_image[y0 : y0 + th_act, x0 : x0 + tw_act]
            tile_img[:th_act, :tw_act] = crop
            if float(crop.max()) < 0.01: continue

            blend = self.blend_kernel[:th_act, :tw_act]
            weight_map[y0 : y0 + th_act, x0 : x0 + tw_act] += blend
            model_outputs = self._predict_tile_model(tile_img)

            for key in list(model_accum.keys()):
                if key not in model_outputs: continue
                logits = model_outputs[key]
                logits2d = logits[0] if (logits.ndim == 3 and logits.shape[0] == 1) else logits
                if logits2d.ndim != 2: continue
                model_accum[key][y0 : y0 + th_act, x0 : x0 + tw_act] += _sigmoid_np(logits2d[:th_act, :tw_act]) * blend

            if roof_accum is not None and self.ROOF_KEY in model_outputs:
                roof_logits = model_outputs[self.ROOF_KEY]
                for c in range(min(5, roof_logits.shape[0])):
                    roof_accum[c, y0 : y0 + th_act, x0 : x0 + tw_act] += roof_logits[c, :th_act, :tw_act] * blend

            if self.yolo is not None and selected_point_keys:
                raw_detections.extend(self._run_yolo_tile(tile_img, x0, y0, tw_act, th_act, selected_point_keys))

        safe_w = np.maximum(weight_map, 1e-6)
        final_results = {}
        for key, accum in model_accum.items():
            final_results[key] = (accum / safe_w >= self.threshold).astype(np.uint8)

        if roof_accum is not None:
            roof_mask = np.argmax(_softmax_np(roof_accum / safe_w[None], axis=0), axis=0).astype(np.uint8)
            if "building_mask" in final_results:
                from inference.postprocess import refine_roof_types
                roof_mask = refine_roof_types(final_results["building_mask"], roof_mask)
                roof_mask[final_results["building_mask"] <= 0.5] = 0
            final_results[self.ROOF_KEY] = roof_mask

        detections = _nms_detections(raw_detections, iou_threshold=self.yolo_iou)
        det_point_masks = self._detections_to_point_masks(detections, h, w, selected)

        for point_key in self.POINT_KEYS:
            if point_key not in selected: continue
            seg_prob = final_results.get(point_key)
            det_prob = det_point_masks.get(point_key)
            final_results[point_key] = np.maximum(seg_prob if seg_prob is not None else 0, det_prob if det_prob is not None else 0)

        for key in selected:
            if key not in final_results:
                final_results[key] = np.zeros((h, w), dtype=np.uint8 if key == self.ROOF_KEY else np.float32)

        final_results["detections"] = detections
        return final_results


def _resolve_weights_path(weights_path: str) -> Optional[Path]:
    p = Path(weights_path)
    if p.exists(): return p
    for cand in [Path("checkpoints/best.pt"), Path("checkpoints/best_latest.pt")]:
        if cand.exists(): return cand
    return None


def _resolve_yolo_path(yolo_path: Optional[str]) -> Optional[str]:
    if yolo_path:
        p = Path(yolo_path)
        if p.exists(): return str(p)
    for cand in [Path("checkpoints/yolov8s.pt"), Path("checkpoints/yolov8n.pt")]:
        if cand.exists(): return str(cand)
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
    from models.model import EnsembleSvamitvaModel
    model = EnsembleSvamitvaModel(pretrained=True)
    resolved_weights = _resolve_weights_path(weights_path)
    if resolved_weights is None: raise FileNotFoundError("No segmentation checkpoint found.")

    ckpt = torch.load(resolved_weights, map_location="cpu", weights_only=False)
    state_dict = _strip_common_state_dict_prefixes(_extract_state_dict_from_checkpoint(ckpt))
    model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded ensemble weights from %s", resolved_weights)

    resolved_yolo = _resolve_yolo_path(yolo_path) if YOLO is not None else None
    return TiledPredictor(
        model=model, device=device, yolo_path=resolved_yolo, use_tta=use_tta,
        yolo_conf=yolo_conf, yolo_iou=yolo_iou, tile_size=tile_size, overlap=overlap,
    )
