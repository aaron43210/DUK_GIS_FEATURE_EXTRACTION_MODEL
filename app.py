"""
SVAMITVA Feature Extraction — Production Ensemble V3 Web Application
"""

import atexit
import io
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import streamlit as st
import torch
from inference.predict import load_ensemble_pipeline
from inference.postprocess import refine_mask, get_threshold

# ── Page Config ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SVAMITVA Ensemble AI",
    page_icon="🛰️",
    layout="wide",
)

st.markdown(
    """
<style>
    /* Premium Dark Theme & Glassmorphism */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e6;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    /* Glassmorphism containers */
    [data-testid="stSidebar"] {
        background-color: rgba(17, 25, 40, 0.75);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Metadata ─────────────────────────────────────────────────────────

FEATURES = {
    "building_mask": ("Built-up Area (Polygon)", (255, 100, 50)),
    "roof_type_mask": ("Roof Classification (Polygon)", (255, 0, 180)),
    "road_mask": ("Road (Polygon)", (255, 255, 100)),
    "road_centerline_mask": ("Road Centre Line (LineString)", (255, 220, 0)),
    "waterbody_mask": ("Water Body (Polygon)", (50, 150, 255)),
    "waterbody_line_mask": ("Water Body Line (LineString)", (0, 210, 255)),
    "waterbody_point_mask": ("Well", (0, 255, 255)),
    "utility_line_mask": ("Utility - Pipeline/Wires (LineString)", (50, 220, 100)),
    "utility_transformer_mask": ("Transformer", (0, 255, 120)),
    "bridge_mask": ("Bridge (Polygon)", (255, 140, 0)),
    "railway_mask": ("Railway (LineString)", (180, 80, 255)),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper for robust weight selection
def get_best_ckpt():
    candidates = [
        "check/best.pt",
        "check/best_latest.pt",
        "checkpoints/best.pt",
        "checkpoints/ensemble_v3.pt",
        "best_latest.pt",
        "best.pt",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return candidates[0]


# Security: Track temp files for cleanup
_temp_files: list = []


def _cleanup_temp_files():
    for f in _temp_files:
        try:
            os.unlink(f)
        except OSError:
            pass


atexit.register(_cleanup_temp_files)


# Upload size limit (10 GB)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    st.title("🛰️ SVAMITVA Production Ensemble V3")

    with st.sidebar:
        st.header("⚙️ Model Config")
        ckpt_path = st.text_input("Segmentation Weights", get_best_ckpt())

        # Try to find the local YOLO model first
        paths_to_check = [
            "check/yolo_best.pt",
            "check/yolo_svamitva_best.pt",
            os.path.expanduser("~/yolo_svamitva_best.pt"),
            os.path.expanduser("~/Downloads/yolo_svamitva_best.pt"),
            "checkpoints/yolov8s.pt",
        ]

        default_yolo = paths_to_check[-1]
        for p in paths_to_check:
            if Path(p).exists():
                default_yolo = str(p)
                break

        yolo_path = st.text_input("YOLOv8 Weights", default_yolo)

        st.divider()
        st.subheader("🛠️ Extraction Tasks")
        selected_masks = []
        for key, meta in FEATURES.items():
            enabled = st.checkbox(meta[0], value=True, key=f"feature_{key}")
            if enabled:
                selected_masks.append(key)

        st.divider()
        st.subheader("🎛️ Parameters")
        threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.50)
        yolo_conf = st.slider("YOLO Confidence", 0.05, 0.95, 0.25)
        yolo_iou = st.slider("YOLO NMS IoU", 0.10, 0.90, 0.45)
        use_tta = st.checkbox("Enable TTA (higher quality, slower)", value=False)
        tile_size = st.select_slider(
            "Inference Tile Size",
            options=[512, 768, 1024, 1280, 1536],
            value=512,
        )
        overlap = st.slider("Tile Overlap", 64, 384, 192, step=32)
        alpha = st.slider("Visual Opacity", 0.1, 0.9, 0.5)

        st.divider()
        st.subheader("📤 Export Settings")
        export_format = st.radio(
            "GIS Format",
            ["GeoPackage (.gpkg)", "ESRI Shapefile (.shp)"],
            index=0,
            help="Shapefiles are better for older GIS software, GeoPackage is recommended for modern QGIS/ArcGIS.",
        )
        export_ext = "SHP" if "Shapefile" in export_format else "GPKG"

    st.markdown("### 🗺️ Data Source")
    uploaded = st.file_uploader(
        "Upload Image (GeoTIFF, JPG, PNG, JPEG)",
        type=["tif", "tiff", "jpg", "jpeg", "png"],
    )

    if uploaded:
        # Security: Validate file size
        file_size = len(uploaded.getvalue())
        if file_size > MAX_UPLOAD_BYTES:
            st.error(
                f"File too large ({file_size / 1e6:.0f} MB). "
                f"Maximum allowed: {MAX_UPLOAD_BYTES / 1e6:.0f} MB."
            )
            st.stop()

        # Security: Validate extension
        ext = Path(uploaded.name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            st.error(f"Unsupported file type: {ext}")
            st.stop()

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tif_path = Path(tmp.name)
            _temp_files.append(tmp.name)

        is_geospatial = ext in [".tif", ".tiff"]

        col_run, col_info = st.columns([1, 2])
        with col_run:
            if st.button(
                "🚀 Execute Unified Pipeline", type="primary", width="stretch"
            ):
                if not selected_masks:
                    st.warning("⚠️ Please select at least one feature to extract!")
                    st.stop()

                with st.spinner("Analyzing Image..."):
                    st.info(
                        "🧬 Applying Training-Matched Normalization (Percentile + ImageNet)"
                    )
                    predictor = load_ensemble_pipeline(
                        weights_path=ckpt_path,
                        yolo_path=yolo_path,
                        device=DEVICE,
                        use_tta=use_tta,
                        yolo_conf=yolo_conf,
                        yolo_iou=yolo_iou,
                        tile_size=tile_size,
                        overlap=overlap,
                    )
                    predictor.threshold = threshold

                    if is_geospatial:
                        st.session_state.results = predictor.predict_tif(
                            tif_path,
                            selected_masks=selected_masks,
                        )
                    else:
                        st.session_state.results = predictor.predict_image(
                            tif_path,
                            selected_masks=selected_masks,
                        )

                    st.session_state.tif_path = tif_path
                    st.session_state.is_geospatial = is_geospatial
                    st.success("Analysis Complete!")

        results = st.session_state.get("results")
        if results:
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.get("is_geospatial", True):
                    if st.button("📥 Generate GIS Layers"):
                        from inference.export import export_predictions

                        with tempfile.TemporaryDirectory() as out_dir:
                            export_predictions(
                                results,
                                tif_path,
                                Path(out_dir),
                                threshold=threshold,
                                roof_type_mask=results.get("roof_type_mask"),
                                export_format=export_ext,
                            )
                            zip_buf = io.BytesIO()
                            with zipfile.ZipFile(zip_buf, "w") as zf:
                                # Shapefiles have multiple files per layer (.shp, .dbf, .shx, .prj)
                                for p in Path(out_dir).rglob("*.*"):
                                    if p.suffix.lower() in [
                                        ".gpkg",
                                        ".shp",
                                        ".dbf",
                                        ".shx",
                                        ".prj",
                                        ".cpg",
                                    ]:
                                        zf.write(p, p.name)
                            st.download_button(
                                f"📩 Download {export_ext} ZIP",
                                zip_buf.getvalue(),
                                f"vidhata_export_{export_ext.lower()}.zip",
                            )
                else:
                    st.info("📦 GIS Export disabled for non-geospatial image formats.")

            import cv2

            with rasterio.open(str(tif_path)) as src:
                H, W = src.height, src.width
                scale = min(1200.0 / max(H, W), 1.0)
                th = max(1, int(H * scale))
                tw = max(1, int(W * scale))
                thumb = np.transpose(
                    src.read(
                        out_shape=(src.count, th, tw),
                        resampling=rasterio.enums.Resampling.bilinear,
                    ),
                    (1, 2, 0),
                )
                if thumb.shape[2] > 3:
                    thumb = thumb[:, :, :3]
                if thumb.shape[2] == 1:
                    thumb = np.repeat(thumb, 3, axis=2)
                if thumb.dtype != np.uint8:
                    t = thumb.astype(np.float32)
                    vmax = float(np.percentile(t, 99.0))
                    if vmax <= 0:
                        vmax = 1.0
                    thumb = np.clip(t / vmax, 0.0, 1.0) * 255.0
                thumb = thumb.astype(np.uint8)

            tab_global, tab_detail = st.tabs(["🌍 Global Overview", "🔍 Detail View"])

            with tab_global:
                st.subheader("Ensemble Overview")
                overlay = thumb.copy()
                for key, (name, color) in FEATURES.items():
                    if key in results:
                        interp = (
                            cv2.INTER_NEAREST
                            if key == "roof_type_mask"
                            else cv2.INTER_LINEAR
                        )
                        m_small = cv2.resize(
                            results[key],
                            (thumb.shape[1], thumb.shape[0]),
                            interpolation=interp,
                        )
                        binary = (
                            m_small > 0
                            if key == "roof_type_mask"
                            else m_small > threshold
                        )
                        for c in range(3):
                            overlay[binary, c] = (
                                overlay[binary, c] * (1 - alpha) + color[c] * alpha
                            )

                # Draw high-visibility markers for points
                if "detections" in results:
                    scale_x = thumb.shape[1] / results.get("width", 1)
                    scale_y = thumb.shape[0] / results.get("height", 1)
                    for det in results["detections"]:
                        x1, y1, x2, y2 = det["box"]
                        cx = int((x1 + x2) / 2 * scale_x)
                        cy = int((y1 + y2) / 2 * scale_y)

                        # Get color from FEATURES
                        mask_key = det.get("mask_key")
                        if mask_key in FEATURES:
                            color = FEATURES[mask_key][1][::-1]  # RGB -> BGR for cv2
                            # Draw a circle + crosshair
                            cv2.circle(overlay, (cx, cy), 12, (255, 255, 255), 2)
                            cv2.circle(overlay, (cx, cy), 10, color, -1)
                            cv2.drawMarker(
                                overlay,
                                (cx, cy),
                                (0, 0, 0),
                                markerType=cv2.MARKER_CROSS,
                                markerSize=15,
                                thickness=2,
                            )

                st.image(overlay.astype(np.uint8), width="stretch")

            with tab_detail:
                st.subheader("Feature Inspection")
                available = [k for k in FEATURES.keys() if k in results]
                if not available:
                    st.info("Run extraction to see individual layers.")
                else:
                    selected = st.selectbox(
                        "Layer to Inspect",
                        available,
                        format_func=lambda x: FEATURES[x][0],
                    )

                    f_name, f_color = FEATURES[selected]
                    m_raw = results[selected]

                    is_point = "point" in selected.lower()
                    if is_point:
                        # Dilate point masks significantly so they survive interpolation
                        kernel = np.ones((5, 5), np.uint8)
                        m_raw = cv2.dilate(
                            m_raw.astype(np.float32), kernel, iterations=2
                        )
                    else:
                        # Apply noise reduction filter to polygons/lines
                        binary = (m_raw > threshold).astype(np.uint8)
                        refined = refine_mask(binary, selected)
                        # Blend refined mask back into probability map for better visual
                        m_raw = np.where(refined > 0, m_raw, 0)

                    interp = (
                        cv2.INTER_NEAREST
                        if selected == "roof_type_mask" or is_point
                        else cv2.INTER_LINEAR
                    )
                    m_disp = cv2.resize(
                        m_raw,
                        (thumb.shape[1], thumb.shape[0]),
                        interpolation=interp,
                    )

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.image(thumb.astype(np.uint8), caption="Original Image")
                    with col_b:
                        if selected == "roof_type_mask":
                            import matplotlib.pyplot as plt

                            cmap = plt.get_cmap("tab10")
                            c_map = (cmap(m_disp)[:, :, :3] * 255).astype(np.uint8)
                            c_map[m_disp == 0] = 0
                            st.image(c_map, caption=f"{f_name} Map")
                        else:
                            c_mask = np.zeros_like(thumb)
                            binary = m_disp > threshold
                            for i in range(3):
                                c_mask[binary, i] = f_color[i]

                            is_point = "point" in selected.lower()
                            caption = f_name if is_point else f"{f_name} Mask"

                            if is_point and "detections" in results:
                                # Draw marker on the mask preview too
                                scale_x = thumb.shape[1] / results.get("width", 1)
                                scale_y = thumb.shape[0] / results.get("height", 1)
                                for det in results["detections"]:
                                    if det.get("mask_key") != selected:
                                        continue
                                    x1, y1, x2, y2 = det["box"]
                                    cx = int((x1 + x2) / 2 * scale_x)
                                    cy = int((y1 + y2) / 2 * scale_y)
                                    cv2.circle(c_mask, (cx, cy), 15, (255, 255, 255), 2)
                                    cv2.circle(c_mask, (cx, cy), 12, f_color[::-1], -1)

                            st.image(c_mask.astype(np.uint8), caption=caption)

                    st.divider()
                    st.write(f"**{f_name} Analysis**")
                    f_ovl = thumb.copy()
                    is_roof = selected == "roof_type_mask"
                    bin_mask = m_disp > 0 if is_roof else m_disp > threshold
                    if is_roof:
                        import matplotlib.pyplot as plt

                        cmap = plt.get_cmap("tab10")
                        c_roof = (cmap(m_disp)[:, :, :3] * 255).astype(np.uint8)
                        f_ovl[bin_mask] = (
                            f_ovl[bin_mask] * (1 - alpha) + c_roof[bin_mask] * alpha
                        )
                    else:
                        for i in range(3):
                            f_ovl[bin_mask, i] = (
                                f_ovl[bin_mask, i] * (1 - alpha) + f_color[i] * alpha
                            )

                    # Also draw point markers in detail view if it's a point layer
                    is_point = "point" in selected.lower()
                    if is_point and "detections" in results:
                        scale_x = thumb.shape[1] / results.get("width", 1)
                        scale_y = thumb.shape[0] / results.get("height", 1)
                        for det in results["detections"]:
                            if det.get("mask_key") != selected:
                                continue
                            x1, y1, x2, y2 = det["box"]
                            cx = int((x1 + x2) / 2 * scale_x)
                            cy = int((y1 + y2) / 2 * scale_y)

                            cv2.circle(f_ovl, (cx, cy), 15, (255, 255, 255), 2)
                            cv2.circle(f_ovl, (cx, cy), 12, f_color[::-1], -1)
                            cv2.drawMarker(
                                f_ovl,
                                (cx, cy),
                                (0, 0, 0),
                                markerType=cv2.MARKER_CROSS,
                                markerSize=20,
                                thickness=2,
                            )

                    st.image(f_ovl.astype(np.uint8), width="stretch")


if __name__ == "__main__":
    main()
