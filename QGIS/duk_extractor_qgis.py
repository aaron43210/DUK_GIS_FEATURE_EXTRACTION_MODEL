"""
DUK Feature Extraction - Industrial QGIS Processing Algorithm
=============================================================
Digital University Kerala — extracts all 9 GIS features from drone orthophotos.

SegFormer outputs (6):
  building_mask, road_mask, road_centerline_mask,
  waterbody_mask, waterbody_line_mask, roof_type_mask

YOLO point detection (3):
  waterbody_point_mask  (Wells)
  utility_transformer_mask (Transformers)
  overhead_tank_mask    (Overhead Tanks)

Universal OS support (Windows / macOS / Linux) with automatic
GPU detection (CUDA → MPS → CPU).
"""

import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFile,
    QgsProcessingParameterString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFolderDestination,
    QgsProject,
    QgsVectorLayer,
)


# ──────────────────────────────────────────────────────────────────────────────
# OS / environment helpers
# ──────────────────────────────────────────────────────────────────────────────


def _detect_gpu_device(python_exe: str, feedback=None) -> str:
    """
    Probe the target Python interpreter for GPU availability.
    Priority: CUDA → MPS (Apple Silicon) → CPU.
    Returns 'cuda', 'mps', or 'cpu'.
    """
    probe = (
        "import sys, importlib.util; "
        "spec = importlib.util.find_spec('torch'); "
        "sys.exit(1) if spec is None else None; "
        "import torch; "
        "print('cuda' if torch.cuda.is_available() else "
        "('mps' if hasattr(torch.backends,'mps') "
        "and torch.backends.mps.is_available() else 'cpu'))"
    )
    try:
        result = subprocess.run(
            [python_exe, "-c", probe],
            capture_output=True,
            text=True,
            timeout=30,
        )
        device = result.stdout.strip()
        if device in ("cuda", "mps", "cpu"):
            if feedback:
                feedback.pushInfo(f"[GPU Probe] Auto-detected: {device}")
            return device
    except Exception as exc:
        if feedback:
            feedback.pushInfo(f"[GPU Probe] Detection failed ({exc}), using cpu")
    return "cpu"


def _build_subprocess_cmd(python_exe: str, script_args: list):
    """
    Build a cross-platform Popen command string + extra kwargs.
    Activates the venv if an activate script lives next to python_exe.
    """
    system = platform.system()  # 'Windows', 'Darwin', 'Linux'
    bin_dir = os.path.dirname(python_exe)

    if system == "Windows":
        quoted = " ".join(f'"{a}"' for a in script_args)
        activate = os.path.join(bin_dir, "activate.bat")
        if os.path.exists(activate):
            cmd = f'"{activate}" && "{python_exe}" -u {quoted}'
        else:
            cmd = f'"{python_exe}" -u {quoted}'
        popen_kw = dict(shell=True)
    else:
        quoted = " ".join(shlex.quote(str(a)) for a in script_args)
        activate = os.path.join(bin_dir, "activate")
        if os.path.exists(activate):
            cmd = f"source {shlex.quote(activate)} && {shlex.quote(python_exe)} -u {quoted}"
        else:
            cmd = f"{shlex.quote(python_exe)} -u {quoted}"
        login_shell = os.environ.get("SHELL", "/bin/bash")
        popen_kw = dict(shell=True, executable=login_shell)

    return cmd, popen_kw


def _default_python() -> str:
    """
    Return the current Python interpreter executable path.
    Uses the active interpreter (venv or system).
    """
    return sys.executable or (
        "python.exe" if platform.system() == "Windows" else "python3"
    )


def _default_cli_path() -> str:
    """
    Locate cli.py relative to this script.
    """
    # Try relative to this file (works when loaded from disk)
    try:
        # First try: inference/cli.py relative to QGIS directory
        candidate = Path(__file__).parent / "inference" / "cli.py"
        if candidate.exists():
            return str(candidate)
        
        # Second try: ../inference/cli.py (if QGIS is in a subdirectory)
        candidate = Path(__file__).parent.parent / "inference" / "cli.py"
        if candidate.exists():
            return str(candidate)
    except NameError:
        pass

    return "inference/cli.py"  # Return default relative path if not found


# ──────────────────────────────────────────────────────────────────────────────
# QGIS Processing Algorithm
# ──────────────────────────────────────────────────────────────────────────────


class DUKExtractorAlgorithm(QgsProcessingAlgorithm):

    # ── Input raster & models ──────────────────────────────────────────
    INPUT = "INPUT"
    MODEL = "MODEL"
    YOLO_MODEL = "YOLO_MODEL"
    PYTHON_EXE = "PYTHON_EXE"
    CLI_SCRIPT = "CLI_SCRIPT"
    OUTPUT_DIR = "OUTPUT_DIR"

    # ── SegFormer layer toggles ────────────────────────────────────────
    BUILDINGS = "BUILDINGS"
    ROADS = "ROADS"
    ROAD_CENTRELINES = "ROAD_CENTRELINES"
    WATER_POLYGONS = "WATER_POLYGONS"
    WATER_LINES = "WATER_LINES"
    ROOF_TYPES = "ROOF_TYPES"

    # ── YOLO point detection toggles ───────────────────────────────────
    DETECT_WELLS = "DETECT_WELLS"
    DETECT_TRANSFORMERS = "DETECT_TRANSFORMERS"
    DETECT_OHT = "DETECT_OHT"  # Overhead Tanks

    # ── Advanced ───────────────────────────────────────────────────────
    DEVICE = "DEVICE"
    USE_SAM = "USE_SAM"
    AUTO_LOAD = "AUTO_LOAD"

    def tr(self, s):
        return s

    def createInstance(self):
        return DUKExtractorAlgorithm()

    def name(self):
        return "duk_extractor"

    def displayName(self):
        return self.tr("DUK Feature Extractor — AI Industrial (v2)")

    def group(self):
        return self.tr("DUK Scripts")

    def groupId(self):
        return "duk_scripts"

    def initAlgorithm(self, config=None):
        # ── Core inputs ──────────────────────────────────────────────
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT, self.tr("Input Orthophoto (GeoTIFF)")
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.MODEL, self.tr("SegFormer Model (best.pt)"), extension="pt"
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.YOLO_MODEL,
                self.tr("YOLO Point Detection Model (.pt) — optional"),
                extension="pt",
                optional=True,
            )
        )

        # ── Paths (auto-detected, user-editable) ─────────────────────
        self.addParameter(
            QgsProcessingParameterString(
                self.PYTHON_EXE,
                self.tr("Python Executable Path"),
                defaultValue=_default_python(),
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.CLI_SCRIPT,
                self.tr("Inference CLI Script (inference/cli.py)"),
                extension="py",
                defaultValue=_default_cli_path(),
            )
        )

        # ── SegFormer Layer Selection ────────────────────────────────
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.BUILDINGS,
                self.tr("[Segmentation] Extract Buildings"),
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ROADS,
                self.tr("[Segmentation] Extract Roads (Polygon)"),
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ROAD_CENTRELINES,
                self.tr("[Segmentation] Extract Road Centrelines (Line)"),
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.WATER_POLYGONS,
                self.tr("[Segmentation] Extract Water Bodies (Polygon)"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.WATER_LINES,
                self.tr("[Segmentation] Extract Canals / Water Lines"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ROOF_TYPES,
                self.tr("[Segmentation] Extract Roof Classification"),
                defaultValue=False,
            )
        )

        # ── YOLO Point Detection ─────────────────────────────────────
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.DETECT_WELLS, self.tr("[YOLO] Detect Wells"), defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.DETECT_TRANSFORMERS,
                self.tr("[YOLO] Detect Transformers"),
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.DETECT_OHT,
                self.tr("[YOLO] Detect Overhead Tanks"),
                defaultValue=True,
            )
        )

        # ── Advanced ─────────────────────────────────────────────────
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DEVICE,
                self.tr("Inference Device"),
                options=[
                    "Auto — GPU first, then CPU (Recommended)",
                    "CPU (Force)",
                    "MPS (Apple Silicon GPU)",
                    "CUDA (NVIDIA GPU)",
                ],
                defaultValue=0,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_SAM,
                self.tr("Use SAM 2.1 Refinement (High-Precision Buildings)"),
                defaultValue=False,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.AUTO_LOAD,
                self.tr("Auto-Load & Style Result Layers in QGIS"),
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR, self.tr("Output Folder for GIS Results")
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # ── Read parameters ──────────────────────────────────────────
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        model_path = self.parameterAsFile(parameters, self.MODEL, context)
        yolo_path = self.parameterAsFile(parameters, self.YOLO_MODEL, context)
        python_exe = self.parameterAsString(
            parameters, self.PYTHON_EXE, context
        ).strip()
        cli_path = self.parameterAsFile(parameters, self.CLI_SCRIPT, context)
        output_dir = self.parameterAsFileOutput(parameters, self.OUTPUT_DIR, context)
        use_sam = self.parameterAsBoolean(parameters, self.USE_SAM, context)
        auto_load = self.parameterAsBoolean(parameters, self.AUTO_LOAD, context)

        # ── Validate paths ───────────────────────────────────────────
        if not python_exe or not Path(python_exe).exists():
            feedback.reportError(
                f"Python executable not found: '{python_exe}'\n"
                "Set a valid path in 'Python Executable Path'."
            )
            return {}

        if not cli_path or not Path(cli_path).exists():
            feedback.reportError(
                f"CLI script not found: '{cli_path}'\n"
                "Set the path to 'inference/cli.py'."
            )
            return {}

        # ── Device ──────────────────────────────────────────────────
        idx = self.parameterAsInt(parameters, self.DEVICE, context)
        if idx == 0:
            feedback.pushInfo("Auto-detecting GPU via target Python interpreter…")
            device = _detect_gpu_device(python_exe, feedback)
        else:
            device = {1: "cpu", 2: "mps", 3: "cuda"}.get(idx, "cpu")
            feedback.pushInfo(f"Device manually set to: {device}")

        # ── Build --layers list ──────────────────────────────────────
        layers_to_run = []

        # SegFormer layers
        if self.parameterAsBoolean(parameters, self.BUILDINGS, context):
            layers_to_run.append("building_mask")
        if self.parameterAsBoolean(parameters, self.ROADS, context):
            layers_to_run.append("road_mask")
        if self.parameterAsBoolean(parameters, self.ROAD_CENTRELINES, context):
            layers_to_run.append("road_centerline_mask")
        if self.parameterAsBoolean(parameters, self.WATER_POLYGONS, context):
            layers_to_run.append("waterbody_mask")
        if self.parameterAsBoolean(parameters, self.WATER_LINES, context):
            layers_to_run.append("waterbody_line_mask")
        if self.parameterAsBoolean(parameters, self.ROOF_TYPES, context):
            layers_to_run.append("roof_type_mask")

        # Bidirectional auto-enable: roofs are attributes ON building polygons
        # If buildings requested → auto-include roof classification
        if "building_mask" in layers_to_run and "roof_type_mask" not in layers_to_run:
            layers_to_run.append("roof_type_mask")
            feedback.pushInfo(
                "Auto-enabled roof classification for building extraction"
            )
        # If roof types requested → auto-include building extraction
        if "roof_type_mask" in layers_to_run and "building_mask" not in layers_to_run:
            layers_to_run.append("building_mask")
            feedback.pushInfo(
                "Auto-enabled building extraction (required for roof classification)"
            )

        # YOLO point detection layers
        if self.parameterAsBoolean(parameters, self.DETECT_WELLS, context):
            layers_to_run.append("waterbody_point_mask")
        if self.parameterAsBoolean(parameters, self.DETECT_TRANSFORMERS, context):
            layers_to_run.append("utility_transformer_mask")
        if self.parameterAsBoolean(parameters, self.DETECT_OHT, context):
            layers_to_run.append("overhead_tank_mask")

        if not layers_to_run:
            feedback.reportError("No layers selected for extraction.")
            return {}

        tif_path = layer.source()

        feedback.pushInfo(
            f"OS: {platform.system()} | Python: {python_exe} | Device: {device}"
        )
        feedback.pushInfo(f"Layers: {', '.join(layers_to_run)}")

        # ── Build and launch subprocess ──────────────────────────────
        script_args = [
            cli_path,
            "--input",
            tif_path,
            "--model",
            model_path,
            "--output",
            output_dir,
            "--device",
            device,
            "--layers",
            *layers_to_run,
        ]
        if yolo_path:
            script_args.extend(["--yolo", yolo_path])
        if use_sam:
            script_args.append("--sam")

        cmd, popen_kw = _build_subprocess_cmd(python_exe, script_args)
        feedback.pushInfo(f"Command: {cmd}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            **popen_kw,
        )
        for line in process.stdout:
            feedback.pushInfo(line.rstrip())
            if feedback.isCanceled():
                process.terminate()
                break

        process.wait()

        if process.returncode != 0:
            feedback.reportError(
                f"Inference failed (exit code {process.returncode}). "
                "See log above for details."
            )
            return {}

        # ── Auto-load & style result layers ──────────────────────────
        if auto_load:
            out_path = Path(output_dir)
            for f in sorted(out_path.glob("*.gpkg")):
                layer_name = f.stem
                style_file = out_path / f"{layer_name}.qml"
                feedback.pushInfo(f"Loading layer: {layer_name}")
                vlayer = QgsVectorLayer(str(f), layer_name, "ogr")
                if vlayer.isValid():
                    if style_file.exists():
                        vlayer.loadNamedStyle(str(style_file))
                        feedback.pushInfo(f"  Style applied: {style_file.name}")
                    QgsProject.instance().addMapLayer(vlayer)
                else:
                    feedback.reportError(f"Could not load: {f.name}")

        feedback.pushInfo(f"✓ Done. Results: {output_dir}")
        return {self.OUTPUT_DIR: output_dir}


# ──────────────────────────────────────────────────────────────────────────────
# QGIS Processing Script entry-points (REQUIRED by QGIS script runner)
# ──────────────────────────────────────────────────────────────────────────────


def createAlgorithm():
    """Entry point used by QGIS 3.x Processing Script Runner."""
    return DUKExtractorAlgorithm()


# Alias for older QGIS versions (< 3.6) that call create() instead
create = createAlgorithm
