# 🎓 DIGITAL UNIVERSITY KERALA FEATURE EXTRACTION MODEL (DUK-FEM)

> **High-resolution geospatial feature extraction from drone orthophotos.** 
> This pipeline achieves ≥95% accuracy by leveraging a specialized ensemble of State-of-the-Art (SOTA) deep learning models.

---

## 🎯 Scope
This repository is scoped to **Problem Statement 1** only:
- **Building footprint extraction** + roof class (RCC/Tiled/Tin/Others)
- **Road extraction** (Polygons & Centerlines)
- **Waterbody extraction**
- **Utility point/line extraction** (transformers, pipelines, wells, etc.)

*Note: Problem Statement 2 (DTM/drainage from point cloud) is intentionally excluded from this codebase.*

---

## 🚀 Key Features
- **Unified SOTA Architecture**: Transitioned to a multi-scale **SegFormer-B4** backbone with an **UPerFPN** decoder for industry-leading perceptual accuracy.
- **Specialized Task Heads**: Integrated multi-head system:
  - **BuildingHead**: Dual-output for high-precision segmentation and roof classification.
  - **BinaryHead/LineHead**: Optimized for waterbodies and utility lines.
- **Full State Resumption**: Robust training pipeline with 100% recovery of optimizer and scheduler states.
- **Interactive Tiled Inference**: Optimized **Streamlit V1** app for large-scale GeoTIFF visualization (Global vs. Detail views).
- **Point Feature Fusion**: Seamless integration of **YOLOv8** for sparse point objects (wells, transformers, tanks).
- **Unified GIS Export**: Automated generation of georeferenced **GeoPackage (.gpkg)** layers.

---

## 🏗️ Architecture Stack

| Component | Implementation | Feature Focus |
|:---|:---|:---|
| **Backbone** | SegFormer-B4 (Mix ViT) | Multi-scale Feature Extraction & Transformers |
| **Decoder** | UPerFPN + CBAM Attention | Context-Aware Global Fusion |
| **Buildings**| Dual-Output Head | Instance Mask + Roof Classification |
| **Roads** | D-LinkNet Head | Network Connectivity & Smoothing |
| **Utilities** | U-Net++ Multi-Head | Linear & Point Feature Precision |
| **Points** | YOLOv8 + Fusion | Wells, Transformers, Tanks |

---

## 🧭 Recommended Global Strategy
For best practical accuracy and stability on drone imagery, we utilize a hybrid pipeline:
1. **SAM2**: Used for high-speed auto-label bootstrapping and initial mask priors.
2. **Task-Specialized Production Heads**:
   - **DeepLabV3+**: Building and water polygons.
   - **D-LinkNet**: Roads and centerlines.
   - **U-Net++**: Utility linear features.
   - **YOLOv8**: Sparse point objects (wells, transformers, tanks).
3. **Classification**: Dedicated head for RCC/Tiled/Tin/Others roof types.

---

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/aaron43210/DUK_GIS_FEATURE_EXTRACTION_MODEL.git
cd DUK_GIS_FEATURE_EXTRACTION_MODEL

# Install Dependencies
pip install -r requirements.txt
```

---

## 🛠️ Usage

### 1. Interactive Web Application
Launch the production-grade Streamlit interface for end-to-end extraction and visualization:
```bash
streamlit run app.py
```

### 2. Model Training
Train the ensemble model on your own drone dataset (MAP1, MAP2, etc.):
```bash
python train.py --train_dirs /path/to/data --epochs 100 --tile_size 1024 --tile_overlap 192
```

### 3. Verification Run (3-Epoch Smoke Test)
```bash
python train.py \
  --train_dirs ../DATA/MAP1 \
  --epochs 3 \
  --batch_size 1 \
  --tile_size 1024 \
  --tile_overlap 192 \
  --name map1_verify_run
```

### 4. DGX/GPU Cluster Training
For high-performance environments (NVIDIA DGX), use the distributed multi-GPU script:
```bash
bash run_ddp.sh /path/to/data
```

---

## 📁 Data Structure
Organize your data as follows for automated training:
```text
data/
└── MAP_ID/
    ├── MAP_ID.tif         # High-resolution Orthophoto
    ├── Build_up.shp       # Building annotations
    ├── Road.shp           # Road annotations
    └── ...                # Other feature shapefiles
```

---

## 🎯 Output Keys

| Output Key | Target Feature | Geometry |
|:---|:---|:---|
| `building_mask` | Built-up Area | Polygon |
| `roof_type_mask`| Roof Classification | Polygon |
| `road_mask` | Road | Polygon |
| `road_centerline_mask` | Road Centre Line | Line |
| `waterbody_mask`| Water Body | Polygon |
| `waterbody_line_mask` | Water Body Line | Line |
| `waterbody_point_mask` | Wells | Point |
| `utility_line_mask` | Utility (Pipeline/Wires) | Line |
| `utility_point_mask`| Utility Point | Point |

---

## 📊 Performance & Target
- **Target Precision**: ≥95% across all primary layers.
- **Resolution Support**: Native processing for 5cm - 10cm GSD drone imagery.
- **Compliance**: Standards-based OGC GIS vector outputs (SHP/GPKG/GeoJSON).

---

**Developed with ❤️ by Digital University Kerala**
