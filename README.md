# Geographical Complexity Assessment

> Design and implement a method to assess the complexity and local uniqueness of small geographic areas.

## Overview

This project develops a comprehensive autonomous visual localization system for UAVs in GPS-denied or spoofed environments. Input orthophoto maps of the Brno target area are transformed via a U-Net / ResNet-18 deep neural network into semantic masks (buildings, roads, water), which are then correlated against a georeferenced OpenStreetMap database to estimate the drone's position.

**Key results:**
- Single-tile localization accuracy: **66.12 %**
- 3×3 multi-tile context accuracy: **75.65 %**
- Near **100 %** accuracy over dense urban areas

---

## Repository Structure

### Data preparation
| File | Description |
|---|---|
| `display_gui.py` | Interactive GIS viewer for browsing and selecting OSM Shapefile layers over Brno |
| `create_mask_image.py` | Downloads orthophoto tiles from the ČÚZK WMS service and generates paired semantic masks from OSM vector data (256×256 px) |

### Dataset inspection
| File | Description |
|---|---|
| `view_all_mask_image.py` | Full-area linked viewer — composites all 2,500 tiles and synchronizes zoom between orthophoto and reference mask |
| `view_mask_image.py` | Single-tile browser (Tkinter GUI) — inspect orthophoto / mask pairs at any grid position [i, j] |

### Model training & validation
| File | Description |
|---|---|
| `neural_network_cpu.py` | Trains U-Net / ResNet-18 on CPU with data augmentation, Cross Entropy + Dice loss, and best-model checkpointing to `model_cpu_checkpoint.pth` |
| `view_all_mask_image_nn.py` | Three-panel linked viewer comparing orthophoto, OSM reference mask, and neural network prediction |

### Localization & evaluation
| File | Description |
|---|---|
| `map_evaluation.py` | Generates the SLAM-Safe Map — a navigation risk heatmap based on per-pixel class probabilities and weighted risk scores |
| `drone_localization_placement.py` | Single-tile localization — compares the drone's predicted mask against all 2,500 reference masks |
| `drone_localization_3x3.py` | 3×3 multi-tile localization — correlates a composite 750×750 m context mask to reduce spatial ambiguity |
| `display_errors_in_map.py` | Renders a red transparent overlay of mislocalized tiles on the full orthophoto and semantic mask |

---

## Getting Started

### Prerequisites

```
Python 3.8+, PyTorch, torchvision, segmentation-models-pytorch
GeoPandas, Albumentations, OpenCV, Matplotlib, Pillow, tkinter
```

```bash
pip install torch torchvision segmentation-models-pytorch geopandas albumentations opencv-python matplotlib pillow
```

### Pipeline

```bash
# 1. Explore and select GIS layers
python display_gui.py

# 2. Generate the dataset (orthophoto tiles + semantic masks)
python create_mask_image.py

# 3. Inspect dataset quality
python view_all_mask_image.py
python view_mask_image.py

# 4. Train the segmentation model
python neural_network_cpu.py

# 5. Validate model predictions
python view_all_mask_image_nn.py

# 6. Generate the SLAM-Safe Map
python map_evaluation.py

# 7. Run localization simulations
python drone_localization_placement.py   # Scenario A — single tile
python drone_localization_3x3.py         # Scenario B — 3×3 context

# 8. Visualize localization errors
python display_errors_in_map.py
```

---

## Technical Report

See [`Geographical_Complexity_Assessment.pdf`](./Geographical_Complexity_Assessment.pdf) for full methodology, experiments, and results.
