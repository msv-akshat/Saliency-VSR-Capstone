# Project Blueprint & Agent Guide: Hard Spatial Routing Video Super-Resolution (VSR)

## 1. Project Overview & Objective
* **Goal:** Build an efficient, asymmetric AI-based video upscaling pipeline that performs high-resolution processing selectively on foreground salient regions while routing background areas through low-cost bicubic interpolation.
* **Target Environment:** Windows / PowerShell, local Python 3.10 virtual environment (`venv`).

## 2. Core Operational Policies
* **Zero-Fallback Policy:** 
  * Strict prohibition against lazy hardcoded bounding boxes, mock thermal maps, or center-biased default rectangles. 
  * If subject localization or saliency extraction fails, the pipeline must raise an explicit `ValueError` and halt execution cleanly rather than outputting garbage or masked debug placeholders.

## 3. Architecture & Module Specifications
* **`main.py`:** Pipeline orchestrator handling frame extraction, execution sequencing, metrics computation, and comparison grid saving. Includes error handling to catch unresolvable saliency/routing exceptions gracefully.
* **`src/pipeline.py`:** Extracts 1080p source video frames and handles intentional multi-step degradation (nearest-neighbor downscaling to $640\times360$ combined with severe compression artifacts) to test VSR efficiency under realistic low-quality input conditions.
* **`src/saliency.py`:** Integrates **YOLOv8-Segmentation (`models/yolov8n-seg.pt`)** via Ultralytics. Filters detections specifically for target foreground classes (e.g., bird class index 14) to extract precise instance masks, resizing them to native $360\text{p}$ ($640\times360$) via bilinear interpolation and normalizing into `[0.0, 1.0]` float saliency maps.
* **`src/router.py`:** Computes tight foreground bounding boxes from instance segmentation contours with strict boundary clamping and zero-fallback safety rules.
* **`src/processors.py`:** Manages asymmetric upscaling—prioritizing **EDSR ($3\times$)** (`models/EDSR_x3.pb`) via OpenCV's `dnn_superres` module for the foreground region of interest (ROI). Applies a bilateral pre-filter (`cv2.bilateralFilter`) to smooth harsh block boundaries prior to neural upsampling, followed by post-sharpening kernels.
* **`src/stitcher.py`:** Manages full-canvas background routing where the base canvas consists of the *entire* 360p frame upscaled to $1920\times1080$ using bicubic interpolation. Blends the EDSR-enhanced foreground ROI ($3\times$) smoothly over the background canvas using saliency-derived **Gaussian feathering/alpha blending** to eliminate visible rectangular seams or background erasure bugs. Computes PSNR/SSIM and outputs the comparison metrics grid alongside standalone `saliency_heatmap.jpg` thermal visuals.

## 4. Model Assets & Asset Management
* **Model Directory (`models/`):** Must maintain absolute path resolution for all local binary assets.
  * `models/yolov8n-seg.pt` (YOLOv8 Segmentation weights for dynamic saliency tracking)
  * `models/EDSR_x3.pb` (OpenCV-compatible EDSR $3\times$ super-resolution network weights)