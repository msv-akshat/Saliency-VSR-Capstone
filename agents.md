# Project Blueprint: Hard Spatial Routing Video Super-Resolution (VSR)

## 1. Project Overview & Objective
* **Goal:** Build an efficient, asymmetric AI-based video upscaling pipeline that performs high-resolution processing selectively on foreground salient regions while routing background areas through low-cost bicubic interpolation.
* **Target Environment:** Windows / PowerShell, local Python 3.10 virtual environment (`venv`).

## 2. Architecture & Modules
* **`main.py`:** Pipeline orchestrator handling frame extraction, execution sequencing, metrics computation, and comparison grid saving.
* **`src/pipeline.py`:** Extracts 1080p source video frames and handles intentional multi-step degradation (downscaling to 360p 640x360) to test VSR efficiency.
* **`src/saliency.py`:** Integrates MediaPipe DeepLabV3 (`deeplab_v3.tflite`) to output binary segmentation masks and Gaussian saliency maps.
* **`src/router.py`:** Computes tight foreground bounding boxes from binary masks with strict boundary clamping and empty-mask fallback handlers.
* **`src/processors.py`:** Manages asymmetric upscaling—prioritizing ESPCN ($3\times$) for the foreground region of interest (ROI) with automatic bicubic fallback safety wrappers.
* **`src/stitcher.py`:** Handles alpha-blending stitch operations onto a 1920x1080 background canvas, computes PSNR/SSIM, and generates the final multi-panel comparison visualization grid.

## 3. Known Issues & Immediate Agent Tasks
1. **Degradation Realism:** Ensure `pipeline.py` applies genuine high-frequency loss and compression artifacts so the 360p input is visibly lower-quality than ground truth.
2. **Compute Calculation Integrity:** Ensure the router's bounding box width and height (`w, h`) are correctly tracked through `main.py` into `calculate_metrics()` so percentage compute savings reflects accurate pixel ratios rather than defaulting to 100%.
3. **OpenCV/Model Path Resilience:** Maintain absolute path resolutions for all `.tflite` and `.pb` model assets inside the `models/` directory.