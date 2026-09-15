# Project Blueprint & Agent Guide: Hard Spatial Routing Video Super-Resolution (VSR)

## 1. Project Overview & Objective
* **Goal:** Build an efficient, asymmetric AI-based video upscaling pipeline that performs high-resolution processing selectively on foreground salient regions (via Real-ESRGAN) while routing background areas through low-cost interpolation (Lanczos4).
* **Target Environment:** Windows / PowerShell, local Python 3.10 virtual environment (`venv`).

## 2. Core Operational Policies
* **Zero-Fallback Policy:** 
  * Strict prohibition against lazy hardcoded bounding boxes, mock thermal maps, or center-biased default rectangles. 
  * If subject localization or saliency extraction fails, the pipeline must raise an explicit `ValueError` and halt execution cleanly rather than outputting garbage or masked debug placeholders.

## 3. Architecture & Module Specifications
* **`main.py`:** Pipeline orchestrator handling frame extraction, execution sequencing, metrics computation, and comparison grid saving. Includes error handling to catch unresolvable saliency/routing exceptions gracefully. Updated to use dual-stream saliency extraction and graceful `ValueError` handling.

* **`src/pipeline.py`:** Extracts 1080p source video frames and handles intentional multi-step degradation to test VSR efficiency under realistic low-quality input conditions. **Updated:** Removed the 160x90 nearest-neighbor pipeline; now uses direct 360p bilinear downscale + heavy JPEG quality-20 compression to simulate a realistic low-bandwidth edge-device camera feed (no artificial blocking artifacts).

* **`src/saliency.py`:** Integrates **YOLOv8-Segmentation (`models/yolov8n-seg.pt`)** via Ultralytics. Filters detections specifically for target foreground classes (e.g., bird class index 14) to extract precise instance masks, resizing them to native 360p ($640\times360$) via bilinear interpolation and normalizing into `[0.0, 1.0]` float saliency maps. Uses a dual-stream approach (lightly-filtered interim frame for mask detection, severely-degraded frame for VSR pipeline).

* **`src/router.py`:** Computes tight foreground bounding boxes from instance segmentation contours with strict boundary clamping and zero-fallback safety rules. **Updated:** Added 15-pixel padding to all sides of the bbox, clamped to `360p` frame boundaries, returned as new coordinates.

* **`src/processors.py`:** Manages asymmetric upscaling. **Key updates:**
  * **Real-ESRGAN:** Replaced OpenCV `dnn_superres` with direct PyTorch inference on RRDBNet architecture. Loads weights manually via `torch.load()` (keys: `params_ema` / `params`). Uses device auto-detection (`cuda` if available, else `cpu`). Prints `[REALESRGAN] Using device: {device}` at startup.
  * **Unsharp Mask:** Applied after the $4\times\to3\times$ downscale step to recover micro-textures lost to GAN over-smoothing: `blur = cv2.GaussianBlur(enhanced_3x, (0, 0), 2.0); enhanced_3x = cv2.addWeighted(enhanced_3x, 1.5, blur, -0.5, 0)`.
  * **Background upscaler:** Changed from `cv2.INTER_CUBIC` to `cv2.INTER_LANCZOS4` in `resize_background()` for highest-fidelity mathematical upscale of background edges.
  * **Color space:** BGR→RGB before tensor conversion, RGB→BGR after inference, to prevent bird color inversion (brown→blue/green).
  * **Zero-Fallback Policy:** Raises `ValueError` if enhancement fails (no mock defaults).

* **`src/stitcher.py`:** Manages full-canvas background routing with YOLO-mask-based alpha blending. **Key updates:**
  * **True YOLO mask:** Extracts the actual normalized `saliency_map` (float32 $[0,1]$ array from YOLO) alongside bbox coordinates.
  * **Scales mask by $3\times$** using `cv2.INTER_LINEAR` to match the Enhanced ROI dimensions.
  * **Dilates mask** with `cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=2)` before blurring to protect subject boundaries (head/beak).
  * **Gaussian blurs** with `ksize=(15,15)` for an ultra-smooth transition.
  * **Exact pixel blending:** `blended = mask_3ch * roi_region + (1 - mask_3ch) * orig_patch`, using the actual foreground shape from YOLO segmentation – eliminating rectangular seams and edge artifacts.
  * Surrounding background remains fully intact across the entire $1920\times1080$ frame.

## 4. Model Assets & Asset Management
* **Model Directory (`models/`):** Must maintain absolute path resolution for all local binary assets.
  * `models/yolov8n-seg.pt` (YOLOv8 Segmentation weights for dynamic saliency tracking)
  * `models/RealESRGAN_x4plus.pth` (Real-ESRGAN x4+ weights, downloaded automatically by the pipeline if missing from `models/`, otherwise loaded from GitHub release)
  * `models/EDSR_x3.pb` (OpenCV-compatible EDSR $3\times$ super-resolution network weights — retained as fallback)

## 5. Execution Flow (Updated)
1. **Frame extraction** (`pipeline.py`): 1080p GT extracted → direct bilinear $\downarrow$360p + JPEG quality-20 compression → degraded 360p frames.
2. **Saliency extraction** (`saliency.py`): YOLOv8-Seg on dual‑stream frames → normalized float32 `saliency_map` $[0,1]$ + bbox.
3. **Bounding box** (`router.py`): Contour-based tight bbox + 15-pixel padding → clamped coordinates.
4. **Foreground enhancement** (`processors.py`): RRDBNet Real-ESRGAN direct PyTorch inference → Unsharp Mask → $4\times\to3\times$ resize with `INTER_AREA` → BGR output.
5. **Background routing** (`stitcher.py`): Full‑canvas bicubic/Lanczos4 up‑scale → extract YOLO mask region → scale×3 → dilate → Gaussian blur → alpha‑blend over background using exact pixel formula.
6. **Metrics & output** (`main.py`): PSNR/SSIM computed, `comparison_metrics_grid.jpg` and `saliency_heatmap.jpg` saved.

## 6. Expected Results
Running `python main.py` produces:
```
Pipeline initialized: 50 frames processed with realistic degradation.
[REALESRGAN] Using device: cpu
[REALESRGAN] RRDBNet loaded on cpu from D:\saliency_vsr_project\models\RealESRGAN_x4plus.pth
[SUCCESS] Grid Saved: data/processed/1080p_Output/comparison_metrics_grid.jpg
Metrics -> PSNR: 30.33 dB | SSIM: 0.8898 | Compute Saved: 66.9%
```
The `comparison_metrics_grid.jpg` shows:
- **Bird**: Crisper internal feather textures (Unsharp Mask removed "waxy" look), natural brown tones (no color inversion).
- **Background**: Sharper edges on branches and foliage (Lanczos4 interpolation).
- **Full bird enhancement**: Real-ESRGAN enhanced, YOLO‑mask blended seamlessly into the mathematical background up‑scale — no visible rectangular seams, no background erasure.