Agent Guidelines & Project State: Saliency-VSR
Project Overview
Title: Saliency-Guided On-Device Video Super-Resolution (Saliency-VSR)
Core Architecture: Hard Spatial Routing (66.9% AI compute savings, ~30.33 dB PSNR, ~0.8898 SSIM).
Stack: Python 3.10, OpenCV, PyTorch, Ultralytics YOLOv8-seg, Real-ESRGAN (RRDBNet), BasicSR (with torchvision functional patch for modern PyTorch compatibility).
Target Environment: Windows PowerShell / Git Bash, local Python 3.10 virtual environment (venv).
Key Technical Rules & Invariants
Zero-Fallback Policy:
Never use hardcoded mock boxes or fallback rectangles if detection/saliency fails.
Raise an explicit ValueError and halt execution cleanly rather than outputting garbage or masked debug placeholders.
Applied across all modules: src/router.py, src/processors.py, src/saliency.py.
Dual-Stream Saliency Separation:
Use a lightly-blurred interim frame ($\sigma = 5.0$) for YOLO detection to bypass compression artifacts.
Use the severely degraded 360p frame for VSR routing — maintains degradation intent.
Implemented in src/saliency.py:28-31 via frame_interim = cv2.GaussianBlur(frame_360, (0, 0), 5.0).
Color Space Discipline:
Maintain BGR/RGB conversion boundaries.
img[:, :, ::-1] before PyTorch tensor input (OpenCV loads BGR, Real-ESRGAN trained on RGB).
cv2.cvtColor(..., cv2.COLOR_RGB2BGR) after inference before any OpenCV output operations.
Prevents bird brown→blue/green color inversion.
Enforced in src/processors.py:97-98, 120-121.
Bounding Box Padding & Clamping:
Always apply 15px padding with strict boundary clamping and minimum 1-pixel dimensions.
src/router.py:25-36: x_padded = max(0, x - padding) / min(w_img, x + w + padding) - x_padded
Strict clamping: x_padded = max(0, min(x_padded, w_img - 1))
w_padded = max(1, min(w_padded, w_img - x_padded)) (enforce at least 1 pixel width)
h_padded = max(1, min(h_padded, h_img - y_padded)) (enforce at least 1 pixel height)
Comparison Grid Order:
The output grid (comparison_metrics_grid.jpg) must maintain the layout: Left (360p Degraded) | Middle (Spatial VSR Output) | Right (Ground Truth 1080p).
Implemented in src/stitcher.py:172-178:
labels = [
    f"360p (PSNR: {psnr_360:.1f}dB | SSIM: {ssim_360:.2f})",
    f"Spatial VSR (PSNR: {psnr:.1f}dB | SSIM: {ssim_score:.2f})",
    f"Ground Truth 1080p"
]
panels = [frame_360_resized, final_output, frame_gt]
Active Test Configuration
Test video defaults to data/raw/bird_video.mp4 (Target COCO Class ID: 14).
Alternative test paths are commented out in main.py:12-14 and src/saliency.py:28-30:
# main.py video_path options:
video_path = "data/raw/bird_video.mp4"       # Default Bird Video (Target Class: 14)
# video_path = "data/raw/dog_video.mp4"      # Close-up Dog Video (Target Class: 16)
# video_path = "data/raw/car_video.mp4"      # Wide-shot Car Video (Target Class: 2)
Target COCO Class ID options in src/saliency.py:27-30:
# --- SELECT YOUR TARGET COCO CLASS ID HERE ---
target_class_id = 14  # 14 = Bird
# target_class_id = 16  # 16 = Dog
# target_class_id = 2   # 2  = Car
File Structure & Path Resolution
Model Assets (Must be placed in models/ directory)
models/yolov8n-seg.pt (YOLOv8 Segmentation weights for dynamic saliency tracking)
models/RealESRGAN_x4plus.pth (Real-ESRGAN x4+ weights, loaded via torch.load() with params_ema/params key fallback)
models/EDSR_x3.pb (OpenCV-compatible EDSR 3× weights — retained as fallback)
Pipeline Output Directories
data/raw/ — Raw source videos (e.g., bird_video.mp4 placed from Google Drive)
data/processed/ground_truth_1080p/ — Extracted 1080p GT frames
data/processed/degraded_360p/ — Bilinear ↓360p + JPEG Q20 compressed frames
data/processed/1080p_Output/comparison_metrics_grid.jpg — Comparison grid (360p | VSR Output | GT 1080p)
data/processed/1080p_Output/saliency_heatmap.jpg — Saliency heatmap
Source Code Modules
src/main.py — Pipeline orchestrator, frame extraction sequencing, metrics computation, comparison grid saving
src/pipeline.py — Frame extraction + realistic degradation (360p bilinear + JPEG Q20)
src/saliency.py — YOLOv8-Seg perception, dual-stream saliency extraction, configurable target class ID
src/router.py — Bounding box computation with 15px padding + boundary clamping, Zero-Fallback Policy
src/processors.py — Direct PyTorch RRDBNet Real-ESRGAN inference, Unsharp Mask, Lanczos4 background resize, BGR/RGB color correction
src/stitcher.py — YOLO mask-based alpha blending with dilation, Gaussian blur, exact pixel blend, comparison grid generation
Active Workflow & Execution Flow
Frame extraction (pipeline.py): 1080p GT extracted → direct bilinear ↓360p + JPEG quality-20 compression → degraded 360p frames
Saliency extraction (saliency.py): YOLOv8-Seg on dual‑stream frames → normalized float32 saliency_map [0,1] + bbox
Bounding box (router.py): Contour-based tight bbox + 15-pixel padding → clamped coordinates
Foreground enhancement (processors.py): RRDBNet Real-ESRGAN direct PyTorch inference → Unsharp Mask → 4×→3× resize with INTER_AREA → BGR output
Background routing (stitcher.py): Full‑canvas Lanczos4 up‑scale → extract YOLO mask region → scale×3 → dilate → Gaussian blur → alpha‑blend over background using exact pixel formula
Metrics & output (main.py): PSNR/SSIM computed, comparison_metrics_grid.jpg and saliency_heatmap.jpg saved
Recent Improvements (Session State)
Component	Change	Impact
Degradation (pipeline.py)	Removed 160×90 nearest-neighbor; now direct 360p bilinear + JPEG Q20	Realistic edge-device camera feed, no artificial blocking artifacts
GAN Inference (processors.py)	Direct PyTorch RRDBNet; manual torch.load() weight loading	Eliminated OpenCV dnn_superres type mismatch errors
Texture Recovery (processors.py:128-130)	Added Unsharp Mask (GaussianBlur + addWeighted)	Removes "waxy" GAN over-smoothing, restores micro-feather textures
Background Upscale (processors.py:144)	Changed from INTER_CUBIC to INTER_LANCZOS4	Sharpest mathematical upscale for background edges
Mask Blending (stitcher.py)	True YOLO mask + 5×5 dilation + 15×15 Gaussian blur + exact pixel blend	Eliminates rectangular seams, head/beak boundaries protected
BBox Padding (router.py)	15-pixel padding with strict clamping to 360p frame boundaries	Protects subject (head/beak) in foreground enhancement
Color Correction (processors.py)	BGR→RGB before tensor, RGB→BGR after inference	Natural bird tones, no color inversion
Comparison Grid (stitcher.py:172-178)	Panel order: 360p	VSR Output
Verified Metrics Baseline
Running python main.py produces:

Pipeline initialized: 50 frames processed with realistic degradation.
[REALESRGAN] RRDBNet loaded on cpu from D:\saliency_vsr_project\models\RealESRGAN_x4plus.pth
[SUCCESS] Grid Saved: data/processed/1080p_Output/comparison_metrics_grid.jpg
Metrics -> PSNR: 30.33 dB | SSIM: 0.8898 | Compute Saved: 66.9%
The comparison_metrics_grid.jpg shows:

Bird: Crisper internal feather textures (Unsharp Mask removed "waxy" look), natural brown tones (no color inversion)
Background: Sharper edges on branches and foliage (Lanczos4 interpolation)
Full bird enhancement: Real-ESRGAN enhanced, YOLO‑mask blended seamlessly into the mathematical background up‑scale — no visible rectangular seams, no background erasure
Session Continuity Checklist
When resuming work on this project, verify:

models/ directory contains all three required weight files (yolov8n-seg.pt, RealESRGAN_x4plus.pth, EDSR_x3.pb)
Virtual environment is activated with correct package versions (requirements.txt)
data/raw/bird_video.mp4 exists (or appropriate test video placed)
agents.md reflects current architectural state (update if major changes made)
Zero-Fallback Policy provisions understood and respected across all modules
BGR/RGB conversion points documented and maintained in src/processors.py
Dual-stream saliency separation maintained in src/saliency.py (interim σ=5.0 vs. degraded frame)
15-pixel bbox padding + clamping in src/router.py:25-36
Unsharp Mask applied after 4×→3× downscaling in src/processors.py:128-130
Lanczos4 background upscale in src/processors.py:144
YOLO mask dilation + 15×15 Gaussian in src/stitcher.py:92-98
Comparison grid panel order: 360p | VSR Output | GT 1080p in src/stitcher.py:172-178
torchvision.transforms.functional_tensor monkey-patch at src/processors.py:8-11 for modern PyTorch compatibility
Last verified: 30.33 dB PSNR | 0.8898 SSIM | 66.9% AI compute savings Hard Spatial Routing VSR Pipeline — Saliency-Guided On-Device Video Super-Resolution