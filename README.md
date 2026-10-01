# Saliency-Guided On-Device Video Super-Resolution (Saliency-VSR)

## Project Vision & Core Problem Statement

The core problem this project addresses is the **fundamental inefficiency of full-frame Video Super-Resolution (VSR)**. Traditional VSR approaches apply neural network processing to every single pixel across all frames, requiring massive computational resources while often wasting compute on stationary or background regions that don't benefit from AI enhancement. 

Our **Hard Spatial Routing thesis** fundamentally reorients this paradigm: instead of treating the entire frame as equally important, we allocate AI compute exclusively to salient foreground regions while routing background areas through mathematically precise, low-cost interpolation. This asymmetric approach delivers two critical advantages:

1. **Compute Efficiency**: Approximately 66.9% AI compute savings by enhancing only the foreground ROI (typically <15% of total frame area)
2. **Preserved Naturalness**: Background edges remain sharp through Lanczos4 interpolation rather than being over-smoothed by GANs

The pipeline achieves this by integrating **YOLOv8-Segmentation** for precise foreground detection, **Real-ESRGAN** (RRDBNet) for high-quality foreground enhancement with texture-preserving Unsharp Mask processing, and **Lanczos4 interpolation** for mathematically superior background upscaling. The enhanced foreground and routed background are then blended using exact pixel-level alpha blending based on the true YOLO segmentation mask — eliminating rectangular seams and edge artifacts entirely.

## End-to-End Architecture & Data Flow

The pipeline executes as a carefully orchestrated sequence of modular steps:

### 1. Frame Extraction & Realistic Degradation (`src/pipeline.py`)
- **Input**: 1080p ground-truth video (e.g., `bird_video.mp4` from Google Drive, placed in `data/raw/`)
- **Process**: 
  - Extracts 50 frames programmatically via OpenCV VideoCapture
  - Direct bilinear downscale: 1920×1080 → 640×360 (no artificial nearest-neighbor blocking)
  - Heavy JPEG quality-20 compression to simulate low-bandwidth edge-device camera feed
  - No artificial `INTER_NEAREST` blocking artifacts — provides GAN with natural compression gradients
- **Output**: 
  - `ground_truth_1080p/` — original frames
  - `degraded_360p/` — compressed, downsampled frames for VSR testing

### 2. Dual-Stream Saliency Perception (`src/saliency.py`)
- **Lightly-filtered interim frame** (Gaussian blur σ=5.0): Passed to YOLOv8 for robust mask detection against compression artifacts
- **Severely-degraded frame**: Feeds the actual VSR routing pipeline
- **YOLOv8-Segmentation** (`models/yolov8n-seg.pt`): Filters for target COCO class (default: class 14 = Bird)
- **Outputs**: 
  - `binary_mask` — np.uint8 array (360, 640), 0/1 values
  - `saliency_map` — np.float32 array (360, 640), values in [0.0, 1.0]

### 3. Spatial Routing & Bounding Box (`src/router.py`)
- Computes tight foreground bounding box from YOLO instance segmentation contours
- **15-pixel padding** on all sides with strict clamping to 360p frame boundaries
- **Zero-Fallback Policy**: Raises `ValueError` if no foreground contours found — no mock hardcoded boxes
- **Output**: Padded bounding box coordinates `(x_padded, y_padded, w_padded, h_padded)`

### 4. Asymmetric Foreground Enhancement (`src/processors.py`)
- **Direct PyTorch RRDBNet** Real-ESRGAN inference (replaced OpenCV `dnn_superres`)
- **BGR→RGB** conversion before tensor input (Real-ESRGAN trained on RGB)
- **Model initialization**: Manual `torch.load()` with `params_ema`/`params` key fallback
- **4×→3× downscale**: `cv2.INTER_AREA` resize from 4×-upsampled output to 3× original size
- **Unsharp Mask** (critical for texture recovery):
  ```python
  blur = cv2.GaussianBlur(enhanced_3x, (0, 0), 2.0)
  enhanced_3x = cv2.addWeighted(enhanced_3x, 1.5, blur, -0.5, 0)
  ```
- **RGB→BGR** conversion after inference (OpenCV expects BGR)
- **Output**: Enhanced foreground ROI at 3× scale, BGR format

### 5. Background Routing & Lanczos4 Upscaling (`src/processors.py`)
- **`resize_background()`**: Changed from `cv2.INTER_CUBIC` to `cv2.INTER_LANCZOS4` for highest-fidelity mathematical upscale
- **Output**: Full 1920×1080 background canvas

### 6. True YOLO Mask Alpha Blending (`src/stitcher.py`)
- **Extracts actual normalized `saliency_map`** (float32 [0,1] array from YOLO) alongside bbox coordinates
- **Scales mask by 3×** using `cv2.INTER_LINEAR` to match Enhanced ROI dimensions
- **Dilates mask** with 5×5 kernel (2 iterations) before blurring to protect subject boundaries (head/beak)
- **Gaussian blur** with `ksize=(15,15)` for ultra-smooth transition
- **Exact pixel blending**: `blended = mask_3ch * roi_region + (1 - mask_3ch) * orig_patch`
- **Uses actual foreground shape from YOLO segmentation** — eliminates rectangular seams
- **Surrounding background fully intact** across entire 1920×1080 frame

### 7. Metrics & Comparison Grid (`main.py` + `src/stitcher.py`)
- **PSNR/SSIM computation** between ground truth and final output
- **Comparison grid** saved as `comparison_metrics_grid.jpg` with three panels:
  - Left: 360p Degraded input (with PSNR/SSIM label)
  - Middle: Spatial VSR Output (with PSNR/SSIM label)
  - Right: Ground Truth 1080p (with "Ground Truth 1080p" label)
- **Banner**: `AI Compute Saved: 66.9% | Hard Spatial Routing Pipeline`

## Current Implementation Status & MVP Milestones

| Milestone | Status |
|-----------|--------|
| **Modular `src/` architecture** | 6 specialized modules (`main.py`, `pipeline.py`, `saliency.py`, `router.py`, `processors.py`, `stitcher.py`) |
| **50-frame sequence execution** | Verified pipeline processes all 50 frames |
| **Verified metrics** | **30.33 dB PSNR**, **0.8898 SSIM**, **66.9% AI compute savings** |
| **Realistic degradation** | Removed 160×90 nearest-neighbor; now bilinear ↓360p + JPEG Q20 |
| **Real-ESRGAN via direct PyTorch** | Replaced OpenCV `dnn_superres`; manual weight loading via `torch.load()` |
| **Unsharp Mask** | Added after 4×→3× downscaling to recover micro-textures |
| **Lanczos4 background** | Upgraded from `INTER_CUBIC` for sharpest edge preservation |
| **YOLO mask blending** | True segmentation mask alpha blend with dilation + 15×15 Gaussian |
| **15-pixel bbox padding** | Clamped to 360p frame boundaries, Zero-Fallback on failure |
| **BGR/RGB color correction** | Prevents bird brown→blue/green inversion |
| **Dual-stream saliency** | Lightly-filtered interim frame for mask detection, degraded frame for VSR |
| **Configurable test videos** | Commented options for bird (class 14), dog (class 16), car (class 2) |
| **Full video export roadmap** | HANDOFF_GUIDE.md outlines cv2.VideoWriter + argparse implementation |

**Verified Pipeline Output:**
```
Pipeline initialized: 50 frames processed with realistic degradation.
[REALESRGAN] RRDBNet loaded on cpu from D:\saliency_vsr_project\models\RealESRGAN_x4plus.pth
[SUCCESS] Grid Saved: data/processed/1080p_Output/comparison_metrics_grid.jpg
Metrics -> PSNR: 30.33 dB | SSIM: 0.8898 | Compute Saved: 66.9%
```

## Technical Stack Reference Table

| Component | Library/Module | Version | Role |
|-----------|---------------|---------|------|
| **PyTorch** | `torch` | ≥2.0.0 | Direct RRDBNet inference, tensor operations |
| **TorchVision** | `torchvision` | ≥0.15.0 | Supporting basic transforms, monkey-patch requirement |
| **Ultralytics YOLO** | `ultralytics` | ≥8.0.0 | YOLOv8-Segmentation for foreground mask detection |
| **OpenCV** | `cv2` | ≥4.7.0 | Frame extraction, resizing, JPEG compression, alpha blending |
| **NumPy** | `numpy` | ≥1.23.0 | Array operations, mask manipulations |
| **scikit-image** | `scikit-image` | ≥0.19.0 | SSIM computation (`structural_similarity`) |
| **BasicSR** | `basicsr` | ≥1.4.2 | RRDBNet architecture import (requires monkey-patch) |
| **Model Weights** | `yolov8n-seg.pt` | — | YOLOv8-Segmentation foreground class detection |
| **Model Weights** | `RealESRGAN_x4plus.pth` | — | Real-ESRGAN x4+ RRDBNet weights |
| **Model Weights** | `EDSR_x3.pb` | — | OpenCV fallback weights (retained but not currently used) |

## Technical Challenges & Corrective Actions

### 1. BasicSR Dependency Patch for Modern PyTorch
**Problem**: `basicsr.archs.rrdbnet_arch.RRDBNet` import fails in modern PyTorch ≥2.0 due to `torchvision.transforms.functional_tensor.ModuleNotFoundError`.

**Corrective Action** (implemented in `src/processors.py:7-11`):
```python
# --- Monkey-patch to fix broken basicsr import in modern PyTorch ---
import torchvision
import torchvision.transforms.functional as TF
sys.modules['torchvision.transforms.functional_tensor'] = TF
# ------------------------------------------------------------
```
This patch redirects the broken module reference to the functional API that exists in the installed torchvision version.

### 2. 360p Compression Artifact Handling
**Problem**: Extreme JPEG quality-20 compression creates blocking artifacts and aliased edges that can confuse YOLOv8 detection.

**Corrective Action** (dual-stream approach in `src/saliency.py`):
- Lightly-filtered interim frame (Gaussian blur σ=5.0) for YOLO mask detection
- Severely-degraded frame feeds the actual VSR pipeline
- This separation prevents YOLO from failing on compression artifacts while maintaining VSR degradation integrity

### 3. Strict Boundary Clamping
**Problem**: 15-pixel bbox padding could exceed frame boundaries, causing out-of-array access errors or invalid stitching.

**Corrective Action** (in `src/router.py:25-36`):
- `max(0, ...)` and `min(dimension, ...)` clamping for all padded coordinates
- Enforce minimum 1-pixel width/height: `max(1, min(w_padded, w_img - x_padded))`
- Strict clamping: `x_padded = max(0, min(x_padded, w_img - 1))`
- Zero-Fallback Policy raises `ValueError` if no valid contours detected instead of defaulting to mock boxes

### 4. Color Space Inversion Prevention
**Problem**: OpenCV loads images in BGR format, but Real-ESRGAN was trained on RGB. Without correction, brown bird feathers appear blue/green.

**Corrective Action** (in `src/processors.py:97-98, 120-121`):
- `img = img[:, :, ::-1]` — BGR → RGB before tensor conversion
- `out_bgr = cv2.cvtColor(output_uint8, cv2.COLOR_RGB2BGR)` — RGB → BGR after inference

### 5. Zero-Fallback Policy Enforcement
**Problem**: Temptation to use hardcoded default bounding boxes or mock thermal maps when saliency extraction fails.

**Corrective Action** (across all modules):
- `src/router.py`: Raises `ValueError` if no foreground contours found
- `src/processors.py`: Raises `ValueError` if enhancement fails (no mock defaults)
- `src/saliency.py`: Raises `ValueError` if no foreground class detected
- **Philosophy**: Pipeline must halt cleanly rather than outputting garbage or masked debug placeholders

## Future Technical Roadmap

### Near-Term: Configurable Class Filtering
- Currently filters YOLO detections for specific COCO class IDs (14=Bird, 16=Dog, 2=Car)
- Commented configuration in `src/saliency.py:27-30` and `main.py:11-14` enables quick video swapping
- HANDOFF_GUIDE.md roadmap includes `argparse` implementation for CLI-driven class and video selection

### Mid-Term: Spatiotemporal Video Salient Object Detection (VSOD)
**Transition from class-filtered YOLO to true autonomous saliency**:

| Current Approach | Future Approach |
|-----------------|-----------------|
| YOLOv8-Segmentation with COCO class filtering (e.g., class 14 = Bird) | VSOD models (STRA-Net, SAM 2) that detect salient objects without class pre-specification |
| Dual-stream: interim frame for detection, degraded frame for VSR | End-to-end spatiotemporal saliency from raw video frames |
| Bounding box routing from instance contours | Pixel-level saliency maps with temporal consistency |
| Single-object focus (one target class per run) | Multi-object simultaneous enhancement |

**Potential VSOD Models**:
- **STRA-Net** (Spatio-Temporal Recurrent Aggregation): Designed for video salient object detection with temporal consistency
- **SAM 2** (Segment Anything Model 2): Meta's successor to SAM, provides promptable video segmentation with state-of-the-art performance
- **PVP-Net**: Real-time video salient object detection pipeline

**Implementation Path**:
1. Replace YOLOv8-Segmentation backbone with VSOD model encoder
2. Maintain dual-stream degradation pipeline (frames remain 1080p→360p compressed)
3. Adapt router.py to work with continuous saliency masks instead of discrete bounding boxes
4. Modify stitcher.py alpha blending for soft, progressive mask transitions
5. Update metrics computation for per-pixel rather than per-ROI comparison

### Long-Term: End-to-End Trainable Asymmetric Upscaling
- Train a joint model that learns both foreground enhancement and background routing policies
- Optimize for compute-aware routing (learned percentage of frame to enhance vs. interpolate)
- End-to-end trainable alpha blending coefficients instead of fixed 15-pixel dilation + 15×15 Gaussian

## Directory Structure

```
saliency_vsr_project/
├── data/
│   ├── raw/              # Raw source videos (e.g., bird_video.mp4 from Google Drive)
│   └── processed/
│       ├── ground_truth_1080p/   # Extracted 1080p GT frames
│       ├── degraded_360p/        # Bilinear ↓360p + JPEG Q20 compressed frames
│       └── 1080p_Output/         # Generated outputs
│           ├── comparison_metrics_grid.jpg
│           └── saliency_heatmap.jpg
├── models/               # Required model weights (must be placed locally)
│   ├── yolov8n-seg.pt      # YOLOv8-Segmentation weights
│   ├── RealESRGAN_x4plus.pth  # Real-ESRGAN x4+ RRDBNet weights
│   └── EDSR_x3.pb          # OpenCV fallback weights (retained)
├── src/                  # Modular pipeline components
│   ├── main.py           # Pipeline orchestrator
│   ├── pipeline.py       # Frame extraction + degradation
│   ├── saliency.py       # YOLOv8-Seg perception (dual-stream)
│   ├── router.py         # Bounding box with 15px padding + clamping
│   ├── processors.py     # Real-ESRGAN + Unsharp Mask + Lanczos4
│   └── stitcher.py       # YOLO mask alpha blending + grid generation
├── agents.md             # Agent guidelines & architectural state
├── HANDOFF_GUIDE.md      # Teammate handoff guide
└── requirements.txt      # Python dependencies
```

## Installation & Setup

### Prerequisites
- **Python 3.10** installed and added to system PATH
- **Git** installed for repository cloning

### Clone & Environment
```powershell
# PowerShell
git clone <repository-url>
cd saliency_vsr_project
python -m venv venv
venv\Scripts\Activate.ps1
```

```bash
# Git Bash / Linux / macOS
git clone <repository-url>
cd saliency_vsr_project
python -m venv venv
source venv/Scripts/activate
```

### Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Model Assets
Create `models/` directory and place required weights:
```powershell
mkdir models
# Download/copy these three files into models/:
# - yolov8n-seg.pt (Ultralytics YOLOv8-Segmentation)
# - RealESRGAN_x4plus.pth (Real-ESRGAN x4+ weights)
# - EDSR_x3.pb (OpenCV fallback, retained)
```

### Raw Video Source
Download `bird_video.mp4` from team Google Drive and place in `data/raw/`:
```
saliency_vsr_project/
└── data/
    └── raw/
        └── bird_video.mp4
```

**Note**: Do NOT download pre-extracted frames — the pipeline handles frame extraction and degradation programmatically via `src/pipeline.py`.

## How to Run the Pipeline

### Baseline Execution
```powershell
python main.py
```

**What this command does**:
1. Reads `data/raw/bird_video.mp4`
2. Extracts 50 frames, saving ground-truth 1080p frames and realistic 360p degraded frames (bilinear downscale + JPEG Q20) into `data/processed/`
3. Runs dual-stream YOLOv8-Segmentation saliency extraction
4. Computes 15px-padded bounding boxes and processes the foreground through Real-ESRGAN (with BGR/RGB correction and Unsharp Mask texture recovery)
5. Routes the background through Lanczos4 interpolation and performs true YOLO mask-based alpha blending
6. Computes PSNR (~30.33 dB) and SSIM (~0.8898), saving the visual breakdown to `data/processed/1080p_Output/comparison_metrics_grid.jpg`

### Switching Test Videos
Comment/uncomment the `video_path` variable at the top of `main.py`:

```python
# --- SELECT YOUR TEST VIDEO HERE ---
video_path = "data/raw/bird_video.mp4"       # Default Bird Video (Target Class: 14)
# video_path = "data/raw/dog_video.mp4"      # Close-up Dog Video (Target Class: 16)
# video_path = "data/raw/car_video.mp4"      # Wide-shot Car Video (Target Class: 2)
```

**Target class IDs** (comment/uncomment in `src/saliency.py:27-30`):
```python
# --- SELECT YOUR TARGET COCO CLASS ID HERE ---
target_class_id = 14  # 14 = Bird
# target_class_id = 16  # 16 = Dog
# target_class_id = 2   # 2  = Car
```

### Expected Output
```
Pipeline initialized: 50 frames processed with realistic degradation.
[REALESRGAN] RRDBNet loaded on cpu from D:\saliency_vsr_project\models\RealESRGAN_x4plus.pth
[SUCCESS] Grid Saved: data/processed/1080p_Output/comparison_metrics_grid.jpg
Metrics -> PSNR: 30.33 dB | SSIM: 0.8898 | Compute Saved: 66.9%
```

The `comparison_metrics_grid.jpg` displays three panels:
- **Left**: 360p Degraded input with PSNR/SSIM label
- **Middle**: Spatial VSR Output with PSNR/SSIM label  
- **Right**: Ground Truth 1080p with "Ground Truth 1080p" label

## Code Quality & Operational Policies

### Zero-Fallback Policy
Strict prohibition against:
- Lazy hardcoded bounding boxes
- Mock thermal maps
- Center-biased default rectangles

**If** subject localization or saliency extraction fails, the pipeline **must** raise an explicit `ValueError` and halt execution cleanly rather than outputting garbage or masked debug placeholders.

### Dual-Stream Separation
- **Interim frame** (lightly Gaussian-blurred, σ=5.0): Used exclusively for YOLO mask detection — robust to compression artifacts
- **Degraded frame** (heavily JPEG-compressed, bilinear ↓360p): Feeds the actual VSR routing pipeline — preserves degradation intent
- **Never mix** the two streams' purposes

### Strict Boundary Clamping
- All bounding box coordinates clamped to frame dimensions using `max(0, ...)` / `min(dimension, ...)`
- Minimum 1-pixel width/height enforced
- No assumptions about subject position in frame

### Color Space Discipline
- **BGR→RGB** before tensor conversion (OpenCV loads BGR, Real-ESRGAN trained on RGB)
- **RGB→BGR** after inference before any OpenCV output operations
- Maintains natural bird brown tones (no color inversion to blue/green)

### Configurability Without CLI Parser
- Video path selection: Comment/uncomment lines at `main.py:11-14`
- Target class ID: Comment/uncomment lines at `src/saliency.py:27-30`
- Designed for quick testing of multiple videos (bird, dog, car) without building full argument parser

## Session-to-Session Continuity Checklist

When resuming work on this project, verify:

- [ ] `models/` directory contains all three required weight files
- [ ] `data/raw/bird_video.mp4` exists (or appropriate test video)
- [ ] Virtual environment is activated with correct package versions
- [ ] `agents.md` reflects current architectural state (update if major changes made)
- [ ] `requirements.txt` matches installed packages
- [ ] Zero-Fallback Policy provisions understood and respected
- [ ] BGR/RGB conversion points documented in `src/processors.py`
- [ ] Dual-stream saliency separation maintained in `src/saliency.py`
- [ ] 15-pixel bbox padding + clamping in `src/router.py`
- [ ] Unsharp Mask applied after 4×→3× downscaling in `src/processors.py:128-130`
- [ ] Lanczos4 background upscale in `src/processors.py:144`
- [ ] YOLO mask dilation + 15×15 Gaussian in `src/stitcher.py:92-98`
- [ ] Comparison grid panel order: 360p | VSR Output | GT 1080p in `src/stitcher.py:172-178`

---

*Project: Saliency-Guided On-Device Video Super-Resolution (Saliency-VSR)*
*Hard Spatial Routing Video Super-Resolution pipeline achieving asymmetric AI upscaling with 66.9% compute savings*
*Last verified: 30.33 dB PSNR | 0.8898 SSIM | 50-frame sequence execution*