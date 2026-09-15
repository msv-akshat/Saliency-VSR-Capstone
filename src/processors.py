import sys
import os
import cv2
import numpy as np
import torch

# --- Monkey-patch to fix broken basicsr import in modern PyTorch ---
import torchvision
import torchvision.transforms.functional as TF
sys.modules['torchvision.transforms.functional_tensor'] = TF
# ------------------------------------------------------------

from basicsr.archs.rrdbnet_arch import RRDBNet


# ---------------------------------------------------------------------------
# Model initialisation (lazy – executed once per process)
# ---------------------------------------------------------------------------
_model = None
_device = None


def _init_model(device="cpu"):
    """Load RRDBNet and its pre-trained weights directly via torch.load.

    Returns the eval model on the requested device.  The import patch above
    ensures `basicsr.archs.rrdbnet_arch.RRDBNet` resolves without the
    `torchvision.transforms.functional_tensor` ModuleNotFoundError.
    """
    global _model, _device
    if _model is not None:
        return _model, _device

    _device = torch.device(device)

    # RRDBNet architecture
    _model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    ).to(_device)

    # ------------------------------------------------------------------
    # Load weights – the Real-ESRGAN x4+ model stores its state under
    # 'params_ema' (ema of the RRDBNet weights).  Fall back to plain 'params'.
    # ------------------------------------------------------------------
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "RealESRGAN_x4plus.pth",
    )
    loadnet = torch.load(model_path, map_location=_device)

    if "params_ema" in loadnet:
        _model.load_state_dict(loadnet["params_ema"], strict=False)
    elif "params" in loadnet:
        _model.load_state_dict(loadnet["params"], strict=False)
    else:
        raise RuntimeError("Unknown state‑dict key in Real-ESRGAN model file.")

    _model.eval()
    print(f"[REALESRGAN] RRDBNet loaded on {_device} from {model_path}")
    return _model, _device


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def upscale_foreground(roi_360p):
    """Enhance the foreground ROI using direct PyTorch inference on RRDBNet.

    Steps:
    1. Initialise model (once).
    2. Preprocess ROI: BGR→float32→[0,1]→CHW→BCHW → send to device.
    3. Run model inside torch.no_grad().
    4. Post‑process: squeeze batch, clamp[0,1], CHW→HWC, *255, uint8.
    5. Resize the 4×‑upsampled patch to exactly 3× original size
       with cv2.INTER_AREA to match our 1080p stitcher canvas.

    Zero-Fallback Policy: raise ValueError if any step fails.
    """
    _init_model(device="cuda" if torch.cuda.is_available() else "cpu")

    h, w = roi_360p.shape[:2]
    target_w, target_h = w * 3, h * 3

    try:
        # ---- Preprocess: BGR HWC -> float32 CHW -> BCHW -----------------
        # Convert to float32 and normalise to [0, 1]
        img = roi_360p.astype(np.float32) / 255.0

        # BGR -> RGB (Real-ESRGAN was trained on RGB)
        img = img[:, :, ::-1]

        # HWC -> CHW
        img_chw = np.ascontiguousarray(img.transpose(2, 0, 1))

        # Add batch dimension → BCHW tensor
        tensor_in = torch.from_numpy(img_chw).unsqueeze(0).to(_device)

        # ---- Inference ---------------------------------------------------
        with torch.no_grad():
            output = _model(tensor_in)          # output: BCHW

        # ---- Postprocess -------------------------------------------------
        # Remove batch dim: (1,3,H,W) -> (3,H,W)
        output = output.squeeze(0)
        # Clamp to [0, 1], move to CPU, convert to numpy
        output = torch.clamp(output, 0, 1).cpu()
        # CHW -> HWC, *255, uint8
        output_np = output.numpy()
        output_hwc = np.transpose(output_np, (1, 2, 0))  # HWC
        output_uint8 = (output_hwc * 255.0).astype(np.uint8)

        # ---- 4x → 3x downscale -----------------------------------------
        enhanced_3x = cv2.resize(
            output_uint8, (target_w, target_h), interpolation=cv2.INTER_AREA
        )
        return enhanced_3x

    except Exception as e:
        print(f"[WARNING] Direct Real-ESRGAN inference failed ({e}).")
        raise ValueError(
            "Direct Real-ESRGAN processing failed. "
            "Cannot compute enhanced ROI without valid model output. "
            "Check model compatibility."
        )


def resize_background(frame_360p, target_shape=(1920, 1080)):
    """Up-scale the full 360p frame to 1080p using bicubic interpolation."""
    return cv2.resize(frame_360p, target_shape, interpolation=cv2.INTER_CUBIC)