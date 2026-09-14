import os
import cv2
import numpy as np

# Sharpening kernel for post-processing
SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

def upscale_foreground(roi_360p, model_path=None):
    """Passes the isolated foreground patch through the EDSR VSR network with bilateral pre-filter and sharpening."""
    if roi_360p is None or roi_360p.size == 0 or roi_360p.shape[0] == 0 or roi_360p.shape[1] == 0:
        return np.zeros((30, 30, 3), dtype=np.uint8)
        
    # Apply bilateral filter to smooth JPEG block boundaries while preserving edges
    roi_filtered = cv2.bilateralFilter(roi_360p, d=9, sigmaColor=75, sigmaSpace=75)
    
    h, w = roi_360p.shape[:2]
    target_w, target_h = w * 3, h * 3
    
    if model_path is None or not os.path.exists(model_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "EDSR_x3.pb")
        
    try:
        if os.path.exists(model_path):
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(model_path)
            sr.setModel('edsr', 3)
            output = sr.upsample(roi_filtered)
            if output is not None and output.size > 0:
                # Apply sharpening filter to maximize visual distinctness
                output = cv2.filter2D(output, -1, SHARPEN_KERNEL)
                return output
    except Exception as e:
        print(f"[WARNING] EDSR model execution failed ({e}). Falling back to bicubic for patch.")
        
    # Fallback: bilateral + bicubic upscale
    return cv2.resize(roi_filtered, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

def resize_background(frame_360p, target_shape=(1920, 1080)):
    return cv2.resize(frame_360p, target_shape, interpolation=cv2.INTER_CUBIC)