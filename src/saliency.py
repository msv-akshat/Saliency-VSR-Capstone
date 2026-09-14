import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def generate_saliency_map(frame_360p_bgr):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "deeplab_v3.tflite")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing at: {model_path}.")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
    segmenter = vision.ImageSegmenter.create_from_options(options)
    
    frame_rgb = cv2.cvtColor(frame_360p_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    segmentation_result = segmenter.segment(mp_image)
    
    category_mask = segmentation_result.category_mask.numpy_view()
    if category_mask.ndim == 3:
        category_mask = category_mask[0]
        
    binary_mask = (category_mask > 0).astype(np.uint8)
    
    # Absolute safeguard: Ensure contours are always found by enforcing a center ROI if empty
    if np.sum(binary_mask) == 0:
        h_img, w_img = binary_mask.shape
        binary_mask[int(h_img * 0.1):int(h_img * 0.9), int(w_img * 0.1):int(w_img * 0.9)] = 1

    saliency_map = cv2.GaussianBlur(binary_mask.astype(np.float32), (15, 15), 0)
    return binary_mask, saliency_map