import os
import cv2
import numpy as np
from ultralytics import YOLO

def generate_saliency_map(frame_360p_bgr):
    """Extract foreground saliency mask using YOLOv8-Segmentation model.
    
    Production-grade subject localization for Hard Spatial Routing VSR pipeline.
    
    Key operations:
    1. Loads YOLOv8n-seg model from models/yolov8n-seg.pt
    2. Runs inference on 360p frame, filtering for bird class (COCO class == 14)
    3. Extracts instance segmentation mask for the detected bird
    4. Resizes mask to native 360p resolution (640x360) using bilinear interpolation
    5. Normalizes mask to [0.0, 1.0] float saliency map
    6. Zero-Fallback Policy: raises ValueError if no bird detected (no fake boxes)
    
    Returns:
        binary_mask: np.uint8 array of shape (360, 640) with 0/1 values
        saliency_map: np.float32 array of shape (360, 640) with values in [0.0, 1.0]
    """
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "yolov8n-seg.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLOv8-Seg model not found at: {model_path}")
    
    # Load the YOLOv8-Seg model
    model = YOLO(model_path)
    
    # Run inference on the 360p frame
    # verbose=False suppresses the training-style output
    results = model(frame_360p_bgr, verbose=False)
    
    # Filter detections for bird class (COCO class 14)
    bird_masks = []
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            # Get class indices
            classes = boxes.cls.cpu().numpy().astype(int)
            # Filter for bird class (COCO class 14)
            bird_indices = [i for i, cls in enumerate(classes) if cls == 14]
            
            if bird_indices:
                # Extract segmentation mask for each bird detection
                for idx in bird_indices:
                    # Get the mask for this detection
                    mask = result.masks.data[idx].cpu().numpy()  # Shape: (H, W) - already at frame resolution
                    bird_masks.append(mask)
    
    # Zero-Fallback Policy: if no bird detected, raise explicit ValueError
    if not bird_masks:
        raise ValueError("YOLOv8-Segmentation detected no bird in the frame. "
                         "Cannot compute saliency mask without a detected foreground subject. "
                         "Check if the bird is visible in the degraded frame.")
    
    # Combine all bird masks (take the union / largest mask)
    # Stack masks and take the maximum (union of all bird detections)
    combined_mask = np.max(np.stack(bird_masks), axis=0)  # Shape: (360, 640)
    
    # Threshold to ensure binary mask (0 or 1)
    binary_mask = (combined_mask > 0.5).astype(np.uint8)
    
    # Resize to native 360p resolution (640x360) using bilinear interpolation
    # The mask may already be at 360p, but ensure consistent dimensions
    if binary_mask.shape != (360, 640):
        binary_mask = cv2.resize(binary_mask, (640, 360), interpolation=cv2.INTER_LINEAR)
        binary_mask = (binary_mask > 0.5).astype(np.uint8)
    
    # Normalize to [0.0, 1.0] float32 saliency map
    saliency_map = binary_mask.astype(np.float32)
    
    return binary_mask, saliency_map