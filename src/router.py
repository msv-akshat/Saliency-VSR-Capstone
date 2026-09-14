import cv2
import numpy as np

def compute_bounding_box(binary_mask):
    """Calculates tight bounding box coordinates from the binary mask with strict boundary clamping."""
    binary_mask_u8 = np.ascontiguousarray(binary_mask, dtype=np.uint8)
    h_img, w_img = binary_mask.shape
    
    contours, _ = cv2.findContours(binary_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return int(w_img * 0.2), int(h_img * 0.2), int(w_img * 0.6), int(h_img * 0.6)
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Clamp coordinates to stay completely inside image limits
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(10, min(w, w_img - x))
    h = max(10, min(h, h_img - y))
    
    return x, y, w, h