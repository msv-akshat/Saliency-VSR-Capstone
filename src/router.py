import cv2
import numpy as np

def compute_bounding_box(binary_mask):
    """Compute the tightest possible bounding box from a binary foreground mask.
    
    Zero-Fallback Policy: if no foreground contours are detected, raise ValueError
    instead of defaulting to a hardcoded rectangular mock box. The caller must
    handle this exception or the pipeline stops — no silent fallbacks.
    """
    binary_mask_u8 = np.ascontiguousarray(binary_mask, dtype=np.uint8)
    h_img, w_img = binary_mask.shape

    contours, _ = cv2.findContours(binary_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No foreground contours found in binary mask. "
                         "Saliency extraction failed to detect any subject. "
                         "Cannot compute bounding box without valid contour data.")

    # Select the largest contour as the primary subject
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Strict clamping to image boundaries — no mock dimension defaults
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))  # enforce at least 1 pixel width
    h = max(1, min(h, h_img - y))  # enforce at least 1 pixel height

    return x, y, w, h