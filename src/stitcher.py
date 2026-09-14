import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def alpha_blend_stitch(background_1080p, roi_1080p, saliency_map, bbox):
    """Stitch the EDSR-upscaled foreground ROI onto the 1080p background using smooth Gaussian feathering.
    
    Eliminates visible rectangular seams by applying a Gaussian-weighted alpha blend
    around the ROI edges where it merges into the bicubic background canvas. The
    saliency map provides the feathering gradient. No hard-coded boundary lines are drawn.
    
    Key flow:
    1. Place the upscaled ROI onto a blank canvas at the correct position
    2. Build a full-frame feather mask derived from the saliency map
    3. Alpha-blend the canvas using the feather mask for seamless transitions
    """
    x, y, w, h = bbox

    # --- Step 1: Place ROI on canvas ---
    canvas_1080p = np.zeros((1080, 1920, 3), dtype=np.float32)

    target_x, target_y = x * 3, y * 3

    patch_h, patch_w = roi_1080p.shape[:2]

    end_x = min(target_x + patch_w, 1920)
    end_y = min(target_y + patch_h, 1080)

    actual_w = end_x - target_x
    actual_h = end_y - target_y

    if actual_w > 0 and actual_h > 0:
        # Resize ROI to exactly fit the target area (already 3x-upsampled from 360p)
        if roi_1080p.shape[:2] != (actual_h, actual_w):
            roi_1080p = cv2.resize(roi_1080p, (actual_w, actual_h), interpolation=cv2.INTER_LINEAR)
        canvas_1080p[target_y:end_y, target_x:end_x] = roi_1080p
    # --- End Step 1 ---

    # --- Step 2: Build feather mask from saliency map ---
    # Resize saliency map to match the ROI region dimensions
    roi_region_height = end_y - target_y
    roi_region_width = end_x - target_x
    saliency_roi = cv2.resize(saliency_map, (roi_region_width, roi_region_height), interpolation=cv2.INTER_LINEAR)
    # Normalize to [0, 1]
    saliency_norm = saliency_roi / 255.0 if saliency_roi.max() > 1 else saliency_roi
    # Apply Gaussian blur to create smooth feather gradient at edges
    feather_sigma = max(1, min(roi_region_width, roi_region_height) // 10)
    feather_mask = cv2.GaussianBlur(saliency_norm, (0, 0), feather_sigma)
    # Clip to [0, 1] range
    feather_mask = np.clip(feather_mask, 0, 1)
    # --- End Step 2 ---

    # --- Step 3: Apply full-frame feather mask ---
    # Start with ones (no feathering = full ROI), then mask the edges
    full_feather = np.ones((1080, 1920), dtype=np.float32)
    # Apply the feathered mask only over the ROI region on the canvas
    full_feather[target_y:end_y, target_x:end_x] = feather_mask
    # Expand to 3 channels for broadcasting
    alpha_3d = np.repeat(full_feather[:, :, np.newaxis], 3, axis=2)
    # --- End Step 3 ---

    # --- Step 4: Alpha blend using the feather mask ---
    # canvas_1080p is float, background is uint8 -> convert background to float
    background_float = background_1080p.astype(np.float32)
    # Feathered blend: where feather_mask=1.0 → full ROI, where 0.0 → pure background
    final_output = (alpha_3d * canvas_1080p) + ((1.0 - alpha_3d) * background_float)
    return final_output.astype(np.uint8)


def calculate_metrics(frame_gt, final_output, bbox):
    total_pixels = 640 * 360

    if len(bbox) == 4:
        w, h = bbox[2], bbox[3]
    elif len(bbox) == 2:
        w, h = bbox[0], bbox[1]
    else:
        raise ValueError("calculate_metrics received bbox with invalid length. "
                         "Expected 4-tuple (x, y, w, h) or 2-tuple (w, h).")

    roi_pixels = max(1, w * h)
    compute_saved = max(0.0, 100.0 - ((roi_pixels / total_pixels) * 100.0))

    if frame_gt.shape != final_output.shape:
        h_gt, w_gt = frame_gt.shape[:2]
        final_output = cv2.resize(final_output, (w_gt, h_gt), interpolation=cv2.INTER_LINEAR)

    psnr_score = cv2.PSNR(frame_gt, final_output)
    return psnr_score, compute_saved


def generate_comparison_grid(frame_360p, frame_gt, final_output, psnr, compute_saved):
    h_gt, w_gt = frame_gt.shape[:2]

    if final_output.shape[:2] != (h_gt, w_gt):
        final_output = cv2.resize(final_output, (w_gt, h_gt), interpolation=cv2.INTER_LINEAR)

    frame_360_resized = cv2.resize(frame_360p, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC)

    gray_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2GRAY)
    gray_out = cv2.cvtColor(final_output, cv2.COLOR_BGR2GRAY)
    ssim_score = ssim(gray_gt, gray_out)

    gray_360 = cv2.cvtColor(frame_360_resized, cv2.COLOR_BGR2GRAY)
    ssim_360 = ssim(gray_gt, gray_360)
    psnr_360 = cv2.PSNR(frame_gt, frame_360_resized)

    labels = [
        f"360p (PSNR: {psnr_360:.1f}dB | SSIM: {ssim_360:.2f})",
        f"Ground Truth 1080p",
        f"Spatial VSR (PSNR: {psnr:.1f}dB | SSIM: {ssim_score:.2f})"
    ]

    panels = [frame_360_resized, frame_gt, final_output]
    display_w = 640
    display_h = int(display_w * (h_gt / w_gt))

    resized_panels = []
    for panel, label in zip(panels, labels):
        resized = cv2.resize(panel, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
        annotated = resized.copy()

        # Place label banner at the BOTTOM of each panel
        cv2.rectangle(annotated, (10, display_h - 45), (display_w - 10, display_h - 10), (0, 0, 0), -1)
        cv2.putText(annotated, label, (18, display_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        resized_panels.append(annotated)

    grid = np.hstack(resized_panels)

    banner = np.zeros((50, grid.shape[1], 3), dtype=np.uint8)
    summary_text = f"AI Compute Saved: {compute_saved:.1f}% | Hard Spatial Routing Pipeline"
    cv2.putText(banner, summary_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    final_grid = np.vstack([grid, banner])
    return final_grid, ssim_score, psnr