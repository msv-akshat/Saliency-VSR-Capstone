import cv2
import numpy as np
from skimage.metrics import structural_similarity as sssim


def alpha_blend_stitch(background_1080p, roi_1080p, saliency_map, bbox):
    """Stitch the EDR/Real-ESRGAN-enhanced foreground ROI onto the full 1080p background canvas
    using the true YOLO segmentation mask for alpha blending.

    Key operations:
    1. Build full 1080p base canvas from background (all background, no ROI)
    2. Place EDR/Real-ESRGAN-enhanced foreground ROI at 3x-scaled position from 360p bbox
    3. Extract the true saliency mask region, scale by 3x via cv2.INTER_LINEAR,
       Gaussian-feedter with ksize=(7,7), and blend the enhanced ROI smoothly over
       the background canvas using the actual foreground shape from YOLO segmentation –
       eliminating rectangular seams and edge artifacts.
    4. The surrounding background (trees, leaves, branches) remains fully intact
       across the entire 1080p frame.

    Parameters
    ----------
    background_1080p : np.ndarray
        The full 1080p background canvas (bicubic-upscaled 360p, float32 or uint8).
    roi_1080p : np.ndarray
        The EDR/Real-ESRGAN-enhanced foreground patch at 3x scale, shape (H, W, 3),
        uint8, matching the 3x-scaled bbox dimensions.
    saliency_map : np.ndarray
        Normalised float32 saliency map from YOLOv8-Segmentation, shape (360, 640),
        values in [0.0, 1.0].
    bbox : tuple of int
        Bounding box (x, y, w, h) in 360p coordinate space.

    Returns
    -------
    np.ndarray
        The stitched 1080p output image (uint8).
    """
    x, y, w, h = bbox

    # ---------------------------------------------------------------
    # Step 1: Build full 1080p base canvas from background
    # ---------------------------------------------------------------
    if background_1080p.shape != (1080, 1920, 3):
        canvas_1080p = cv2.resize(background_1080p, (1920, 1080), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    else:
        canvas_1080p = background_1080p.astype(np.float32)

    # ---------------------------------------------------------------
    # Step 2: Place EDR/Real-ESRGAN-enhanced foreground ROI at 3x position
    # ---------------------------------------------------------------
    # Scale bbox from 360p to 1080p (factor of 3)
    target_x, target_y = x * 3, y * 3

    patch_h, patch_w = roi_1080p.shape[:2]

    end_x = min(target_x + patch_w, 1920)
    end_y = min(target_y + patch_h, 1080)

    actual_w = end_x - target_x
    actual_h = end_y - target_y

    # Save the original background patch that will be overwritten
    orig_patch = canvas_1080p[target_y:end_y, target_x:end_x].copy()

    if actual_w > 0 and actual_h > 0:
        # Resize ROI to exactly fit the target area
        if roi_1080p.shape[:2] != (actual_h, actual_w):
            roi_1080p_resized = cv2.resize(roi_1080p, (actual_w, actual_h), interpolation=cv2.INTER_LINEAR)
        else:
            roi_1080p_resized = roi_1080p
        canvas_1080p[target_y:end_y, target_x:end_x] = roi_1080p_resized.astype(np.float32)
    # ---------------------------------------------------------------
    # Step 3: Extract the true YOLO segmentation mask, scale & feather
    # ---------------------------------------------------------------
    # The saliency_map is (360, 640) float32 [0,1], same as 360p frame resolution.
    # Extract the region within the 360p bbox (clip to image bounds).
    y1 = max(0, y)
    y2 = min(360, y + h)
    x1 = max(0, x)
    x2 = min(640, x + w)

    # Extract the saliency mask inside the bbox
    if y2 > y1 and x2 > x1:
        mask_roi = saliency_map[y1:y2, x1:x2]  # shape: (h_bb, w_bb), float32 [0,1]
    else:
        mask_roi = np.zeros((1, 1), dtype=np.float32)

    # Scale this mask crop by 3x to match the exact dimensions of the
    # EDR/Real-ESRGAN-enhanced ROI (which is 3x the original 360p ROI size).
    mask_scaled = cv2.resize(mask_roi, (actual_w, actual_h), interpolation=cv2.INTER_LINEAR)

    # Dilate the mask slightly to protect the subject's boundaries (head/beak)
    # from being eaten away by the Gaussian blur.
    mask_dilated = cv2.dilate(mask_scaled, np.ones((5, 5), np.uint8), iterations=2)

    # Apply Gaussian blur for smooth edge feathering – ksize=(15,15) gives
    # an ultra-smooth transition while the preceding dilation preserves edge.
    mask_blurred = cv2.GaussianBlur(mask_dilated, (15, 15), 0)

    # Clip to valid [0, 1] range
    mask_blurred = np.clip(mask_blurred, 0, 1)

    # Expand mask to 3 channels so it broadcasts over the RGB ROI
    mask_3ch = np.repeat(mask_blurred[:, :, np.newaxis], 3, axis=2)

    # ---------------------------------------------------------------
    # Step 4: Exact pixel blending using the true YOLO mask
    # ---------------------------------------------------------------
    # The canvas already has the ROI placed (Step 2).  Blend the ROI with the
    # original background patch using the feathered mask:
    #       blended = mask_3ch * roi_region + (1 - mask_3ch) * orig_patch
    blended_region = (mask_3ch * canvas_1080p[target_y:end_y, target_x:end_x].astype(np.float32) +
                      (1.0 - mask_3ch) * orig_patch.astype(np.float32))

    # Place the blended region back onto the output canvas
    output = canvas_1080p.copy()
    output[target_y:end_y, target_x:end_x] = blended_region

    # Outside the ROI region, the output is pure original background (already in canvas)
    # No additional processing needed
    # ---------------------------------------------------------------

    return output.astype(np.uint8)


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

    # cv2.PSNR expects images of the same type; convert to uint8 if needed
    if frame_gt.dtype != final_output.dtype:
        frame_gt_u8 = cv2.convertScaleAbs(frame_gt)
        final_output_u8 = cv2.convertScaleAbs(final_output)
    else:
        frame_gt_u8 = frame_gt
        final_output_u8 = final_output

    psnr_score = cv2.PSNR(frame_gt_u8, final_output_u8)
    return psnr_score, compute_saved


def generate_comparison_grid(frame_360p, frame_gt, final_output, psnr, compute_saved):
    h_gt, w_gt = frame_gt.shape[:2]

    if final_output.shape[:2] != (h_gt, w_gt):
        final_output = cv2.resize(final_output, (w_gt, h_gt), interpolation=cv2.INTER_LINEAR)

    frame_360_resized = cv2.resize(frame_360p, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC)

    gray_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2GRAY)
    gray_out = cv2.cvtColor(final_output, cv2.COLOR_BGR2GRAY)
    ssim_score = sssim(gray_gt, gray_out)

    gray_360 = cv2.cvtColor(frame_360_resized, cv2.COLOR_BGR2GRAY)
    ssim_360 = sssim(gray_gt, gray_360)
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