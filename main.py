import os
import cv2
import numpy as np
from src.pipeline import extract_and_degrade_video
from src.saliency import generate_saliency_map
from src.router import compute_bounding_box
from src.processors import upscale_foreground, resize_background
from src.stitcher import alpha_blend_stitch, calculate_metrics, generate_comparison_grid

def run_pipeline():
    video_path = "data/raw/bird_video.mp4"
    output_dir = "data/processed"
    
    gt_dir, degraded_dir = extract_and_degrade_video(video_path, output_dir, max_frames=50)
    
    frame_path_360 = os.path.join(degraded_dir, "frame_0000.png")
    frame_path_gt = os.path.join(gt_dir, "frame_0000.png")
    
    frame_360 = cv2.imread(frame_path_360)  # Severely degraded frame for VSR
    frame_gt = cv2.imread(frame_path_gt)
    
    # === Dual-Stream Saliency Extraction ===
    # Create a lightly-filtered/interim copy of the frame for YOLO mask detection.
    # This prevents YOLOv8 from failing on extreme JPEG/compression artifacts,
    # while the original severely-degraded frame feeds the actual VSR routing pipeline.
    # Light Gaussian blur (σ=5.0) restores some high-frequency info for detection
    # without compromising the VSR degradation intent.
    frame_interim = cv2.GaussianBlur(frame_360, (0, 0), 5.0)
    # Optional: slight bicubic resize then back could also help, but simple blur suffices
    
    try:
        binary_mask, saliency_map = generate_saliency_map(frame_360, frame_interim_bgr=frame_interim)
    except ValueError as e:
        print(f"[ERROR] Saliency extraction failed: {e}")
        print("Cannot proceed with Hard Spatial Routing without a detected foreground subject.")
        return
    # =========================================
    
    x, y, w, h = compute_bounding_box(binary_mask)
    roi_360 = frame_360[y:y+h, x:x+w]
    
    model_path = os.path.join(os.path.dirname(__file__), "models", "EDSR_x3.pb")
    roi_1080 = upscale_foreground(roi_360, model_path=model_path)
    bg_1080 = resize_background(frame_360, target_shape=(1920, 1080))
    
    final_output = alpha_blend_stitch(bg_1080, roi_1080, saliency_map, (x, y, w, h))
    
    # Pass the full 4-tuple bbox (x, y, w, h) for accurate pixel math
    psnr, compute_saved = calculate_metrics(frame_gt, final_output, (x, y, w, h))
    
    comparison_grid, ssim_score, _ = generate_comparison_grid(frame_360, frame_gt, final_output, psnr, compute_saved)
    
    comparison_path = os.path.join(output_dir, "1080p_Output", "comparison_metrics_grid.jpg")
    cv2.imwrite(comparison_path, comparison_grid)
    
    print(f"\n[SUCCESS] Grid Saved: {comparison_path}")
    print(f"Metrics -> PSNR: {psnr:.2f} dB | SSIM: {ssim_score:.4f} | Compute Saved: {compute_saved:.1f}%")

if __name__ == "__main__":
    run_pipeline()