import os
import cv2

def extract_and_degrade_video(video_path, output_dir, max_frames=50):
    """Extracts frames from raw video, strictly downsamples to 360p with heavy degradation to test VSR."""
    gt_dir = os.path.join(output_dir, "ground_truth_1080p")
    degraded_dir = os.path.join(output_dir, "degraded_360p")
    
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(degraded_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Ensure base frame is 1920x1080 ground truth
        h, w = frame.shape[:2]
        if (w, h) != (1920, 1080):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
            
        gt_path = os.path.join(gt_dir, f"frame_{count:04d}.png")
        cv2.imwrite(gt_path, frame)
        
        # Aggressive degradation: multi-stage downsampling with extreme pixelation & loss
        # Step 1: Heavy blur to destroy high frequencies (σ ≥ 3.0)
        blurred = cv2.GaussianBlur(frame, (5, 5), 3.0)
        # Step 2: Downscale to tiny size (160x90) using nearest-neighbor → massive blocky pixels
        tiny = cv2.resize(blurred, (160, 90), interpolation=cv2.INTER_NEAREST)
        # Step 3: Upscale back to 640x360 using nearest-neighbor → preserves blocky pixelation + aliasing
        frame_360p = cv2.resize(tiny, (640, 360), interpolation=cv2.INTER_NEAREST)
        
        # Aggressive JPEG compression (quality ≤ 10) to add compression artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
        result, enc_frame_360p = cv2.imencode('.jpg', frame_360p, encode_param)
        if result:
            frame_360p = cv2.imdecode(enc_frame_360p, cv2.IMREAD_COLOR)
        
        # Verification: Compute mean pixel difference between 360p and up-sampled GT
        gt_cv = cv2.imread(gt_path)
        gt_upscaled = cv2.resize(gt_cv, (640, 360), interpolation=cv2.INTER_LINEAR)
        diff = cv2.absdiff(gt_upscaled, frame_360p)
        mean_diff = cv2.mean(diff)[0]
        print(f"[VERIFY] 360p shape: {frame_360p.shape} | Mean pixel diff vs GT: {mean_diff:.1f}")
        
        degraded_path = os.path.join(degraded_dir, f"frame_{count:04d}.png")
        cv2.imwrite(degraded_path, frame_360p)
        
        count += 1
        
    cap.release()
    print(f"Pipeline initialized: {count} frames processed with heavy degradation.")
    return gt_dir, degraded_dir