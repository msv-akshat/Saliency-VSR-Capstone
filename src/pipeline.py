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
        
        # Genuine degradation: Downscale to 360p (640x360) using rough interpolation & compression artifacting
        # Step 1: Slight blur to destroy fine high-frequency details
        blurred = cv2.GaussianBlur(frame, (5, 5), 1.5)
        # Step 2: Downscale to 360p
        frame_360p = cv2.resize(blurred, (640, 360), interpolation=cv2.INTER_NEAREST)
        
        degraded_path = os.path.join(degraded_dir, f"frame_{count:04d}.png")
        cv2.imwrite(degraded_path, frame_360p)
        
        count += 1
        
    cap.release()
    print(f"Pipeline initialized: {count} frames processed with heavy degradation.")
    return gt_dir, degraded_dir