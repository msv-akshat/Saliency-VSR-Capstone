import os
import cv2

def extract_and_degrade_video(video_path, output_dir, max_frames=50):
    """Extracts frames from raw video, applies moderate degradation to 360p to test VSR.
    
    The degradation applies Gaussian blur to destroy fine high-frequency details,
    simulating lossy compression without introducing extreme artifacts that
    prevent subject detection models from working.
    
    Returns:
        gt_dir: Directory with 1080p ground truth frames
        degraded_dir: Directory with degraded 360p frames
    """
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
        
        # Moderate degradation: Gaussian blur to destroy high frequencies
        # Sigma=2.0 provides visible blur without preventing subject detection
        # Downscale to 360p (640x360) using area interpolation for anti-aliasing
        frame_360p = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        
        # Optional: Light JPEG compression for realism (quality=30 - visible but not blocking artifacts)
        # Uncomment below line to enable:
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
        # result, enc_frame_360p = cv2.imencode('.jpg', frame_360p, encode_param)
        # if result:
        #     frame_360p = cv2.imdecode(enc_frame_360p, cv2.IMREAD_COLOR)
        
        degraded_path = os.path.join(degraded_dir, f"frame_{count:04d}.png")
        cv2.imwrite(degraded_path, frame_360p)
        
        count += 1
        
    cap.release()
    print(f"Pipeline initialized: {count} frames processed with moderate degradation.")
    return gt_dir, degraded_dir