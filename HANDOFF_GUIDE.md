# Hard Spatial Routing VSR Pipeline: Teammate Handoff & Execution Guide


Welcome to the project! This guide provides everything you need to clone, set up, and run the Hard Spatial Routing Video Super-Resolution (VSR) pipeline, as well as your first engineering tasks.


---


## 1. Prerequisites & Environment Setup


### System Requirements
- **Python 3.10** installed and added to your system path.
- **Git** installed.


### Step 1: Clone and Set Up Virtual Environment
Open your terminal (PowerShell on Windows or Git Bash) and run:


```powershell
# Clone the repository
git clone <repository-url>
cd saliency_vsr_project


# Create a local virtual environment named 'venv'
python -m venv venv


# Activate the virtual environment
# For Windows PowerShell:
venv\Scripts\Activate.ps1
# For Git Bash / Linux / macOS:
# source venv/Scripts/activate
```

### Step 2: Install Exact Dependencies
Make sure your virtual environment is active, then install the required packages:


PowerShell
pip install --upgrade pip
pip install -r requirements.txt

(If requirements.txt does not exist yet, create it in the root directory with the contents listed in Section 5 below).


2. Model Asset Management (Crucial)
All neural network weight files must reside inside the models/ folder at the root of the project.


Create the folder if it doesn't exist:


PowerShell
mkdir models
Ensure these three files are inside models/:


models/yolov8n-seg.pt (YOLOv8 Segmentation weights)


models/RealESRGAN_x4plus.pth (Real-ESRGAN x4+ weights)


models/EDSR_x3.pb (OpenCV fallback weights)


Note: The script will attempt to auto-download missing weights, but placing them locally prevents network blocks.


3. Google Drive Asset Instructions (Raw Video Only)
You only need to download the raw source video file from Google Drive.


Do not download pre-extracted frames. The pipeline handles frame extraction and degradation automatically.


Download bird_video.mp4 from the team Google Drive.


Place it inside the data/raw/ directory:


Plaintext
saliency_vsr_project/
└── data/
    └── raw/
        └── bird_video.mp4
4. How to Run the Pipeline (Commands)
Once your environment is set up, dependencies are installed, model weights are in place, and the video is in data/raw/, you can run the baseline pipeline with this exact command:


PowerShell
python main.py
What this command does:
Reads data/raw/bird_video.mp4.


Extracts 50 frames, saving ground-truth 1080p frames and realistic 360p degraded frames (bilinear downscale + JPEG Q20) into data/processed/.


Runs dual-stream YOLOv8-Segmentation saliency extraction.


Computes 15px-padded bounding boxes and processes the foreground through Real-ESRGAN (with BGR/RGB correction and Unsharp Mask texture recovery).


Routes the background through Lanczos4 interpolation and performs true YOLO mask-based alpha blending.


Computes PSNR (~30.33 dB) and SSIM (~0.8898), saving the visual breakdown to data/processed/1080p_Output/comparison_metrics_grid.jpg.


5. Required requirements.txt File
If your repository does not have a requirements.txt (or if it needs updating), create/update it in the root directory with these exact versions:


Plaintext
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.7.0
numpy>=1.23.0
scikit-image>=0.19.0
basicsr>=1.4.2
6. Your Next Engineering Task (Roadmap)
Your main objective is to move the project from a per-frame static grid generator to a full-video rendering pipeline with command-line arguments.


Task Breakdown:
Implement argparse in main.py so users can pass arguments:


PowerShell
python main.py --input data/raw/bird_video.mp4 --target-class 14 --output data/processed/output_video.mp4
Extend the loop to a full video writer (cv2.VideoWriter):


Loop over all extracted frames in data/processed/degraded_360p/.


Run the full pipeline (Saliency → Router → Real-ESRGAN Processor → Stitcher) on every frame.


Write the resulting 1080p enhanced frame into an mp4v video stream using cv2.VideoWriter.


Export the final playable file to --output.