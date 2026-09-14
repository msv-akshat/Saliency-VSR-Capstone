# Hard Spatial Routing Video Super-Resolution (VSR)

## Architecture Overview
- **Layer 1 (Saliency):** MediaPipe DeepLabV3 generates a pixel-wise probability mask from 360p input frames.
- **Layer 2 (Spatial Router):** NumPy custom logic thresholds the mask to extract target bounding boxes via contour analysis.
- **Layer 3 (Asymmetric Processing):** Isolated foreground patches pass through ESPCN ($3\times$ deep neural network), while backgrounds pass through cheap bicubic upscaling.
- **Layer 4 (Stitcher):** Continuous alpha blending merges the tensors back into a 1080p canvas, eliminating boundary seams.

## Execution
Run the complete end-to-end pipeline via:
python main.py