import urllib.request
import os

os.makedirs("models", exist_ok=True)
url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite"
urllib.request.urlretrieve(url, "models/deeplabv3.tflite")
print("Model downloaded successfully!")