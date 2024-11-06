import torch
import logging
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import imageio
import imageio_ffmpeg as iio

# Set up logging
logging.basicConfig(level=logging.INFO)

# Path to the YOLO model
model_path = './models/best.pt'
logging.info(f"Model loaded from {model_path}")
model = YOLO(model_path)

# Streamlit app setup
st.title("YOLO Object Detection Stream")
st.text("This app shows live object detection on webcam feed.")

# Initialize Streamlit elements
image_placeholder = st.empty()

def process_frame():
    # Set up video reader
    reader = imageio.get_reader("<video0>")  # Use "<video0>" for the first camera device

    frame_count = 0
    try:
        for frame in reader:
            frame_count += 1
            logging.info(f"Processing frame {frame_count}")
            
            # Resize frame for processing
            pil_frame = Image.fromarray(frame)
            resized_frame = pil_frame.resize((640, 640))
            tensor_frame = torch.from_numpy(np.array(resized_frame)).to('cuda' if torch.cuda.is_available() else 'cpu').permute(2, 0, 1).float() / 255.0
            tensor_frame = tensor_frame.unsqueeze(0)

            results = model(tensor_frame, device='cuda' if torch.cuda.is_available() else 'cpu', conf=0.1)

            # Check for detections
            if results and len(results[0].boxes) > 0:
                annotated_frame = results[0].plot()
                annotated_frame_rgb = Image.fromarray(annotated_frame)
            else:
                annotated_frame_rgb = resized_frame

            # Display the frame in Streamlit
            image_placeholder.image(annotated_frame_rgb, use_column_width=True)
            
    finally:
        reader.close()

# Run the frame processing in Streamlit
process_frame()
