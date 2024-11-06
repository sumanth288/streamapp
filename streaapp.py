import torch
import cv2
import logging
import numpy as np
import streamlit as st
from ultralytics import YOLO
# import winsound  # For playing beep sound on detection (Windows only)

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
    # Initialize webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open webcam.")
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to read frame from webcam.")
            break
        
        frame_count += 1
        logging.info(f"Processing frame {frame_count}")
        
        # Resize frame for processing
        resized_frame = cv2.resize(frame, (640, 640))
        tensor_frame = torch.from_numpy(np.expand_dims(resized_frame, 0)).to('cuda' if torch.cuda.is_available() else 'cpu').permute(0, 3, 1, 2).float() / 255.0
        results = model(tensor_frame, device='cuda' if torch.cuda.is_available() else 'cpu', conf=0.1)
        
        # Check for detections
        if results and len(results[0].boxes) > 0:
            annotated_frame = results[0].plot()
            
            # Play beep sound
            # winsound.Beep(1000, 500)  # Frequency: 1000 Hz, Duration: 500 ms
        else:
            annotated_frame = resized_frame
        
        # Convert frame to RGB for Streamlit display
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame in Streamlit
        image_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()

# Run the frame processing in Streamlit
process_frame()
