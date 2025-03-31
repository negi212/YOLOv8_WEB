import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Streamlit app setup
st.title("Real-time Object Detection (YOLOv8 + Web Camera)")
start_button = st.button("Start Camera")
stop_button = st.button("Stop Camera")
FRAME_WINDOW = st.image([])

# Camera setup
camera = cv2.VideoCapture(1)

# App state
if 'running' not in st.session_state:
    st.session_state['running'] = False

# Button actions
if start_button:
    st.session_state['running'] = True
if stop_button:
    st.session_state['running'] = False

# Real-time object detection
if st.session_state['running']:
    while True:
        _, frame = camera.read()
        if frame is None:
            st.write("Camera disconnected or unavailable.")
            st.session_state['running'] = False
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)
        annotated_frame = results[0].plot()
        FRAME_WINDOW.image(annotated_frame, channels="RGB")
        if not st.session_state['running']:
            break
else:
    st.write("Camera stopped.")

camera.release()