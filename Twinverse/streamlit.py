# Import necessary libraries
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# Initializing the Mediapipe Pose Estimator
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Streamlit UI for best user interface
st.title("Live Camera - Lighting, Pose, and Position Analysis")

st.markdown('### Lighting')
lighting_bar = st.progress(0)
st.markdown('### Pose')
pose_bar = st.progress(0)
st.markdown('### Position')
position_bar = st.progress(0)

# Start OpenCV webcam feed
cap = cv2.VideoCapture(0) # 0 is default value for camera access in system

# Placeholder for live video
frame_window = st.image([])

# Function to analyze frame
def analyze_frame(frame):
    """ Process video frame for light, pose, and position estimation """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    pose_ok = False
    position_ok = False

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        if nose_y < hip_y:
            pose_ok = True

        nose_x = landmarks[mp_pose.PoseLandmark.NOSE].x
        if 0.4 < nose_x < 0.6:
            position_ok = True

    return brightness, pose_ok, position_ok

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image")
        break

    # Frame analysis
    brightness, pose_ok, position_ok = analyze_frame(frame)

    # Brightness Normalization
    brightness_score = int((brightness / 255) * 100)
    
    # Adjusting colors(green, red) for better understanding of lighting, pose and position
    lighting_color = "green" if brightness_score > 30 else "red"
    pose_color = "green" if pose_ok else "red"
    position_color = "green" if position_ok else "red"

    lighting_bar.markdown(f'<div style="background-color:{lighting_color};padding:5px;color:white;font-weight:bold;">Lighting</div>', unsafe_allow_html=True)
    pose_bar.markdown(f'<div style="background-color:{pose_color};padding:5px;color:white;font-weight:bold;">Pose</div>', unsafe_allow_html=True)
    position_bar.markdown(f'<div style="background-color:{position_color};padding:5px;color:white;font-weight:bold;">Position</div>', unsafe_allow_html=True)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame, channels="RGB")

cap.release()
st.write("Stream ended.")