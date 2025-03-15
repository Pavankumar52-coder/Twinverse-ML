# Importing necessary libraries
from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp

# Initializing the flask app
app = Flask(__name__)

# Utilizing mediapipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function for image anaysis in camera
def analyze_image(frame):
    """ Process frame to extract lighting, pose, and position """

    # Convert frame to grayscale using cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Compute brightness level (mean intensity)
    brightness = np.mean(gray)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Pose estimation
    results = pose.process(rgb_frame)
    pose_ok, position_ok = False, False

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        if nose_y < hip_y:
            pose_ok = True

        # Checking for person if centered
        nose_x = landmarks[mp_pose.PoseLandmark.NOSE].x
        if 0.4 < nose_x < 0.6:
            position_ok = True

    return {
        "brightness": brightness,
        "pose_ok": pose_ok,
        "position_ok": position_ok
    }

@app.route("/image_processn", methods=["POST"])
def image_process():
    """ Endpoint to process image and return analysis """
    file = request.files["image"]
    if not file:
        return jsonify({"error": "No image received"}), 400
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Image analysis
    result = analyze_image(frame)
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)