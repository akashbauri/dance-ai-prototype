import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os

from scoring import pose_score

st.set_page_config(page_title="AI Dance Coach", layout="wide")

st.title("🕺 AI Dance Pose Coach")
st.markdown("Real-time dance pose feedback using MediaPipe")

# Sidebar
st.sidebar.title("Controls")
run = st.sidebar.checkbox("Start Camera")

st.sidebar.markdown("""
### Features
✔ Real-time pose tracking  
✔ Teacher vs Student comparison  
✔ Pose similarity scoring  
✔ Live skeleton visualization
""")

# Load teacher pose safely
teacher_pose = None
if os.path.exists("teacher_pose.npy"):
    teacher_pose = np.load("teacher_pose.npy")
else:
    st.warning("teacher_pose.npy not found. Score will not be calculated.")

# Layout
col1, col2 = st.columns(2)

col1.subheader("Teacher Reference")
col1.info("Reference pose used for scoring")

col2.subheader("Student Webcam")

FRAME_WINDOW = col2.image([])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while run:

    ret, frame = cap.read()

    if not ret:
        st.error("Webcam not detected")
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(image)

    if results.pose_landmarks:

        landmarks = []

        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        if teacher_pose is not None:

            score = pose_score(landmarks, teacher_pose)

            cv2.putText(
                frame,
                f"Score: {int(score)}",
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            st.metric("Dance Accuracy Score", f"{int(score)}%")

    FRAME_WINDOW.image(frame)

cap.release()
