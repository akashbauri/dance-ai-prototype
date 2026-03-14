import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

from pose_engine import extract_landmarks
from scoring import pose_score, joint_error

st.set_page_config(page_title="AI Dance Coach", layout="wide")

st.title("AI Dance Coach")

st.subheader("Real-Time Dance Pose Feedback")

col1, col2 = st.columns(2)

teacher_pose = np.load("teacher_pose.npy")

run = st.checkbox("Start Camera")

cap = cv2.VideoCapture(0)

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

while run:

    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame)

    if results.pose_landmarks:

        landmarks = []

        for lm in results.pose_landmarks.landmark:

            landmarks.extend([lm.x, lm.y, lm.z])

        score = pose_score(landmarks, teacher_pose)

        errors = joint_error(landmarks, teacher_pose)

        cv2.putText(frame,
                    f"Score: {score}",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)

    col2.image(frame)

cap.release()
