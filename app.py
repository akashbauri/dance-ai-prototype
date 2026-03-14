import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

from pose_utils import calculate_angle, similarity_score

st.title("AI Dance Pose Prototype")

run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

teacher_angle = 120

while run:

    ret, frame = cap.read()

    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(image)

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        shoulder = [landmarks[11].x, landmarks[11].y]
        elbow = [landmarks[13].x, landmarks[13].y]
        wrist = [landmarks[15].x, landmarks[15].y]

        angle = calculate_angle(shoulder, elbow, wrist)

        score = similarity_score(angle, teacher_angle)

        cv2.putText(frame, f"Elbow Angle: {int(angle)}",
        (50,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2)

        cv2.putText(frame, f"Score: {int(score)}",
        (50,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2)

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    FRAME_WINDOW.image(frame)

cap.release()
