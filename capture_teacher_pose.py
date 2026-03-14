import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

print("Press 's' to save teacher pose")

while True:

    ret, frame = cap.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(image)

    if results.pose_landmarks:

        landmarks = []

        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Teacher Pose Capture", frame)

    key = cv2.waitKey(1)

    if key == ord("s"):
        np.save("teacher_pose.npy", landmarks)
        print("Teacher pose saved")
        break

cap.release()
cv2.destroyAllWindows()
