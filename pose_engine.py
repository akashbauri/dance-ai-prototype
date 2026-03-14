import mediapipe as mp

mp_pose = mp.solutions.pose

pose = mp_pose.Pose()

def extract_landmarks(frame):

    results = pose.process(frame)

    if results.pose_landmarks:

        landmarks = []

        for lm in results.pose_landmarks.landmark:

            landmarks.append([lm.x, lm.y, lm.z])

        return landmarks

    return None
