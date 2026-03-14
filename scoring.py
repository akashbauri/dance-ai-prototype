import numpy as np

def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def pose_score(student_pose, teacher_pose):

    sim = cosine_similarity(student_pose, teacher_pose)

    return round(sim * 100, 2)


def joint_error(student, teacher):

    diff = np.abs(np.array(student) - np.array(teacher))

    return diff
