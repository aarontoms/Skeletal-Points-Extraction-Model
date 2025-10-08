import cv2
import mediapipe as mp

video_path = "input/WIN_20250915_02_21_24_Pro.mp4"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(video_path)

landmark_list = []
ret, frame = cap.read()
if not ret:
    raise ValueError("Cannot read video")

rotate_flag = None
print(frame.shape)
if frame.shape[1] > frame.shape[0]:
    print("not rotate")
    rotate_flag = None
else:
    print("rotate")
    rotate_flag = cv2.ROTATE_90_CLOCKWISE