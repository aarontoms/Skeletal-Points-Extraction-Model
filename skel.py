import cv2
import mediapipe as mp
import pandas as pd
import os, glob

os.makedirs("output csv", exist_ok=True)
video_files = glob.glob("input/test*.mp4")
video_names = [os.path.splitext(os.path.basename(f))[0] for f in video_files]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

keep_ids = [
    0,   # Nose
    7, 8,   # Left Ear, Right Ear
    11, 12, # Left Shoulder, Right Shoulder
    13, 14, # Left Elbow, Right Elbow
    15, 16, # Left Wrist, Right Wrist
    17, 18, # Left Pinky, Right Pinky
    19, 20, # Left Index, Right Index
    21, 22, # Left Thumb, Right Thumb
]

for video_path, video_name in zip(video_files, video_names):
    print(f"Processing {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    landmark_list = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            row = [frame_idx]
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                if idx in keep_ids:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            landmark_list.append(row)

        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    
    if not landmark_list:
        print(f"No pose detected in {video_name}, skipping saving.")
        continue

    columns = ["frame"]
    for idx in keep_ids:
        columns.extend([f"landmark_{idx}_x", f"landmark_{idx}_y", f"landmark_{idx}_z", f"landmark_{idx}_vis"])

    df = pd.DataFrame(landmark_list, columns=columns)
    output_csv = f"output csv/landmarks_{video_name}.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

cv2.destroyAllWindows()