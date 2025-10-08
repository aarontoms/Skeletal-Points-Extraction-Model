import os, glob
import cv2
import pandas as pd

input_video = "vid3.mp4"
video_name, _ = os.path.splitext(input_video)
features_csv = f"features/features_landmarks_{video_name}.csv"

df = pd.read_csv(features_csv)
df["label"] = None

cap = cv2.VideoCapture("input/" + input_video)

for i, row in df.iterrows():
    start, end = int(row["start_frame"]), int(row["end_frame"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    while True:
        ret, frame = cap.read()
        if not ret or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > end:
            break
        frame = cv2.resize(frame, (800, 450))
        cv2.imshow("Window", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"Window {i} ({start}-{end})")

    key = input("Label this clip (t=trigger, n=normal, s=skip): ").strip().lower()
    if key == "t":
        df.at[i, "label"] = "trigger"
    elif key == "n":
        df.at[i, "label"] = "normal"
    elif key == "s":
        df.at[i, "label"] = None

cap.release()
cv2.destroyAllWindows()

os.makedirs("labeled", exist_ok=True)
df = df.dropna(subset=["label"])
df.drop(columns=["start_frame", "end_frame"], inplace=True, errors="ignore")
video_name = os.path.splitext(input_video)[0]
df.to_csv(os.path.join("labeled", f"labeled_{video_name}.csv"), index=False)

print(f"Saved labeled features to labeled/labeled_{video_name}.csv")