import cv2
import pandas as pd
import ast

video_path = "input/test.mp4"
pred_csv = "prediction/predict.csv"

pred_df = pd.read_csv(pred_csv)

pred_df["env_factors_pred"] = pred_df["env_factors_pred"].apply(lambda x: ast.literal_eval(x))

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_overlay.mp4", fourcc, fps, 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_idx = 0
current_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if current_idx < len(pred_df) and frame_idx > pred_df.iloc[current_idx]["end_frame"]:
        current_idx += 1

    if current_idx < len(pred_df):
        row = pred_df.iloc[current_idx]
        if row["start_frame"] <= frame_idx <= row["end_frame"]:
            label = row["label_pred"]
            env = ", ".join(row["env_factors_pred"]) if row["env_factors_pred"] else "None"

            cv2.putText(frame, f"Prediction: {label}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == "Normal" else (0, 0, 255), 2)

            cv2.putText(frame, f"Env: {env}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Overlay video saved as output_overlay.mp4")
