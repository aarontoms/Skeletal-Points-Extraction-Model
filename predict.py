import pandas as pd
import joblib

model = joblib.load("multi_rf_model.pkl")
mlb = joblib.load("env_label_encoder.pkl")

input_csv = "features/features_landmarks_test.csv"
df = pd.read_csv(input_csv)

frame_cols = df[["start_frame", "end_frame"]].copy()

X = df.drop(columns=["start_frame", "end_frame"])

y_pred = model.predict(X)

label_pred = y_pred[:, 0]
env_pred = y_pred[:, 1:]

env_labels = mlb.inverse_transform(env_pred)

result = pd.DataFrame({
    "start_frame": frame_cols["start_frame"],
    "end_frame": frame_cols["end_frame"],
    "label_pred": ["Autistic Trigger" if x == 1 else "Normal" for x in label_pred],
    "env_factors_pred": [str(list(e)) for e in env_labels]
})

result.to_csv("prediction/predict.csv", index=False)
print("✅ Saved predictions to prediction/predict.csv")