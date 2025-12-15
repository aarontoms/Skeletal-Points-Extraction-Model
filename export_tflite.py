# train_to_tflite.py
import json, ast, os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# ---- CONFIG ----
INPUT_CSV = "combined labeled/all_labeled2.csv"   # your concatenated CSV
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH = 32
EPOCHS = 100
PATIENCE = 8
WINDOW_COLS = {"start_frame", "end_frame"}  # drop if present
# ----------------

df = pd.read_csv(INPUT_CSV).dropna(subset=["label"])

# parse env_factors (semicolon-separated or list string)
def parse_env(x):
    if pd.isna(x) or x == "":
        return []
    # if already looks like list string: ["a","b"]
    if x.strip().startswith("["):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    # else semicolon separated
    return [s.strip() for s in x.split(";") if s.strip()]

df["env_factors"] = df.get("env_factors", "").apply(parse_env)

# MultiLabelBinarizer for env factors
mlb = MultiLabelBinarizer()
env_encoded = pd.DataFrame(mlb.fit_transform(df["env_factors"]),
                           columns=[f"env_{c}" for c in mlb.classes_])

# Prepare target matrix: first column = label (0/1), then env columns
y = pd.concat([df[["label"]].astype(int).reset_index(drop=True), env_encoded.reset_index(drop=True)], axis=1)

# Prepare feature matrix X: drop label, env_factors, optional window cols
drop_cols = {"label", "env_factors"} | WINDOW_COLS
X = df.drop(columns=[c for c in drop_cols if c in df.columns]).reset_index(drop=True)

# ensure only numeric columns remain (drop any stray non-numeric)
X = X.select_dtypes(include=[np.number])
feature_columns = list(X.columns)

# train/test split (stratify by label)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE
)

# scale features (save mean/scale to JSON for Flutter)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("artifacts", exist_ok=True)
with open("artifacts/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)
with open("artifacts/env_labels.json", "w") as f:
    json.dump(list(mlb.classes_), f)
with open("artifacts/scaler.json", "w") as f:
    json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)

# ---- Build Keras model ----
input_dim = X_train_scaled.shape[1]
output_dim = y_train.shape[1]  # 1 + number of env classes

model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(output_dim, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# callbacks
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

model.fit(
    X_train_scaled, y_train.values,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=[es],
    verbose=2
)

# evaluate
loss, acc = model.evaluate(X_test_scaled, y_test.values, verbose=0)
print(f"Test loss: {loss:.4f}, Test acc: {acc:.4f}")

from sklearn.metrics import classification_report
y_probs = model.predict(X_test_scaled)
y_pred = (y_probs>0.5).astype(int)
for i,col in enumerate(y.columns):
    print(f"=== {col} ===\n", classification_report(y_test.values[:,i], y_pred[:,i]))

# save SavedModel
model.export("saved_model")

# ---- Convert to TFLite with post-training quantization (dynamic range) ----
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# optional representative dataset for full integer quantization (commented unless you have memory)
def representative_gen():
    for i in range(min(100, X_train_scaled.shape[0])):
        yield [X_train_scaled[i:i+1].astype(np.float32)]
# Uncomment next line for better quantization if you want integer ops and provide repr data
# converter.representative_dataset = representative_gen

tflite_model = converter.convert()
open("artifacts/model.tflite", "wb").write(tflite_model)
print("Saved artifacts/model.tflite and saved_model/ and artifacts/*.json")
