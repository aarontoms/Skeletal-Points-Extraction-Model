import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
import ast
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("combined labeled/all_labeled3.csv")

df['env_factors'] = df['env_factors'].apply(lambda x: x.split(';') if pd.notna(x) and x != '' else [])

mlb = MultiLabelBinarizer()
env_encoded = pd.DataFrame(mlb.fit_transform(df["env_factors"]),
                           columns=[f"env_{c}" for c in mlb.classes_])

y = pd.concat([df[["label"]], env_encoded], axis=1)

X = df.drop(columns=["label", "env_factors"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=df["label"], random_state=42
)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
multi_model = MultiOutputClassifier(rf)
multi_model.fit(X_train, y_train)

y_pred = multi_model.predict(X_test)

import json
from sklearn.metrics import classification_report

metrics = {}
for i, col in enumerate(y.columns):
    metrics[col] = classification_report(
        y_test.values[:, i],
        y_pred[:, i],
        output_dict=True,
        zero_division=0
    )

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)


for idx, col in enumerate(y.columns):
    print(f"=== Report for {col} ===")
    print(classification_report(y_test[col], y_pred[:, idx]))
    
for i, col in enumerate(y.columns):
    cm = confusion_matrix(y_test.values[:, i], y_pred[:, i])

    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {col}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"cm_{col}.png")
    plt.close()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Normal", "Predicted Trigger"],
    yticklabels=["Actual Normal", "Actual Trigger"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Autism Trigger Detection")
plt.savefig("confusion_matrix_autism_trigger_detection.png")

from sklearn.tree import plot_tree

estimator = multi_model.estimators_[0][0]

plt.figure(figsize=(20,10))
plot_tree(
    estimator,
    feature_names=X.columns,
    class_names=["Normal", "Trigger"],
    filled=True,
    max_depth=3
)
plt.savefig("sample_tree.png", dpi=300)
plt.close()

rf_label = multi_model.estimators_[0]
importances = rf_label.feature_importances_
fi = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:15]

plt.figure(figsize=(6,4))
fi.plot(kind="barh")
plt.title("Top Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

joblib.dump(multi_model, "multi_rf_model.pkl")
joblib.dump(mlb, "env_label_encoder.pkl")

print("Saved multi_rf_model.pkl and env_label_encoder.pkl")