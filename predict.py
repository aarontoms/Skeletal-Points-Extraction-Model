import pandas as pd
import joblib

clf = joblib.load("rf_model.pkl")

df = pd.read_csv("features test/features_landmarks_aut2 - Trim2.csv")

if "label" in df.columns:
    X = df.drop("label", axis=1)
    y = df["label"]
else:
    X = df
    y = None

preds = clf.predict(X)

print("Predictions per window:")
print(preds)

final = max(set(preds), key=list(preds).count)
print("\nFinal classification:", final)

if y is not None:
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nEvaluation:")
    print(classification_report(y, preds))
    print(confusion_matrix(y, preds))
