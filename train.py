import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import ast
import joblib

df = pd.read_csv("combined labeled/all_labeled2.csv")

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

for idx, col in enumerate(y.columns):
    print(f"=== Report for {col} ===")
    print(classification_report(y_test[col], y_pred[:, idx]))

joblib.dump(multi_model, "multi_rf_model.pkl")
joblib.dump(mlb, "env_label_encoder.pkl")
print("Saved multi_rf_model.pkl and env_label_encoder.pkl")