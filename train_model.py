
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# 1. Load dataset
df = pd.read_csv("data/transfusion.data", skipinitialspace=True)

df.columns = [
    "Recency",
    "Frequency",
    "Monetary",
    "Time",
    "Donated_Blood",
]

X = df[["Recency", "Frequency", "Monetary", "Time"]]
y = df["Donated_Blood"]

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(y.value_counts())

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 3. Define three ML algorithms
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    ),
}

# 4. Train, evaluate, and save models
os.makedirs("models", exist_ok=True)
results = []

for name, model in models.items():
    print(f"\n{'=' * 50}")
    print("Training:", name)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, os.path.join("models", filename))

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    })

# 5. Compare all algorithms
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("F1-score", ascending=False)

print("\n\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv("models/model_comparison.csv", index=False)

print("\nModels saved in the models folder.")
