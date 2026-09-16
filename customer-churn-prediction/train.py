from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import FEATURES, TARGET, load_or_create_data

NUMERIC = ["tenure_months", "monthly_charge", "support_calls", "late_payments", "senior_citizen"]
CATEGORICAL = ["contract_type", "internet_service", "paperless_billing"]


def make_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def metrics(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred, zero_division=0), 4),
            "f1": round(f1_score(y, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y, prob), 4)}, pred


def main():
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    df = load_or_create_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET])
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=220, max_depth=12,
                                                  random_state=42, class_weight="balanced", n_jobs=-1)
    }
    scores, trained, preds = {}, {}, {}
    for name, estimator in candidates.items():
        model = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
        model.fit(X_train, y_train)
        scores[name], preds[name] = metrics(model, X_test, y_test)
        trained[name] = model
    best_name = max(scores, key=lambda name: scores[name]["roc_auc"])
    best = trained[best_name]
    joblib.dump(best, "models/churn_model.joblib")
    result = {"best_model": best_name, "test_rows": len(X_test), "models": scores}
    Path("outputs/metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    ConfusionMatrixDisplay(confusion_matrix(y_test, preds[best_name])).plot(cmap="Blues")
    plt.title(f"Confusion Matrix: {best_name}")
    plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=160); plt.close()

    if best_name == "Random Forest":
        names = best.named_steps["preprocessor"].get_feature_names_out()
        values = best.named_steps["model"].feature_importances_
        pd.Series(values, index=names).sort_values().tail(12).plot(kind="barh", figsize=(9, 5))
        plt.title("Top Churn Risk Features"); plt.xlabel("Importance")
        plt.tight_layout(); plt.savefig("outputs/feature_importance.png", dpi=160); plt.close()

    print(json.dumps(result, indent=2))
    print("\nTraining complete. Run: python -m streamlit run app.py")

if __name__ == "__main__":
    main()
