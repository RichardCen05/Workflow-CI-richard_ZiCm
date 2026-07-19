from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "breast_cancer_preprocessing"
ARTIFACT_DIR = BASE_DIR / "training_artifacts"
TRACKING_DIR = BASE_DIR / "mlruns"


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    X_train = train_df.drop(columns=["diagnosis"])
    y_train = train_df["diagnosis"]
    X_test = test_df.drop(columns=["diagnosis"])
    y_test = test_df["diagnosis"]
    return X_train, X_test, y_train, y_test


def train_model():
    X_train, X_test, y_train, y_test = load_data()
    ARTIFACT_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file:{TRACKING_DIR}")
    mlflow.set_experiment("SMSML_richard_ZiCm_BreastCancer")

    with mlflow.start_run(run_name="manual_tuning"):
        # Manual logging dilakukan setelah evaluasi model.
        param_grid = {
            "n_estimators": [80, 120],
            "max_depth": [4, 8, None],
            "min_samples_split": [2, 5],
        }
        search = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            param_grid=param_grid,
            scoring="f1",
            cv=3,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        mlflow.log_params({"cv_best_" + key: value for key, value in search.best_params_.items()})

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        report_path = ARTIFACT_DIR / "classification_report.txt"
        report_path.write_text(classification_report(y_test, y_pred), encoding="utf-8")

        cm_path = ARTIFACT_DIR / "confusion_matrix.png"
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()

        fi_path = ARTIFACT_DIR / "feature_importance.csv"
        if hasattr(best_model, "feature_importances_"):
            pd.DataFrame({"feature": X_train.columns, "importance": best_model.feature_importances_}).sort_values(
                "importance", ascending=False
            ).to_csv(fi_path, index=False)
        else:
            pd.DataFrame({"feature": X_train.columns, "coefficient": best_model.coef_[0]}).to_csv(fi_path, index=False)

        mlflow.log_params(best_model.get_params())
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(best_model, artifact_path="model", input_example=X_test.head(5))
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(cm_path), artifact_path="plots")
        mlflow.log_artifact(str(fi_path), artifact_path="reports")
        print({
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "tracking_uri": mlflow.get_tracking_uri(),
        })


if __name__ == "__main__":
    train_model()
