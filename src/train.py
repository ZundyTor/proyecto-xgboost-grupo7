import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "water_potability.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "xgboost_water.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

DATASET_URL = (
    "https://raw.githubusercontent.com/amankharwal/Website-data/master/water_potability.csv"
)

FEATURE_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]
TARGET_COLUMN = "Potability"


def load_data() -> pd.DataFrame:
    """Carga el dataset local o lo descarga si no existe en data/."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print(f"Descargando dataset desde {DATASET_URL}...")
        df = pd.read_csv(DATASET_URL)
        df.to_csv(DATA_PATH, index=False)
        print(f"Dataset guardado en {DATA_PATH}")
    else:
        print(f"Cargando dataset local desde {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    return df


def train_and_evaluate():
    df = load_data()

    # Eliminar filas completamente vacías si existieran
    df = df.dropna(how="all").copy()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    # División estratificada 80% entrenamiento / 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Ajuste por desbalanceo de clases (~61% No Potable / ~39% Potable)
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count

    # XGBoost maneja valores nulos (NaN) de forma nativa mediante sparsity-aware split finding
    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.5,
        reg_lambda=1.5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # Validación cruzada estratificada (5 folds)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_roc_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    # Entrenamiento final
    model.fit(X_train, y_train)

    # Evaluación en conjunto de prueba
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset_rows": int(len(df)),
        "dataset_features": len(FEATURE_COLUMNS),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "class_distribution": {
            "no_potable_0": int((y == 0).sum()),
            "potable_1": int((y == 1).sum()),
        },
        "cv_roc_auc_mean": round(float(np.mean(cv_roc_auc)), 4),
        "cv_roc_auc_std": round(float(np.std(cv_roc_auc)), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "test_precision": round(float(precision_score(y_test, y_pred)), 4),
        "test_recall": round(float(recall_score(y_test, y_pred)), 4),
        "test_f1": round(float(f1_score(y_test, y_pred)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importances": {
            feat: round(float(imp), 4)
            for feat, imp in sorted(
                zip(FEATURE_COLUMNS, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        },
        "feature_stats": {
            col: {
                "min": round(float(df[col].min()), 2),
                "max": round(float(df[col].max()), 2),
                "median": round(float(df[col].median()), 2),
            }
            for col in FEATURE_COLUMNS
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n--- Resultados del Entrenamiento XGBoost ---")
    print(f"ROC-AUC (CV 5-Fold): {metrics['cv_roc_auc_mean']} ± {metrics['cv_roc_auc_std']}")
    print(f"Accuracy (Test):     {metrics['test_accuracy']}")
    print(f"Precision (Test):    {metrics['test_precision']}")
    print(f"Recall (Test):       {metrics['test_recall']}")
    print(f"F1-Score (Test):     {metrics['test_f1']}")
    print(f"ROC-AUC (Test):      {metrics['test_roc_auc']}")
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
