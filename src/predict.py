from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_water.json"


def cargar_modelo(ruta: Path = MODEL_PATH) -> XGBClassifier | None:
    """Carga el modelo XGBoost serializado en formato JSON si existe."""
    if ruta.exists():
        model = XGBClassifier()
        model.load_model(ruta)
        return model
    return None


def predecir_muestra(datos: dict, model: XGBClassifier | None = None) -> dict:
    """Realiza la predicción de potabilidad para un diccionario de parámetros."""
    if model is None:
        model = cargar_modelo()
    if model is None:
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en {MODEL_PATH}. Ejecuta src/train.py primero."
        )

    df_input = pd.DataFrame([datos])
    prob_potable = float(model.predict_proba(df_input)[0][1])
    prediccion = int(prob_potable >= 0.5)

    return {
        "potable": bool(prediccion),
        "clase": prediccion,
        "probabilidad_potable": round(prob_potable, 4),
    }
