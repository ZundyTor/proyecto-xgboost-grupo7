import os
import joblib

MODEL_PATH = os.path.join("models", "modelo.pkl")


def cargar_modelo(ruta: str = MODEL_PATH):
    """Carga el modelo serializado si existe en el directorio models/.
    Retorna None si el modelo aún está pendiente por entrenar.
    """
    if os.path.exists(ruta):
        return joblib.load(ruta)
    return None