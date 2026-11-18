import pandas as pd


def preparar_datos_entrada(input_dict: dict) -> pd.DataFrame:
    """Convierte el diccionario de entradas del usuario en un DataFrame
    compatible con el modelo XGBoost.
    (Se ajustarán las transformaciones cuando se defina el dataset).
    """
    df_input = pd.DataFrame([input_dict])
    return df_input