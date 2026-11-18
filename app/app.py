import os
import sys
import joblib
import pandas as pd
import streamlit as st

# Permitir la importación de módulos desde la carpeta raíz (src/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import cargar_modelo
from src.preprocessing import preparar_datos_entrada

# Configuración de la página
st.set_page_config(
    page_title="Grupo 7 - Predictor XGBoost", layout="centered"
)

# Encabezado e información del equipo (Sin inventar problema ni dataset)
st.title("Predictor con Gradient Boosting (XGBoost)")
st.subheader("Inteligencia Artificial I - Actividad 3 (Entrega 1: Avances)")

with st.expander("Información del Proyecto - Grupo 7", expanded=True):
    st.markdown("""
        * **Algoritmo asignado:** Gradient Boosting (`XGBoost`)
        * **Integrantes:** Diego Alejandro Barreto Mejía, Sebastián Blanco Barón, Daniel Andrés Lerzundy Torres
        * **Dataset:** *Aún por definir*
        * **Estado actual:** Entorno base y despliegue en prueba (Entrega 1)
        """)

# Intentar cargar el modelo serializado (models/modelo.pkl)
model = cargar_modelo("models/modelo.pkl")

if model is None:
    st.warning(
        " **Estado del modelo:** `Por realizar` (`models/modelo.pkl` no encontrado). "
        "La interfaz se encuentra en modo de prueba funcional para validar la captura de datos y el despliegue."
    )
else:
    st.success(" Modelo `models/modelo.pkl` cargado correctamente.") 

st.markdown("---")
st.header("Parámetros de Entrada del Usuario")
st.caption(
    "Interfaz preliminar con los 3 inputs mínimos requeridos por la guía técnica. Se renombrarán según las features del dataset definitivo."
)

# Inputs del usuario (Mínimo 3 inputs requeridos por la guía en pág. 13 y plantilla en pág. 5)
feature1 = st.number_input("Feature 1 (Numérica)", value=0.0, step=0.1)
feature2 = st.selectbox("Feature 2 (Categórica)", ["A", "B", "C"])
feature3 = st.number_input("Feature 3 (Numérica)", value=0.0, step=1.0)

# Botón de predicción y manejo de errores básicos
if st.button("Predecir"):
    try:
        # Preparar input siguiendo la guía (pág. 5)
        input_data = preparar_datos_entrada({
            "feature1": feature1,
            "feature2": feature2,
            "feature3": feature3,
        })

        st.write("**Datos capturados por la aplicación:**")
        st.dataframe(input_data, use_container_width=True)

        # Ejecutar predicción si el modelo ya fue entrenado y serializado
        if model is not None:
            prediccion = model.predict(input_data)
            st.success(f"Predicción: {prediccion[0]}")
        else:
            st.info(
                "Prueba de interfaz exitosa: El DataFrame de entrada se generó correctamente. "
                "La inferencia real se activará automáticamente al subir el archivo `models/modelo.pkl`."
            )
    except Exception as e:
        st.error(f"Error al procesar los datos de entrada: {e}")