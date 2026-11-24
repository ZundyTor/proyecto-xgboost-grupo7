# Despliegue de Modelo con Gradient Boosting (XGBoost) - Grupo 7

## Descripción
Aplicación web interactiva desarrollada para la asignatura de **Inteligencia Artificial I (Actividad 3)**. El sistema permite evaluar la **potabilidad del agua** a partir de 9 parámetros fisicoquímicos medidos en laboratorio mediante un modelo de clasificación supervisada basado en **Gradient Boosting (XGBoost)**.

* **Problema a resolver:** Clasificación binaria para predecir si una muestra de agua es apta para el consumo humano (`Potability = 1`) o requiere tratamiento (`Potability = 0`).

## Demostración

[Enlace a la aplicación desplegada en Streamlit Cloud](https://proyecto-xgboost-grupo7-npmqvdyycj7ixwkmmybmvz.streamlit.app)

## Algoritmo utilizado
* **Algoritmo:** Extreme Gradient Boosting (`XGBClassifier`).
* **Justificación:** Los parámetros fisicoquímicos del agua presentan relaciones altamente no lineales y umbrales críticos combinados (como la interacción entre pH, sulfatos, cloraminas y sólidos disueltos). Además, el dataset real contiene valores faltantes (`NaN`), los cuales XGBoost gestiona de forma nativa mediante *sparsity-aware split finding*, evitando sesgos por imputación artificial. Para manejar el desbalanceo natural de las clases (~61% no potable vs. ~39% potable), se ajustó el hiperparámetro `scale_pos_weight`.
* **Métricas de desempeño obtenidas:** Evaluadas mediante validación cruzada estratificada (5-Folds) y conjunto de prueba (20%) con *Accuracy*, *Precision*, *Recall*, *F1-Score* y *ROC-AUC* (exportadas dinámicamente en `models/metrics.json` y visualizadas en la pestaña **Métricas del Modelo** de la aplicación).

## Dataset
* **Fuente de los datos:** Water Quality Dataset (Kaggle / espejo público `water_potability.csv`).
* **Tamaño del dataset:** 3,276 registros (supera el requisito mínimo de 500 registros).
* **Features utilizadas (9 variables fisicoquímicas):**
  1. `ph`: Nivel de acidez o alcalinidad del agua (0 - 14).
  2. `Hardness`: Dureza del agua por concentración de sales de calcio y magnesio (mg/L).
  3. `Solids`: Total de sólidos disueltos (ppm).
  4. `Chloramines`: Concentración de cloraminas desinfectantes (ppm).
  5. `Sulfate`: Cantidad de sulfatos disueltos (mg/L).
  6. `Conductivity`: Conductividad eléctrica del agua (μS/cm).
  7. `Organic_carbon`: Carbono orgánico total (ppm).
  8. `Trihalomethanes`: Concentración de trihalometanos (μg/L).
  9. `Turbidity`: Medida de partículas suspendidas que afectan la claridad (NTU).

## Estructura del Proyecto
```text
proyecto-xgboost-grupo7/
├── app/
│   └── app.py               # Interfaz web interactiva desarrollada en Streamlit
├── data/
│   └── water_potability.csv # Dataset original (descargado automáticamente al entrenar)
├── models/
│   ├── xgboost_water.json   # Modelo XGBoost entrenado
│   └── metrics.json         # Métricas de evaluación e importancia de variables
├── notebooks/               # Notebooks de exploración y análisis
├── src/
│   ├── __init__.py
│   ├── train.py             # Pipeline de carga, validación cruzada, entrenamiento y métricas
│   ├── predict.py           # Módulo de inferencia y carga del modelo serializado
│   └── preprocessing.py     # Utilidades de preparación de datos
├── requirements.txt         # Dependencias del entorno
└── README.md
