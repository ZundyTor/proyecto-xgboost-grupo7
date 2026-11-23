import json
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Conectar con la carpeta src/
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from predict import cargar_modelo, predecir_muestra

MODEL_PATH = BASE_DIR / "models" / "xgboost_water.json"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
DATA_PATH = BASE_DIR / "data" / "water_potability.csv"

st.set_page_config(
    page_title="Predicción de Potabilidad del Agua | XGBoost",
    page_icon="💧",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    """Carga el modelo y las métricas; si no existen aún, ejecuta el entrenamiento."""
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        from train import train_and_evaluate

        train_and_evaluate()

    model = cargar_modelo(MODEL_PATH)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None
    return model, metrics, df


model, metrics, df = load_artifacts()

# --- ENCABEZADO ---
st.title("💧 Sistema Inteligente de Predicción de Potabilidad del Agua")
st.markdown(
    "**Inteligencia Artificial I — Actividad 3 (Grupo 7)** | Modelo basado en **Gradient Boosting (XGBoost)**"
)
st.divider()

# --- PESTAÑAS PRINCIPALES ---
tab_sim, tab_metrics, tab_data = st.tabs(
    ["🧪 Simulador de Muestra", "📊 Métricas del Modelo", "🗂️ Exploración del Dataset"]
)

# ==========================================
# TAB 1: SIMULADOR INTERACTIVO
# ==========================================
with tab_sim:
    st.subheader("Parámetros Fisicoquímicos de la Muestra")
    st.caption(
        "Ajusta los valores medidos en laboratorio o selecciona un caso de prueba predefinido."
    )

    preset_col1, preset_col2, preset_col3 = st.columns([1, 1, 2])
    if "preset" not in st.session_state:
        st.session_state.preset = "mediana"

    with preset_col1:
        if st.button("🟢 Cargar Ejemplo Potable", use_container_width=True):
            st.session_state.preset = "potable"
    with preset_col2:
        if st.button("🔴 Cargar Ejemplo No Potable", use_container_width=True):
            st.session_state.preset = "no_potable"
    with preset_col3:
        if st.button("🔄 Restablecer Valores Mediana", use_container_width=True):
            st.session_state.preset = "mediana"

    presets = {
        "mediana": {
            "ph": 7.04,
            "Hardness": 196.97,
            "Solids": 20927.83,
            "Chloramines": 7.13,
            "Sulfate": 333.07,
            "Conductivity": 421.88,
            "Organic_carbon": 14.22,
            "Trihalomethanes": 66.62,
            "Turbidity": 3.96,
        },
        "potable": {
            "ph": 7.25,
            "Hardness": 185.0,
            "Solids": 16500.0,
            "Chloramines": 7.40,
            "Sulfate": 310.0,
            "Conductivity": 390.0,
            "Organic_carbon": 11.5,
            "Trihalomethanes": 60.0,
            "Turbidity": 3.20,
        },
        "no_potable": {
            "ph": 4.80,
            "Hardness": 275.0,
            "Solids": 41000.0,
            "Chloramines": 10.80,
            "Sulfate": 440.0,
            "Conductivity": 610.0,
            "Organic_carbon": 22.0,
            "Trihalomethanes": 105.0,
            "Turbidity": 5.80,
        },
    }
    defaults = presets[st.session_state.preset]

    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        ph = st.slider("pH (0 - 14)", 0.0, 14.0, float(defaults["ph"]), 0.05)
        hardness = st.slider(
            "Dureza / Hardness (mg/L)", 45.0, 325.0, float(defaults["Hardness"]), 1.0
        )
        solids = st.slider(
            "Sólidos Disueltos / Solids (ppm)",
            300.0,
            61500.0,
            float(defaults["Solids"]),
            100.0,
        )

    with col_in2:
        chloramines = st.slider(
            "Cloraminas (ppm)", 0.3, 13.5, float(defaults["Chloramines"]), 0.1
        )
        sulfate = st.slider(
            "Sulfatos (mg/L)", 125.0, 485.0, float(defaults["Sulfate"]), 1.0
        )
        conductivity = st.slider(
            "Conductividad (μS/cm)", 180.0, 760.0, float(defaults["Conductivity"]), 1.0
        )

    with col_in3:
        organic_carbon = st.slider(
            "Carbono Orgánico (ppm)", 2.0, 29.0, float(defaults["Organic_carbon"]), 0.2
        )
        trihalomethanes = st.slider(
            "Trihalometanos (μg/L)", 0.5, 125.0, float(defaults["Trihalomethanes"]), 0.5
        )
        turbidity = st.slider(
            "Turbidez (NTU)", 1.0, 7.0, float(defaults["Turbidity"]), 0.05
        )

    # Predicción usando src/predict.py
    datos_muestra = {
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity,
    }

    resultado = predecir_muestra(datos_muestra, model=model)
    prob_potable = resultado["probabilidad_potable"]
    prediction = resultado["clase"]

    st.divider()
    res_col1, res_col2 = st.columns([1, 1.3])

    with res_col1:
        st.subheader("Resultado del Diagnóstico")
        if prediction == 1:
            st.success("### ✅ AGUA APTA PARA CONSUMO (Potable)")
            st.write(
                f"El modelo estima una probabilidad de potabilidad del **{prob_potable * 100:.1f}%**."
            )
        else:
            st.error("### ⚠️ AGUA NO POTABLE (Requiere Tratamiento)")
            st.write(
                f"El modelo estima una probabilidad de potabilidad de solo **{prob_potable * 100:.1f}%**."
            )

        st.info(
            "**Nota técnica:** XGBoost evalúa interacciones no lineales entre acidez (pH), "
            "sólidos disueltos y compuestos químicos (sulfatos y cloraminas) para determinar "
            "si la muestra cumple umbrales seguros."
        )

    with res_col2:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob_potable * 100,
                number={"suffix": "%"},
                title={"text": "Índice de Probabilidad de Potabilidad"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0, 40], "color": "#ffcccc"},
                        {"range": [40, 60], "color": "#fff2cc"},
                        {"range": [60, 100], "color": "#ccf2d6"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ==========================================
# TAB 2: MÉTRICAS E IMPORTANCIA DE FEATURES
# ==========================================
with tab_metrics:
    st.subheader("Evaluación del Modelo en Conjunto de Prueba (20%)")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['test_accuracy'] * 100:.2f}%")
    m2.metric("Precision", f"{metrics['test_precision'] * 100:.2f}%")
    m3.metric("Recall", f"{metrics['test_recall'] * 100:.2f}%")
    m4.metric("F1-Score", f"{metrics['test_f1'] * 100:.2f}%")
    m5.metric("ROC-AUC", f"{metrics['test_roc_auc']:.4f}")

    st.caption(
        f"Validación Cruzada Estratificada (5-Folds) ROC-AUC: **{metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}**"
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Importancia de Variables (Feature Importance)")
        imp_df = pd.DataFrame(
            {
                "Variable": list(metrics["feature_importances"].keys()),
                "Importancia": list(metrics["feature_importances"].values()),
            }
        ).sort_values("Importancia", ascending=True)

        fig_imp = px.bar(
            imp_df,
            x="Importancia",
            y="Variable",
            orientation="h",
            color="Importancia",
            color_continuous_scale="Blues",
        )
        fig_imp.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    with chart_col2:
        st.markdown("#### Matriz de Confusión (Test Set)")
        cm = metrics["confusion_matrix"]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicción", y="Valor Real", color="Muestras"),
            x=["No Potable (0)", "Potable (1)"],
            y=["No Potable (0)", "Potable (1)"],
            color_continuous_scale="Teal",
        )
        fig_cm.update_layout(height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

# ==========================================
# TAB 3: EXPLORACIÓN DEL DATASET
# ==========================================
with tab_data:
    st.subheader("Dataset: Water Quality & Potability")
    if df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total de Registros", f"{len(df):,}")
        c2.metric("Variables Predictoras", "9 fisicoquímicas")
        c3.metric("Porcentaje Potable", f"{df['Potability'].mean() * 100:.1f}%")

        st.markdown("#### Muestra de Registros Reales")
        st.dataframe(df.head(15), use_container_width=True)

        st.markdown("#### Estadísticas Descriptivas y Valores Faltantes")
        summary_df = df.describe().T[["count", "mean", "std", "min", "50%", "max"]]
        summary_df["nulos"] = df.isnull().sum()
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.warning("No se encontró el archivo CSV local en `data/water_potability.csv`.")
