"""
Laboratorio Final — Starbucks Rewards
PUNTO DE PARTIDA para su propia app de Streamlit. Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus
resultados del laboratorio — el Qini de su T-learner y la tabla de políticas —
con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

st.set_page_config(page_title="Laboratorio Final — Starbucks Rewards", page_icon="☕", layout="wide")

GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]


@st.cache_data
def load_data():
    # Obtener la carpeta donde se encuentra este archivo Python
    BASE_DIR = Path(__file__).resolve().parent

    # Construir la ruta del CSV
    ruta_csv = BASE_DIR / "data" / "uplift_campaign.csv"

    # Cargar los datos
    return pd.read_csv(ruta_csv)


@st.cache_data
def entrenar_t_learner(df: pd.DataFrame, test_size: float, seed: int):

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["treatment_group"]
    )

    # Separar clientes tratados y de control
    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]

    # Modelo Random Forest para tratados
    modelo_tratado = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    ).fit(
        train_t[FEATURES],
        train_t["responded_60d"]
    )

    # Modelo Random Forest para control
    modelo_control = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    ).fit(
        train_c[FEATURES],
        train_c["responded_60d"]
    )

    # Calcular uplift estimado
    test = test.copy()
    X_test = test[FEATURES]

    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]

    test["uplift_estimado"] = (
        test["p_tratado"] - test["p_control"]
    )

    return train, test

def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)


# --- Título y carga de datos ---
st.title("Laboratorio Final — Starbucks Rewards")
st.caption(
    "Análisis de la campaña Starbucks Rewards mediante Random Forest. "
    "Esta aplicación permite identificar el porcentaje de clientes "
    "que conviene contactar para maximizar el beneficio estimado "
    "de una campaña de cupones de $5."
)

df = load_data()

# TODO: agreguen un slider en la barra lateral para el % de la base a contactar
with st.sidebar:
    st.header("Parámetros")
    pct_contactar = st.slider(
    "Porcentaje de clientes a contactar (%)",
    min_value=10,
    max_value=100,
    value=20,
    step=10
)

train, test = entrenar_t_learner(df, test_size=0.30, seed=42)

# --- TODO: calculen y grafiquen la curva Qini de su T-learner ---
# ============================================
# CURVA QINI DEL MODELO T-LEARNER
# ============================================

def curva_qini(test_df, score_col, steps=40):

    orden = test_df.sort_values(
        score_col,
        ascending=False
    ).reset_index(drop=True)

    es_tratado = (
        orden["treatment_group"] == "treatment"
    ).values

    respondio = orden["responded_60d"].values

    acumulado_tratados = np.cumsum(
        np.where(es_tratado, respondio, 0)
    )

    acumulado_control = np.cumsum(
        np.where(~es_tratado, respondio, 0)
    )

    n_tratados = np.cumsum(es_tratado)
    n_control = np.cumsum(~es_tratado)

    n = len(orden)

    xs, ys = [0.0], [0.0]

    # Evaluar desde el 2.5% hasta el 100%
    puntos = np.linspace(
        max(1, int(n * 0.025)),
        n,
        steps
    ).astype(int)

    for i in puntos:

        nt = n_tratados[i - 1]
        nc = n_control[i - 1]

        if nt == 0 or nc == 0:
            continue

        tasa_t = acumulado_tratados[i - 1] / nt
        tasa_c = acumulado_control[i - 1] / nc

        xs.append(i / n)
        ys.append((tasa_t - tasa_c) * i)

    return np.array(xs), np.array(ys)

def coeficiente_qini(xs, ys, n):

    area_modelo = np.trapezoid(ys, xs)

    area_azar = np.trapezoid(
        np.linspace(0, ys[-1], len(xs)),
        xs
    )

    return (area_modelo - area_azar) / n


# Calcular curva Qini
xs, ys = curva_qini(
    test,
    "uplift_estimado"
)

qini = coeficiente_qini(
    xs,
    ys,
    len(test)
)

# ============================================
# MOSTRAR GRÁFICA
# ============================================

st.header("📈 Curva Qini - Random Forest")

fig, ax = plt.subplots(figsize=(10, 5))

# Curva del modelo
ax.plot(
    xs,
    ys,
    color=GREEN,
    linewidth=3,
    label=f"Random Forest (Qini = {qini:.3f})"
)

# Línea de referencia aleatoria
ax.plot(
    [0, 1],
    [0, ys[-1]],
    color=GRAY,
    linestyle="--",
    label="Selección aleatoria"
)

# Mostrar porcentaje seleccionado
ax.axvline(
    pct_contactar / 100,
    color=ORANGE,
    linestyle=":",
    label=f"Contacto seleccionado: {pct_contactar}%"
)

ax.set_xlabel("Porcentaje acumulado de clientes")

ax.set_ylabel("Respuestas incrementales acumuladas")

ax.set_title("Evaluación del T-learner con Random Forest")

ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig)

plt.close(fig)

# Mostrar indicador Qini
st.metric(
    "Coeficiente Qini",
    f"{qini:.3f}"
)

# --- TODO: muestren la tabla de políticas (valor por cliente y total) para el % elegido ---

# ============================================
# TABLA DE POLÍTICAS DE CONTACTO
# ============================================

st.divider()
st.header("📋 Evaluación de políticas de contacto")

resultados = []

for pct in range(10, 101, 10):

    n_clientes = int(len(test) * pct / 100)

    seleccionados = test.nlargest(
        n_clientes,
        "uplift_estimado"
    ).index

    valor_cliente, clientes = valor_de_la_politica(
        test,
        seleccionados
    )

    resultados.append({
        "Porcentaje": pct,
        "Clientes": clientes,
        "Valor por cliente": round(valor_cliente, 4),
        "Beneficio total": round(valor_cliente * clientes, 2)
    })

tabla_politicas = pd.DataFrame(resultados)

st.dataframe(
    tabla_politicas,
    use_container_width=True,
    hide_index=True
)

# ============================================
# RESULTADO DEL PORCENTAJE SELECCIONADO
# ============================================

seleccion = tabla_politicas[
    tabla_politicas["Porcentaje"] == pct_contactar
].iloc[0]

st.subheader("🎯 Estrategia seleccionada")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Clientes seleccionados",
    f"{seleccion['Clientes']:,.0f}"
)

col2.metric(
    "Beneficio por cliente",
    f"${seleccion['Valor por cliente']:,.4f}"
)

col3.metric(
    "Beneficio total estimado",
    f"${seleccion['Beneficio total']:,.2f}"
)

# ============================================
# RECOMENDACIÓN FINAL
# ============================================

mejor = tabla_politicas.loc[
    tabla_politicas["Beneficio total"].idxmax()
]

st.divider()
st.header("🏆 Recomendación de Marketing")

st.success(
    f"El porcentaje recomendado es {mejor['Porcentaje']:.0f}% "
    f"de los clientes, con un beneficio estimado de "
    f"${mejor['Beneficio total']:,.2f} en el conjunto de prueba."
)

st.info(
    f"Aplicando el porcentaje seleccionado a los 40,000 clientes, "
    f"se contactarían aproximadamente "
    f"{int(len(df) * pct_contactar / 100):,} clientes."
)

st.caption(
    "Los beneficios se estiman con datos experimentales "
    "y no representan ganancias garantizadas para toda la campaña."
)

st.info("Reemplacen cada TODO con su propio código. Revisen el notebook del laboratorio para la lógica exacta de cada pieza.")
