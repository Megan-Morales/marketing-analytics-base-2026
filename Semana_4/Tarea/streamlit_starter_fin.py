"""Laboratorio de recomendación de cafetería — app basada en streamlit_starter.py.

Publicación: seleccionar Semana_4/app.py en Streamlit Community Cloud.
Los datos se leen de Semana_4/data, junto a este archivo.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Café | Recomendaciones para ti", page_icon="☕", layout="wide")
BASE = Path(__file__).resolve().parent
VERDE = "#00704A"
NARANJA = "#C4512F"

st.markdown(
    """<style>
    .stApp {background-color: #faf8f4;}
    h1, h2, h3 {color: #154a38;}
    [data-testid="stMetric"] {background: white; border: 1px solid #e8e5df;
        padding: 14px; border-radius: 12px;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparando los recomendadores...")
def cargar_modelos():
    """Matriz cliente-producto, similitudes y modelos: solo entrenamiento con historial."""
    archivos = ["catalogo.csv", "clientes.csv", "ordenes_historial.csv",
                "ordenes_holdout.csv", "experimento.csv"]
    faltantes = [n for n in archivos if not (BASE / "data" / n).is_file()]
    if faltantes:
        raise FileNotFoundError(
            "Faltan archivos en Semana_4/data: " + ", ".join(faltantes)
        )

    catalogo = pd.read_csv(BASE / "data" / "catalogo.csv")
    clientes = pd.read_csv(BASE / "data" / "clientes.csv")
    historial = pd.read_csv(BASE / "data" / "ordenes_historial.csv")
    holdout = pd.read_csv(BASE / "data" / "ordenes_holdout.csv")
    experimento = pd.read_csv(BASE / "data" / "experimento.csv")

    productos = catalogo["item"].tolist()
    ids = clientes["customer_id"].to_numpy()
    matriz = (pd.crosstab(historial["customer_id"], historial["item"]) > 0).astype(int)
    matriz = matriz.reindex(index=ids, columns=productos, fill_value=0)
    X = matriz.to_numpy(dtype=np.float32)

    H = (pd.crosstab(holdout["customer_id"], holdout["item"]) > 0).astype(int)
    H = H.reindex(index=ids, columns=productos, fill_value=0).to_numpy()
    NUEVO = H * (1 - X)  # nuevas compras durante el holdout
    col_agua = productos.index("Agua")
    NUEVO[:, col_agua] = 0

    es_control = (experimento.set_index("customer_id")
                  .reindex(ids)["grupo"].eq("control").fillna(False).to_numpy())
    elegibles = es_control & (NUEVO.sum(axis=1) > 0)

    # TODO 1 del starter: similitud entre productos.
    S_it = cosine_similarity(X.T)
    np.fill_diagonal(S_it, 0)

    # TODO 2: similitudes por ítem; popularidad como línea base.
    popularidad = X.sum(axis=0).astype(float)
    item_scores = X @ S_it

    # TODO 3: vecinos similares por cliente. Una sola vez, hasta K=30.
    similitudes_clientes = cosine_similarity(X)
    np.fill_diagonal(similitudes_clientes, 0)
    vecinos = np.argsort(-similitudes_clientes, axis=1)[:, :30]
    pesos_vecinos = np.take_along_axis(similitudes_clientes, vecinos, axis=1)
    del similitudes_clientes

    # Modelo adicional: factores latentes.
    n_factores = min(8, min(X.shape) - 1)
    svd = TruncatedSVD(n_components=n_factores, random_state=42)
    svd_scores = svd.fit_transform(X) @ svd.components_

    return (catalogo, productos, ids, X, NUEVO, elegibles, col_agua,
            S_it, popularidad, item_scores, vecinos, pesos_vecinos, svd_scores)


try:
    (catalogo, productos, ids, X, NUEVO, elegibles, COL_AGUA,
     S_it, popularidad, item_scores, vecinos, pesos_vecinos,
     svd_scores) = cargar_modelos()
except (FileNotFoundError, KeyError, ValueError) as exc:
    st.error(f"No se pudieron cargar los datos: {exc}")
    st.info("Verifica que los CSV estén en la carpeta data/ junto a app.py y tengan las columnas de la plantilla.")
    st.stop()

METODOS = ["Popularidad", "Filtrado por ítem", "Filtrado por cliente",
           "SVD (8 factores)", "Final (ítem + respaldo)"]


@st.cache_data(show_spinner=False)
def puntajes_clientes(k: int) -> np.ndarray:
    """El slider K cambia el cálculo del modelo, no solo su explicación."""
    vec = vecinos[:, :k]
    pesos = pesos_vecinos[:, :k]
    compras_vecinos = X[vec]  # [clientes, K, productos]
    return np.einsum("nk,nkp->np", pesos, compras_vecinos, optimize=True)


def puntajes(metodo: str, k: int) -> np.ndarray:
    # Cuando no hay historial, los modelos colaborativos usan popularidad.
    if metodo == "Popularidad":
        return np.broadcast_to(popularidad, X.shape)
    if metodo == "Filtrado por ítem":
        resultado = item_scores
    elif metodo == "Filtrado por cliente":
        resultado = puntajes_clientes(k)
    elif metodo == "SVD (8 factores)":
        resultado = svd_scores
    else:
        resultado = item_scores
    return np.where((X.sum(axis=1) > 0)[:, None], resultado, popularidad)


def top3_para_todos(scores: np.ndarray) -> np.ndarray:
    disponibles = np.where(X > 0, -np.inf, scores).astype(float)
    disponibles[:, COL_AGUA] = -np.inf
    return np.argsort(-disponibles, axis=1)[:, :3]


def por_que(metodo: str, fila: int, columna: int, k: int) -> str:
    if not X[fila].any():
        return "Como todavía no tiene historial, recomendamos un producto popular."
    if metodo == "Popularidad":
        return "Es uno de los productos que más compran los clientes."
    if metodo == "Filtrado por cliente":
        votos = int((X[vecinos[fila, :k], columna] > 0).sum())
        return f"{votos} de sus {k} clientes similares también lo compraron."
    if metodo == "SVD (8 factores)":
        return "El modelo detectó patrones de compra que se relacionan con sus gustos."
    compras = np.flatnonzero(X[fila])
    parecido = compras[np.argmax(S_it[compras, columna])]
    return (f"Porque ya compró {productos[parecido]} "
            f"(similitud de {S_it[parecido, columna]:.2f}).")


@st.cache_data(show_spinner="Calculando indicadores sobre el grupo de control...")
def comparar(k: int) -> pd.DataFrame:
    """Evaluación exclusivamente en clientes control con compras nuevas."""
    filas = []
    indices = np.flatnonzero(elegibles)
    for metodo in METODOS:
        top = top3_para_todos(puntajes(metodo, k))
        aciertos = NUEVO[indices[:, None], top[indices]]
        filas.append({
            "Recomendador": metodo,
            "Hit rate@3 (control)": float(aciertos.any(axis=1).mean()) if len(indices) else np.nan,
            "Precisión@3": float(aciertos.sum(axis=1).mean() / 3) if len(indices) else np.nan,
            "Cobertura (productos)": int(len(np.unique(top[indices]))) if len(indices) else 0,
        })
    return pd.DataFrame(filas)


st.title("☕ Café | Recomendaciones para ti")
st.write("Explore qué **3 productos nuevos** recomendar a un cliente y compare los modelos "
         "usando compras de los 90 días del **grupo de control**.")

# TODO 4 del starter: controles de negocio.
with st.sidebar:
    st.header("Configurar recomendaciones")
    metodo = st.selectbox("Método", METODOS, index=4)
    cid = st.selectbox("Cliente", ids.tolist(), format_func=lambda c: f"Cliente {c}")
    K = st.slider("Número de vecinos (K)", min_value=5, max_value=30, value=30, step=5,
                  help="Cambia las recomendaciones y los indicadores del filtrado por cliente.")
    st.caption("Datos simulados para fines académicos.")

fila = int(np.where(ids == cid)[0][0])
matriz_puntajes = puntajes(metodo, K)
recomendaciones = top3_para_todos(matriz_puntajes)[fila]

col1, col2, col3 = st.columns(3)
col1.metric("Productos ya comprados", int(X[fila].sum()))
col2.metric("Recomendaciones", len(recomendaciones))
col3.metric("Clientes en la base", f"{len(ids):,}")

tab1, tab2, tab3 = st.tabs(["🎯 Recomendaciones", "📊 Comparación", "ℹ️ Metodología"])

# TODO 5 del starter: top 3 y explicación por producto.
with tab1:
    st.subheader(f"Sugerencias para el cliente {cid}")
    compras = [productos[j] for j in np.flatnonzero(X[fila])]
    st.write("**Ya compró:**", ", ".join(compras) if compras else "Sin historial (arranque en frío)")
    columnas = st.columns(3)
    for posicion, columna in enumerate(recomendaciones):
        producto = catalogo.iloc[int(columna)]
        with columnas[posicion]:
            st.markdown(f"### {posicion + 1}. {producto['item']}")
            st.caption(str(producto["categoria"]))
            st.write(f"**Precio:** US${float(producto['precio']):.2f}")
            st.write(f"**Puntaje del modelo:** {float(matriz_puntajes[fila, columna]):.3f}")
            st.info("**¿Por qué?** " + por_que(metodo, fila, int(columna), K))
    st.caption("No se recomienda Agua ni productos ya comprados.")

# TODO 6 del starter: comparación solamente con control elegible.
with tab2:
    st.subheader("Comparación de recomendadores")
    st.caption(f"Evaluación con {int(elegibles.sum()):,} clientes de control que probaron algo nuevo.")
    resultados = comparar(K)

    # La comparación causal por recomendador requiere datos de resultados contrafactuales
    # del paso 5 del notebook; no se puede calcular a partir del hit rate.
    comparacion_csv = BASE / "comparacion_incremental.csv"
    if comparacion_csv.is_file():
        incremental = pd.read_csv(comparacion_csv)
        necesarias = {"Recomendador", "Ingreso incremental (US$/cliente)"}
        if necesarias.issubset(incremental.columns) and incremental["Recomendador"].is_unique:
            resultados = resultados.merge(incremental[list(necesarias)], on="Recomendador", how="left")
        else:
            st.warning("comparacion_incremental.csv necesita columnas 'Recomendador' e "
                       "'Ingreso incremental (US$/cliente)', con un registro por método.")
    else:
        st.info("**Ingreso incremental:** no está disponible aún. Al terminar el paso 5 del notebook, "
                "puedes subir un archivo `comparacion_incremental.csv` a Semana_4 con las columnas "
                "`Recomendador` e `Ingreso incremental (US$/cliente)` calculadas en tu laboratorio.")

    st.dataframe(resultados.style.format({
        "Hit rate@3 (control)": "{:.1%}", "Precisión@3": "{:.1%}",
        "Cobertura (productos)": "{:.0f}",
        **({"Ingreso incremental (US$/cliente)": "US${:.3f}"}
           if "Ingreso incremental (US$/cliente)" in resultados else {}),
    }), hide_index=True, use_container_width=True)
    st.bar_chart(resultados.set_index("Recomendador")["Hit rate@3 (control)"])
    st.warning("Un hit rate alto NO significa que el carrusel haya generado esas compras. "
               "Para afirmar ingreso adicional se necesita el análisis causal del experimento.")

with tab3:
    st.subheader("Cómo interpretar la aplicación")
    st.write("**Popularidad:** productos más comprados por todos. **Ítem:** similitud de productos "
             "según compras previas. **Cliente:** usa los K vecinos más parecidos. "
             "**SVD:** identifica patrones compartidos de compra. "
             "**Final:** usa ítem si hay historial y popularidad para clientes nuevos.")
    st.write("**Hit rate@3:** porcentaje de clientes evaluables del control con al menos una "
             "coincidencia en las tres recomendaciones. **Precisión@3:** coincidencias promedio "
             "divididas entre tres. **Cobertura:** productos distintos recomendados entre los "
             "clientes evaluados.")
    st.caption("Los modelos se entrenan solo con historial; el holdout se reserva para evaluar. "
               "El incremento de ingresos debe obtenerse del análisis del paso 5 del notebook.")
