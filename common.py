# common.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import altair as alt

# --- Variables Comunes ---
USERS = {
    "thales_admin_01_@gmail.com": {"password": "1", "role": "admin"},
    "thales_admin_02@gmail.com": {"password": "adminT02E3", "role": "admin"},
    "thales_admin_03@gmail.com": {"password": "adminT03E3", "role": "admin"},
    "usr01@gmail.com": {"password": "userT01E3", "role": "user"},
    "usr02@gmail.com": {"password": "userT02E3", "role": "user"},
    "usr03@gmail.com": {"password": "userT03E3", "role": "user"}
}

# --- Funciones de Sesión y Utilidad ---
def logout():
    """Maneja el cierre de sesión."""
    st.divider() 
    if st.button("Cerrar sesión"):
        st.session_state['logged_in'] = False
        st.session_state['user_type'] = None
        st.rerun()

# --- Lógica de Gráficos (Dashboard) ---
def renderizar_graficos_dashboard(titulo):
    """Renderiza los gráficos del dashboard (lógica común)."""
    
    # [CÓDIGO COMPLETO DE renderizar_graficos_dashboard]
    
    def create_gauge(title, value, color="#3d8bfd"):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=value,
            title={'text': title, 'font': {'size': 18, 'color': '#ffffff'}},
            number={'suffix': "%", 'font': {'color': '#ffffff'}},
            gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#a0a0c0"},
                   'bar': {'color': color, 'thickness': 0.75},
                   'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 0,
                   'steps': [{'range': [0, 100], 'color': '#1e1e3f'}]}
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#ffffff'}, height=200, margin=dict(l=10, r=10, t=50, b=10)
        )
        return fig

import streamlit as st

import streamlit as st

# En common.py

def cargar_estilos_globales():
    # Usamos CSS para forzar estilos visuales específicos
    
    st.markdown("""
    <style>
        /* 1. IMPORTAR FUENTE JOST */
        @import url('https://fonts.googleapis.com/css2?family=Jost:wght@300;400;600;700&display=swap');

        /* 2. APLICAR JOST A TODO */
        html, body, [data-testid="stAppViewContainer"], p, h1, h2, h3, h4, h5, h6, 
        .stMarkdown, .stText, button, input, select, textarea, label, .stRadio {
            font-family: 'Jost', sans-serif !important;
        }
        
        /* Excepción: Iconos */
        .material-icons, [data-testid="stExpander"] svg {
            font-family: 'Material Icons' !important;
        }

        /* 3. RECUPERAR EL COLOR DE LAS CAJAS DE TEXTO (Multiselect) */
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #1096f1 !important;
            color: white !important;
            border: 1px solid #1096f1 !important;
        }
        .stMultiSelect [data-baseweb="tag"] svg {
            fill: white !important;
        }

        /* 4. BORDE FINO EN LAS TARJETAS */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #1a1a2e;
            border-radius: 10px;
        }

        /* 5. METRICAS Y LIMPIEZA */
        div[data-testid="stMetricValue"] { font-size: 1.2rem; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

/* ------------------------------------------------------- */
        /* 6. RADIO BUTTONS (LA CORRECCIÓN DE FRANCOTIRADOR) */
        /* ------------------------------------------------------- */

        /* ESTADO: SELECCIONADO */
        /* Buscamos la etiqueta (label) que CONTIENE un input marcado (:checked) */
        /* Y atacamos al primer DIV que encontramos dentro, que es siempre el círculo visual */
        
        div[role="radiogroup"] label:has(input:checked) > div:first-of-type {
            background-color: #1096f1 !important;
            border-color: #1096f1 !important;
        }
        
        /* El punto interno (el div dentro del div) lo pintamos blanco */
        div[role="radiogroup"] label:has(input:checked) > div:first-of-type > div {
            background-color: white !important;
        }

        /* ESTADO: NO SELECCIONADO */
        /* Buscamos la etiqueta que tiene un input NO marcado */
        div[role="radiogroup"] label:has(input:not(:checked)) > div:first-of-type {
            border-color: #e0e0ff !important;    /* Borde grisáceo/blanco */
            background-color: transparent !important;
        }
        
        /* Ocultamos el punto interno cuando no está seleccionado */
        div[role="radiogroup"] label:has(input:not(:checked)) > div:first-of-type > div {
            background-color: transparent !important;
        }
        
        /* AJUSTE FINAL: Aseguramos que el texto vuelva a ser blanco normal */
        div[role="radiogroup"] label p {
            color: white !important;
            background-color: transparent !important; /* Por si acaso se quedó el fondo azul */
        }

    </style>
    """, unsafe_allow_html=True)

def create_gauge(label, value, color_grafico="#1E90FF"):
    """
    Crea un gráfico de medidor (Gauge) estilizado con fuente Jost y tema oscuro.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': label, 
            'font': {'size': 18, 'family': "Jost", 'color': "#e0e0ff"} # Color de texto claro
        },
        number={
            'font': {'size': 24, 'family': "Jost", 'color': color_grafico} # Número en azul
        },
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#0f0f1f", 'visible': False},
            'bar': {'color': color_grafico}, # La barra de progreso en azul
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 100], 'color': "#2b313c"} # Fondo gris oscuro del arco
            ],
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", # Fondo transparente para integrarse al contenedor
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Jost"},
        margin=dict(l=10, r=10, t=30, b=10),
        height=150 # Altura ajustada para que quepan bien en las columnas
    )
    
    return fig

    # --- Fila 1: Medidores y Mapa ---
    col1, col2 = st.columns([1, 1.3])
    with col1:
        with st.container(border=True):
            st.subheader("Tendencias tipo de delito")
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(create_gauge("ROBO", 75), use_container_width=True)
                st.plotly_chart(create_gauge("ABUSO", 30), use_container_width=True)
            with g2:
                st.plotly_chart(create_gauge("ROBO S/VIOLENCIA", 40), use_container_width=True)
                st.plotly_chart(create_gauge("SECUESTRO", 85), use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("Incidencias por alcaldía")
            map_text, map_chart = st.columns([1, 1.5])
            with map_text:
                st.text("Alcaldías más peligrosas")
                st.markdown("<h4>Cuauhtémoc<br>Iztapalapa<br>Gustavo A. Madero</h4>", unsafe_allow_html=True)
                st.text("Periodo")
                st.markdown("<h5>Últimos 30 días</h5>", unsafe_allow_html=True)
            with map_chart:
                map_data = pd.DataFrame({
                    'lat': [19.4326, 19.3553, 19.4913],
                    'lon': [-99.1332, -99.0621, -99.1115],
                    'size': [15, 15, 15]
                })
                st.map(map_data, zoom=9.5, use_container_width=True, size='size', color="#3d8bfd")

    # --- Fila 2: Cluster, Barras y Líneas ---
    col3, col4, col5 = st.columns(3)
    with col3:
        with st.container(border=True):
            st.subheader("Cluster")
            np.random.seed(42)
            c_data1 = np.random.randn(20, 2) + np.array([0, 0])
            c_data2 = np.random.randn(20, 2) + np.array([5, 5])
            c_df1 = pd.DataFrame(c_data1, columns=['x', 'y']); c_df1['cluster'] = 'A'
            c_df2 = pd.DataFrame(c_data2, columns=['x', 'y']); c_df2['cluster'] = 'B'
            c_df_total1 = pd.concat([c_df1, c_df2])

            c_data3 = np.random.randn(20, 2) + np.array([0, 5])
            c_data4 = np.random.randn(20, 2) + np.array([5, 0])
            c_df3 = pd.DataFrame(c_data3, columns=['x', 'y']); c_df3['cluster'] = 'C'
            c_df4 = pd.DataFrame(c_data4, columns=['x', 'y']); c_df4['cluster'] = 'D'
            c_df_total2 = pd.concat([c_df3, c_df4])

            chart1 = alt.Chart(c_df_total1).mark_circle().encode(
                x=alt.X('x', axis=None), y=alt.Y('y', axis=None),
                color=alt.Color('cluster', legend=None, scale={'domain': ['A', 'B'], 'range': ['#e0e0ff', '#1096f1']})
            ).properties(height=200).configure_view(strokeWidth=0).configure(background="rgba(0,0,0,0)")

            chart2 = alt.Chart(c_df_total2).mark_circle().encode(
                x=alt.X('x', axis=None), y=alt.Y('y', axis=None),
                color=alt.Color('cluster', legend=None, scale={'domain': ['C', 'D'], 'range': ['#3d8bfd', '#00d4d4']})
            ).properties(height=200).configure_view(strokeWidth=0).configure(background="rgba(0,0,0,0)")

            plot_col1, plot_col2 = st.columns(2)
            with plot_col1: st.altair_chart(chart1, use_container_width=True)
            with plot_col2: st.altair_chart(chart2, use_container_width=True)

    with col4:
        with st.container(border=True):
            st.subheader("Días con más incidencias")
            bar_data = pd.DataFrame({
                'día': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
                'incidencias': [12, 18, 15, 12, 25, 19, 14, 11]
            })
            bar_chart = alt.Chart(bar_data).mark_bar(color="#e0e0ff").encode(
                x=alt.X('día', axis=None), y=alt.Y('incidencias', axis=None)
            ).properties(height=240).configure_view(strokeWidth=0).configure(background="rgba(0,0,0,0)")
            st.altair_chart(bar_chart, use_container_width=True)

    with col5:
        with st.container(border=True):
            st.subheader("Predicción de delito")
            meses = ['Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            n = len(meses)
            df_line = pd.DataFrame({
                'mes_num': list(range(n)) * 2, 'mes': meses * 2,
                'valor': np.concatenate([np.array([10, 12, 18, 15, 17]), np.array([8, 10, 11, 14, 12])]),
                'zona': ['Centro'] * n + ['Periferia'] * n
            })
            line_chart = alt.Chart(df_line).mark_line(point=True).encode(
                x=alt.X('mes_num', title=None, axis=alt.Axis(labels=True, ticks=True, domain=False, grid=True, gridColor="#2a2a3e", labelColor="#a0a0c0", values=list(range(n)), labelExpr="datum.label == 0 ? 'Ago' : datum.label == 1 ? 'Sep' : datum.label == 2 ? 'Oct' : datum.label == 3 ? 'Nov' : 'Dic'")),
                y=alt.Y('valor', title=None, axis=alt.Axis(labels=False, ticks=False, domain=False, grid=True, gridColor="#2a2a3e")),
                color=alt.Color('zona', legend=alt.Legend(orient="top", title=None, labelColor='#e0e0ff', symbolStrokeColor='#e0e0ff'), scale={'domain': ['Centro', 'Periferia'], 'range': ['#3d8bfd', '#00d4d4']})
            ).properties(height=240).configure_view(strokeWidth=0).configure(background="rgba(0,0,0,0)").interactive()
            st.altair_chart(line_chart, use_container_width=True)

