import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import altair as alt



CLUSTERS_C = {
    0: "#1A365D",  # Cluster 0: 
    1: "#1096f1",  # Cluster 1: 
    2: "#2F855A",  # Cluster 2: 
    3: "#ab8a0a",  # Cluster 3: 
    4: "#D32F2F",  # Cluster 4: 
}

NOMBRES_CLUSTERS = {
    0: "Neutros Nulos",
    1: "Malestar Cotidiano",
    2: "Indignación",
    3: "Noticias Logros",
    4: "Crónica Roja"
}
COLORES_POR_NOMBRE = {NOMBRES_CLUSTERS.get(k, str(k)): v for k, v in CLUSTERS_C.items()}

# --- Función Auxiliar para Gauges (Corregida y Ajustada) ---
def _dibujar_reloj(title, value, color, texto_estado):
    """
    Crea la figura Plotly para el medidor con rango [-1, 1].
    
    Args:
        title (str): Título del gráfico (Categoría).
        value (float): Valor numérico.
        color (str): Color de la barra.
        texto_estado (str): Texto 'Positivo', 'Negativo' o 'Neutro'.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 18, 'color': '#ffffff', 'family': "Jost"}},
        number={'font': {'color': '#ffffff', 'size': 24, 'family': "Jost"}}, 
        gauge={
            # CAMBIO 1: Rango correcto [-1, 1] y ticks en los extremos
            'axis': {
                'range': [-1, 1], 
                'tickwidth': 1, 
                'tickcolor': "#ffffff", 
                'tickvals': [-1, -0.5, 0, 0.5, 1], # Marcas que incluyen los extremos exactos
                'ticktext': ["-1", "-0.5", "0", "0.5", "1"] # Texto explícito
            },
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [{'range': [-1, 1], 'color': '#0e3351'}], # Fondo oscuro para todo el rango
        }
    ))

    # CAMBIO 2: Anotación para el texto de estado (Positivo/Negativo)
    fig.add_annotation(
        x=0.5, y=0,  # Posición abajo al centro (coordenadas relativas del plot)
        text=texto_estado,
        showarrow=False,
        font=dict(size=14, color="#ffffff", family="Jost"),
        yshift=-10 # Desplazar un poco hacia abajo para que no choque con la aguja
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff', 'family': "Jost"},
        height=200,
        # CAMBIO 3: Aumentar márgenes para reducir visualmente el gráfico un 20% aprox (Efecto 80% tamaño)
        margin=dict(l=15, r=25, t=10, b=10) 
    )
    return fig

# --- 1. Medidores (Gauge) ---
def mostrar_medidores(df_datos_default, columnas_por_fila=4):
    """
    Muestra los medidores de sentimiento basados en el valor numérico.
    Prioriza los datos cargados por el administrador (viz_sentiment_df).
    """
    # Leer datos desde st.session_state si están disponibles
    df_datos = st.session_state.get('viz_sentiment_df', df_datos_default)

    # Definimos los colores directamente
    COLOR_POSITIVO = "#36aa7f"
    COLOR_NEGATIVO = "#d32f2f"
    COLOR_NEUTRO = "#ab8a0a"

    if df_datos is not None and not df_datos.empty:
        st.subheader("Análisis de Sentimiento de Incidentes Delictivos")
        
        # Usamos el valor de columnas_por_fila pasado por app.py o el default
        num_rows = (len(df_datos) + columnas_por_fila - 1) // columnas_por_fila
        
        for r in range(num_rows):
            cols = st.columns(columnas_por_fila)
            for c in range(columnas_por_fila):
                i = r * columnas_por_fila + c
                if i < len(df_datos):
                    row = df_datos.iloc[i]
                    categoria = row['categorias'] 
                    valor_grafico = row['sentiment_compound']
                    
                    # --- LÓGICA DE COLOR Y TEXTO ---
                    if valor_grafico >= 0.05:
                        color = COLOR_POSITIVO
                        texto_estado = "Positivo"
                    elif valor_grafico <= -0.05:
                        color = COLOR_NEGATIVO
                        texto_estado = "Negativo"
                    else:
                        color = COLOR_NEUTRO
                        texto_estado = "Neutro"
                    # ------------------------

                    # Pasamos el color y el texto calculado a la función de dibujo
                    fig = _dibujar_reloj(categoria, valor_grafico, color, texto_estado)
                    
                    with cols[c]:
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos disponibles para los medidores.")


# --- 2. Cluster 3D (MODIFICADO) ---
def mostrar_cluster_3d(df_default):
    """
    Genera un gráfico de dispersión 3D interactivo con NOMBRES PERSONALIZADOS.
    """
    df = st.session_state.get('viz_cluster_df', df_default)

    st.subheader("Cluster 3D: Sentimiento vs. Categoría vs. Impresiones")
    if df is not None and not df.empty:
        plot_df = df.copy()
        
        # 1. Creamos la columna con el NOMBRE real (ej: "Alta Viralidad")
        plot_df['nombre_cluster'] = plot_df['cluster_label'].apply(lambda x: NOMBRES_CLUSTERS.get(x, str(x)))

        if 'impresiones_log' not in plot_df.columns:
            plot_df['impresiones_log'] = np.log1p(plot_df['public_metrics.impression_count']) if 'public_metrics.impression_count' in plot_df.columns else 0

        # 2. Usamos 'nombre_cluster' para el color
        fig = px.scatter_3d(
            plot_df, x='sentiment_compound', y='categorias', z='impresiones_log',
            color='nombre_cluster',
            title="", opacity=0.7,
            hover_data={'impresiones_log': True, 'nombre_cluster': True, 'cluster_label': False},
            # 3. Mapeamos usando el diccionario Nombre->Color
            color_discrete_map=COLORES_POR_NOMBRE 
        )

        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis_title='Sentimiento (Compound Score)',
                yaxis_title='Categoría',
                zaxis_title='Impresiones (Log-Transformadas)',
            ),
            # La leyenda ahora muestra "Tipo de Perfil"
            legend=dict(title="Tipo de Perfil", font=dict(color="#ffffff"), bgcolor="rgba(0,0,0,0)"),
            font=dict(color="#ffffff", family="Jost")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No hay datos disponibles para generar el gráfico 3D.")


# --- 3. Boxplot y Estadísticas ---
def mostrar_boxplot_y_stats(df_default):
    """
    Genera un Box Plot interactivo y estadísticas descriptivas con NOMBRES PERSONALIZADOS.
    """
    df = st.session_state.get('viz_cluster_df', df_default)

    if df is not None and not df.empty:
        # Preparamos los nombres
        plot_df = df.copy()
        plot_df['nombre_cluster'] = plot_df['cluster_label'].apply(lambda x: NOMBRES_CLUSTERS.get(x, str(x)))

        opcion = st.radio(
            "Analizar distribución de:",
            ["Sentimiento", "Impresiones"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if opcion == "Sentimiento":
            columna_y = 'sentiment_compound'
            titulo_y = "Sentimiento (Compound Score)"
            titulo_grafico = "Variabilidad de Sentimiento por Perfil"
        else:
            if 'impresiones_log' not in plot_df.columns and 'public_metrics.impression_count' in plot_df.columns:
                 plot_df['impresiones_log'] = np.log1p(plot_df['public_metrics.impression_count'])
            elif 'impresiones_log' not in plot_df.columns:
                 st.error("Columna 'impresiones_log' faltante.")
                 return

            columna_y = 'impresiones_log' 
            titulo_y = "Impresiones (Escala Log)"
            titulo_grafico = "Alcance (Impresiones) por Perfil"

        # Boxplot usando Nombres
        fig = px.box(
            plot_df, x='nombre_cluster', y=columna_y, color='nombre_cluster',
            title=titulo_grafico, points="outliers", notched=False,
            # Usamos el mapa auxiliar (Nombre -> Color)
            color_discrete_map=COLORES_POR_NOMBRE 
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={'color': '#ffffff', 'family': 'Jost'}, showlegend=False, 
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis_title="Tipo de Perfil", yaxis_title=titulo_y
        )
        st.plotly_chart(fig, use_container_width=True)

        
        # (PARTE 2: ESTADÍSTICAS)
        st.markdown("---")
        
        with st.expander("Ver Resumen y Estadísticas de los Perfiles", expanded=False):
            st.subheader("Resumen de los Perfiles")

            clusters_ids = sorted(df['cluster_label'].unique())
            
            # Pestañas con NOMBRES (Bajo Impacto, Alta Viralidad, etc.)
            nombres_tabs = [NOMBRES_CLUSTERS.get(c, f"Cluster {c}") for c in clusters_ids]
            tabs = st.tabs(nombres_tabs)

            for i, cluster_id in enumerate(clusters_ids):
                with tabs[i]:
                    data_cluster = df[df['cluster_label'] == cluster_id]
                    
                    sent_mean = data_cluster['sentiment_compound'].mean()
                    sent_max = data_cluster['sentiment_compound'].max()
                    sent_min = data_cluster['sentiment_compound'].min()
                    
                    col_imp = 'public_metrics.impression_count'
                    if col_imp not in df.columns:
                          col_imp = 'impresiones_log'
                          
                    imp_mean = data_cluster[col_imp].mean()
                    imp_max = data_cluster[col_imp].max()

                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        st.markdown("**Sentimiento**")
                        st.metric("Promedio", f"{sent_mean:.2f}")
                        st.caption(f"Max: {sent_max:.2f} | Min: {sent_min:.2f}")
                    
                    with c2:
                        st.markdown("**Impresiones**")
                        st.metric("Promedio", f"{imp_mean:,.0f}")
                        st.caption(f"Máximo alcance: {imp_max:,.0f}")

                    with c3:
                        st.markdown("**Top 3 Delitos**")
                        top_categorias = data_cluster['categorias'].value_counts().head(3)
                        
                        if not top_categorias.empty:
                            for cat, count in top_categorias.items():
                                st.write(f"• **{cat}**: {count} casos")
                        else:
                            st.write("Sin datos de categoría")

    else:
        st.warning("No hay datos disponibles.")