import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# === FUNCIÓN AUXILIAR DE LIMPIEZA ===
def normalizar_texto(texto):
    """
    Normaliza nombres para asegurar el cruce (Map Matching).
    Ej: "Cuauhtémoc" -> "CUAUHTEMOC", "Gustavo A. Madero" -> "GUSTAVO A MADERO"
    """
    if not isinstance(texto, str):
        return str(texto)
    
    # 1. Mayúsculas y espacios
    texto = texto.upper().strip()
    
    # 2. Eliminar acentos
    reemplazos = {'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U'}
    for acento, sin_acento in reemplazos.items():
        texto = texto.replace(acento, sin_acento)
        
    # 3. Eliminar puntuación (puntos)
    texto = texto.replace(".", "")
    
    return texto

# === FUNCIÓN PRINCIPAL ===
def render_mapa_predicciones(df_pred_default, gdf):
    """
    Renderiza el Tablero de Predicción basado en 'mapa_predicciones_24h.csv'.
    Prioriza los datos cargados por el administrador.
    
    Args:
        df_pred_default: DataFrame con ['alcaldia', 'delito', 'count_delitos_predichos'] por defecto.
        gdf: GeoDataFrame de límites de alcaldías.
    """
    # Leer datos desde st.session_state si están disponibles
    df_pred = st.session_state.get('viz_mapa_temporal_df', df_pred_default)

    if df_pred is None or gdf is None:
        st.warning("No hay datos disponibles para visualizar.")
        return

    # 1. PREPROCESAMIENTO (Normalización para mapas)
    # Trabajamos sobre una copia para no alterar el original en caché
    df_viz = df_pred.copy()
        
    # Aplicar normalización a la columna de alcaldía del CSV
    df_viz['alcaldia'] = df_viz['alcaldia'].apply(normalizar_texto)
    
    # Crear llave de cruce en el GeoJSON
    if 'match_key' not in gdf.columns:
        gdf['match_key'] = gdf['NOMGEO'].apply(normalizar_texto)

    # 2. PANEL DE FILTROS (PLEGABLE)
    with st.expander("Filtros de Predicción (Alcaldía / Delito)", expanded=False):
        c1, c2 = st.columns(2)
        
        # A) Filtro Alcaldía
        opciones_alcaldias = ["TODAS"] + sorted(df_viz['alcaldia'].unique())
        with c1:
            sel_alcaldias_input = st.multiselect(
                "Alcaldías:", 
                options=opciones_alcaldias, 
                default=["TODAS"]
            )
        
        # Lógica "TODAS"
        if "TODAS" in sel_alcaldias_input or not sel_alcaldias_input:
            sel_alcaldias = df_viz['alcaldia'].unique()
        else:
            sel_alcaldias = sel_alcaldias_input

        # B) Filtro Delito
        opciones_delitos = ["TODAS"] + sorted(df_viz['delito'].unique())
        with c2:
            sel_delitos_input = st.multiselect(
                "Delitos:", 
                options=opciones_delitos, 
                default=["TODAS"]
            )
            
        # Lógica "TODAS"
        if "TODAS" in sel_delitos_input or not sel_delitos_input:
            sel_delitos = df_viz['delito'].unique()
        else:
            sel_delitos = sel_delitos_input

    # 3. APLICACIÓN DE FILTROS
    df_filtrado = df_viz[
        (df_viz['alcaldia'].isin(sel_alcaldias)) & 
        (df_viz['delito'].isin(sel_delitos))
    ]

    # 4. AGRUPACIÓN Y MÉTRICAS
    # Usamos la nueva columna 'count_delitos_predichos'
    df_agrupado = df_filtrado.groupby('alcaldia')['count_delitos_predichos'].sum().reset_index()
    df_agrupado.columns = ['alcaldia', 'total_predicho']

    # Crear HTML para el Tooltip (Top 5 delitos)
    def crear_resumen_html(sub_df):
        # Agrupar por delito y sumar predicciones
        g = sub_df.groupby('delito')['count_delitos_predichos'].sum().reset_index()
        g = g.sort_values('count_delitos_predichos', ascending=False).head(5)
        
        filas = [f"<b>{row['delito']}:</b> {float(row['count_delitos_predichos']):.1f}" for _, row in g.iterrows()]
        if not filas: return "Sin registros"
        return "<br>".join(filas)

    # Generar el detalle HTML por alcaldía
    if not df_filtrado.empty:
        df_resumen = df_filtrado.groupby('alcaldia').apply(crear_resumen_html).reset_index()
        df_resumen.columns = ['alcaldia', 'detalle_html']
    else:
        df_resumen = pd.DataFrame(columns=['alcaldia', 'detalle_html'])

    # Merge con GeoData (Left Join para mantener el mapa completo aunque esté en gris)
    gdf_mapa = gdf.merge(df_agrupado, left_on="match_key", right_on="alcaldia", how="left")
    gdf_mapa = gdf_mapa.merge(df_resumen, on="alcaldia", how="left")
    
    # Rellenar Nulos
    gdf_mapa['total_predicho'] = gdf_mapa['total_predicho'].fillna(0)
    gdf_mapa['detalle_html'] = gdf_mapa['detalle_html'].fillna("Sin predicciones con filtros actuales")

    # 5. VISUALIZACIÓN DE KPIs
    total_global = df_filtrado['count_delitos_predichos'].sum()
    
    if total_global > 0:
        top_row = df_agrupado.sort_values('total_predicho', ascending=False).iloc[0]
        nombre_top = top_row['alcaldia']
        val_top = top_row['total_predicho']
    else:
        nombre_top = "N/A"
        val_top = 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Predicciones (24h)", f"{total_global:.0f}")
    k2.metric("Alcaldía con Mayor Riesgo", nombre_top, f"{val_top:.1f} delitos")
    k3.metric("Filtros Activos", f"{len(sel_delitos)} tipos de delito")

    # 6. VISUALIZACIÓN DEL MAPA
    m = folium.Map(location=[19.4326, -99.1332], zoom_start=10, tiles="Cartodb dark_matter")
    
    # Capa de Color (Choropleth)
    folium.Choropleth(
        geo_data=gdf_mapa,
        data=gdf_mapa,
        columns=['NOMGEO', 'total_predicho'], 
        key_on='feature.properties.NOMGEO',
        fill_color='PuBu',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Incidencia Predicha (24h)',
        highlight=True
    ).add_to(m)

    style_legend = """
    <style>
        /* El título de la leyenda (Incidencia Predicha) */
        .caption {
            color: white !important;
            font-family: 'Jost', sans-serif !important;
            font-weight: bold !important;
            font-size: 14px !important;
        }
        /* Los números de la escala (SVG text) */
        .legend text {
            fill: white !important; /* En SVG el color se llama 'fill' */
            font-family: 'Jost', sans-serif !important;
        }
        /* Fondo opcional para la leyenda para mejorar lectura sobre el mapa */
        .legend {
            background-color: rgba(15, 15, 31, 0.7); /* Tu fondo oscuro semitransparente */
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #2a2a3e;
        }
    </style>
    """
    m.get_root().html.add_child(folium.Element(style_legend))


    # Capa Interactiva (Tooltip)
    folium.GeoJson(
        gdf_mapa,
        style_function=lambda x: {'color': 'transparent', 'fillColor': 'transparent', 'weight': 0},
        highlight_function=lambda x: {'weight': 3, 'color': '#666'},
        tooltip=folium.GeoJsonTooltip(
            fields=['NOMGEO', 'total_predicho'], 
            aliases=['Alcaldía:', 'Predicción Total:'],
            style="font-size: 14px; font-weight: bold;"
        ),
        popup=folium.GeoJsonPopup(
            fields=['NOMGEO', 'detalle_html'],
            aliases=['Alcaldía', 'Top Delitos Previstos'],
            localize=True,
            labels=False,
            style="font-family: Jost;"
        )
    ).add_to(m)

    st_folium(m, width="100%", height=500)

    # 7. DESGLOSE ANALÍTICO (Debajo del mapa)
    with st.expander("Desglose Analítico Detallado", expanded=False):
        c_chart, c_table = st.columns(2)
        
        with c_chart:
            st.markdown("**Top 10 Delitos Previstos**")
            if not df_filtrado.empty:
                top_delitos_viz = df_filtrado.groupby('delito')['count_delitos_predichos'].sum().sort_values(ascending=False).head(10)
                st.bar_chart(top_delitos_viz, color="#1096f1")
            else:
                st.info("Sin datos.")

        with c_table:
            st.markdown("**Incidencia por Alcaldía**")
            if not df_agrupado.empty:
                df_display = df_agrupado.sort_values('total_predicho', ascending=False).reset_index(drop=True)
                max_val = float(df_display['total_predicho'].max())
                df_styled = df_display.style.bar(
                    subset=['total_predicho'], 
                    color='#1096f1', 
                    vmin=0, 
                    vmax=max_val
                ).format({"total_predicho":"{:.1f}"})

                st.dataframe(
                    df_styled,
                    column_config={
                        "alcaldia": "Alcaldía",
                        "total_predicho": "Predicción" # Solo cambiamos el nombre de la columna
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=300 # Opcional: fija la altura si quieres evitar scroll excesivo
                )