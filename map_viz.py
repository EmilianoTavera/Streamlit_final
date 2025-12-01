import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import pandas as pd

def render_mapa_contexto(df_map_default, delegaciones_gdf):
    """
    Renderiza el mapa avanzado con filtros de contexto.
    Prioriza los datos cargados por el administrador (viz_mapa_global_df).
    
    Args:
        df_map_default (pd.DataFrame): Dataframe con columnas latitud, longitud, delito, trabajo, etc.
        delegaciones_gdf (gpd.GeoDataFrame): Geodataframe con los polígonos de las alcaldías.
    """
    # Leer datos desde st.session_state si están disponibles
    df_map = st.session_state.get('viz_mapa_global_df', df_map_default)
    
    if df_map is None or df_map.empty or delegaciones_gdf is None:
        st.warning("No hay datos disponibles para el Mapa de Contexto.")
        return
    
    # CSS Específico para ajustar métricas dentro del contenedor
    st.markdown("""
    <style>
        div[data-testid="stMetricValue"] {font-size: 1.2rem;}
    </style>
    """, unsafe_allow_html=True)

    # === 1. PANEL DE CONTROL (EXPANDER) ===
    with st.expander("Configuración Avanzada de Mapa y Filtros", expanded=True):
        # --- Fila A: Filtros Geográficos y de Tipo ---
        c1, c2 = st.columns([1, 2])
        with c1:
            # Ordenamos alcaldías y agregamos opción TODAS
            opciones_alcaldia = ["TODAS"] + sorted(df_map["alcaldia_hecho"].astype(str).unique())
            alcaldia_sel = st.selectbox("Alcaldía", opciones_alcaldia)
        with c2:
            delitos_disponibles = sorted(df_map["delito"].unique())
            delito_sel = st.multiselect("Tipos de Delito (Dejar vacío para ver todos)", delitos_disponibles)

        st.markdown("---")
        
        # --- Fila B: Filtros de Contexto (Tri-estado) ---
        st.write(" **Filtros de Contexto** (Selecciona qué ver)")
        f1, f2, f3 = st.columns(3)
        
        # Diccionario para mapear selección a valores lógicos
        opciones_binarias = ["Todos", "Sí (1)", "No (0)"]
        
        with f1:
            opt_trabajo = st.selectbox("Relacionado al Trabajo", opciones_binarias)
        with f2:
            opt_centro = st.selectbox("Zona Centro", opciones_binarias)
        with f3:
            opt_horario = st.selectbox("Horario Laboral", opciones_binarias)

        st.markdown("---")

        # --- Fila C: Configuración Visual ---
        v1, v2 = st.columns(2)
        with v1:
            color_by = st.selectbox("Colorear mapa por:", 
                                     ["Tipo de Delito", "Relacionado con Trabajo", "En Zona Centro", "Horario Laboral"])
        with v2:
            ver_capas = st.multiselect("Capas a mostrar", ["Alcaldías", "Heatmap", "Puntos"], default=["Alcaldías", "Puntos"])

    # === 2. LÓGICA DE FILTRADO ===
    df_filtered = df_map.copy()

    # A. Alcaldía
    if alcaldia_sel != "TODAS":
        df_filtered = df_filtered[df_filtered["alcaldia_hecho"] == alcaldia_sel]

    # B. Delitos
    if delito_sel:
        df_filtered = df_filtered[df_filtered["delito"].isin(delito_sel)]

    # C. Contexto
    map_opciones = {"Sí (1)": 1, "No (0)": 0}

    if opt_trabajo != "Todos":
        df_filtered = df_filtered[df_filtered["trabajo"] == map_opciones[opt_trabajo]]
    if opt_centro != "Todos":
        df_filtered = df_filtered[df_filtered["centro"] == map_opciones[opt_centro]]
    if opt_horario != "Todos":
        df_filtered = df_filtered[df_filtered["horario_laboral"] == map_opciones[opt_horario]]

    # === 3. KPIs DINÁMICOS ===
    # Se muestran fuera del expander para que siempre sean visibles
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Incidentes", f"{len(df_filtered):,}")

    if len(df_filtered) > 0:
        pct_trabajo = (df_filtered['trabajo'].sum() / len(df_filtered)) * 100
        pct_centro = (df_filtered['centro'].sum() / len(df_filtered)) * 100
        pct_horario = (df_filtered['horario_laboral'].sum() / len(df_filtered)) * 100
    else:
        pct_trabajo = pct_centro = pct_horario = 0

    k2.metric("% Trabajos", f"{pct_trabajo:.1f}%")
    k3.metric("% Zona Centro", f"{pct_centro:.1f}%")
    k4.metric("% Horario Lab.", f"{pct_horario:.1f}%")

    # === 4. CONSTRUCCIÓN DEL MAPA ===
    
    # Muestreo para rendimiento
    LIMIT = 2000
    if len(df_filtered) > LIMIT:
        st.warning(f"Visualizando muestra de {LIMIT} puntos (Total filtrado: {len(df_filtered)})")
        data_viz = df_filtered.sample(LIMIT, random_state=42)
    else:
        data_viz = df_filtered

    # Mapa Base
    m = folium.Map(location=[19.4326, -99.1332], zoom_start=11, tiles="Cartodb dark_matter")

    # Capa 1: Límites Alcaldías
    if "Alcaldías" in ver_capas and delegaciones_gdf is not None:
        folium.GeoJson(
            delegaciones_gdf,
            name="Límites",
            style_function=lambda x: {"color": "#444444", "weight": 1, "fillOpacity": 0.0},
            tooltip=folium.GeoJsonTooltip(fields=["NOMGEO"], aliases=["Alcaldía:"])
        ).add_to(m)

    # Capa 2: Heatmap
    if "Heatmap" in ver_capas and not data_viz.empty:
        heat_data = data_viz[["latitud", "longitud"]].values
        HeatMap(heat_data, radius=11, blur=15, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)

    # Configuración de Colores según selección
    colores_hex = {}
    campo_color = ""
    titulo_leyenda = ""

    if color_by == "Tipo de Delito":
        campo_color = "delito"
        unique_vals = data_viz["delito"].unique()
        palette = ["#0E3351", "#AB8A0A", "#1096f1", "#36aa7f", "#d32f2f", "#ffff33", "#831c44", "#f781bf"]
        colores_hex = {val: palette[i % len(palette)] for i, val in enumerate(unique_vals)}
        titulo_leyenda = "Delito"
    elif color_by == "Relacionado con Trabajo":
        campo_color = "trabajo"
        colores_hex = {1: "#ab8a0a", 0: "#0e3351"} # Rojo: Sí, Azul: No
        titulo_leyenda = "Trabajo"
    elif color_by == "En Zona Centro":
        campo_color = "centro"
        colores_hex = {1: "#1096f1", 0: "#0e3351"}
        titulo_leyenda = "Centro"
    elif color_by == "Horario Laboral":
        campo_color = "horario_laboral"
        colores_hex = {1: "#36aa7f", 0: "#d32f2f"}
        titulo_leyenda = "Horario Lab"

    # Capa 3: Puntos
    if "Puntos" in ver_capas and not data_viz.empty:
        for _, row in data_viz.iterrows():
            val = row[campo_color]
            c = colores_hex.get(val, "#000000")
            
            popup_content = f"""
            <b>Delito:</b> {row['delito']}<br>
            <b>Trabajo:</b> {'✅' if row['trabajo']==1 else 'No'}<br>
            <b>Centro:</b> {'✅' if row['centro']==1 else 'No'}<br>
            <b>Horario:</b> {'✅' if row['horario_laboral']==1 else 'No'}
            """
            
            folium.CircleMarker(
                [row["latitud"], row["longitud"]],
                radius=4,
                color=c, fill=True, fill_opacity=0.7, fill_color=c,
                tooltip=f"{row['delito']}",
                popup=folium.Popup(popup_content, max_width=200)
            ).add_to(m)

    # === 5. LEYENDA FLOTANTE HTML ===
    if "Puntos" in ver_capas:
        html_items = ""
        for label, hex_code in colores_hex.items():
            if color_by != "Tipo de Delito":
                txt = "SÍ" if label == 1 else "NO"
            else:
                txt = str(label)
                
            html_items += f"""
            <div style='display: flex; align-items: center; margin-bottom: 4px;'>
                <span style='background:{hex_code}; width:10px; height:10px; border-radius:50%; margin-right:6px;'></span>
                <span style='font-size:11px; color:white;'>{txt}</span>
            </div>
            """

        legend_html = f"""
        <div style="
            position: fixed; 
            bottom: 30px; right: 30px; width: 140px; 
            background-color: #0f0f1f;
            border: 1px solid #1a1a2e; border-radius: 8px; 
            padding: 10px; z-index:9999; font-family: "Jost" , sans-serif;">
            <b style="font-size:12px; display:block; margin-bottom:5px; color:white;">{titulo_leyenda}</b>
            {html_items}
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    # Render final
    st_folium(m, width=800, height=800)
