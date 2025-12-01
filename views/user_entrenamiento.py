import streamlit as st
import time
import sys
import pandas as pd
import io
import zipfile
import re
import unicodedata
# Importamos el procesador de Twitter/Cluster
from processor import process_uploaded_csv 
# Importamos el procesador global
from global_processor import process_global_csv, create_zip_from_dataframe
# Importamos el procesador temporal 
import temporal_processor 

# ------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------

def get_csv_download_link(df, filename):
    """Convierte el DataFrame a CSV para el botón de descarga."""
    if df is not None and not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        return csv
    return b'No data available'

def _load_file_to_dataframe(uploaded_file, file_type):
    """
    Carga un archivo (buffer) en un DataFrame, manejando CSV y ZIP.
    Retorna el DataFrame o None si hay error.
    """
    if uploaded_file is None:
        return None
    
    # 1. Obtener el buffer de bytes
    data_buffer = uploaded_file.getvalue()
    
    try:
        if file_type == 'zip':
            # Manejar archivos ZIP
            zip_input = io.BytesIO(data_buffer)
            with zipfile.ZipFile(zip_input, 'r') as zf:
                # Buscar el primer archivo CSV dentro del ZIP
                csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
                if not csv_files:
                    st.error("❌ El archivo ZIP no contiene un CSV.")
                    return None
                with zf.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(io.TextIOWrapper(csv_file, encoding='utf-8'))
                    return df
        
        elif file_type == 'csv':
            # Manejar archivos CSV
            df = pd.read_csv(io.StringIO(data_buffer.decode('utf-8')))
            return df

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo {uploaded_file.name}: {e}")
        return None
        
    return None

def render():
    st.title(" Entrenamiento de Modelos (Admin)")
    st.markdown("---")
    st.markdown("### Gestión de Datos de Entrenamiento y Resultados del Modelo")

    # =======================================================
    # FILA 1: ANÁLISIS GLOBAL (ZIP)
    # =======================================================
    st.subheader("1. Análisis Global")
    
    col1, col2, col3, col4 = st.columns([1, 2, 1, 2])
    
    with col1:
        st.caption("Carga de Datos (ZIP)")
        uploaded_zip = st.file_uploader(
            "Cargar ZIP de Carpeta FGJ",
            type=['zip'],
            key="upload_global",
            label_visibility="collapsed"
        )
    with col2:
        st.caption("Acción")
        process_global_button = st.button(" Procesar Análisis Global", key="process_global_btn", use_container_width=True)

        if process_global_button:
            if uploaded_zip is not None:
                with st.spinner('Procesando ZIP, Limpieza y Georreferencia...'):
                    df_mapa_res = process_global_csv(uploaded_zip)
                
                if df_mapa_res is not None and not df_mapa_res.empty:
                    zip_data = create_zip_from_dataframe(df_mapa_res, "df_mapa_procesado.csv")
                    st.session_state['zip_mapa_global'] = zip_data
                    st.success("✅ Análisis Global completado. Mapa listo para descargar.")
                    st.rerun()
                else:
                    st.error("❌ Falló el procesamiento global. Verifique el contenido del ZIP (debe contener un CSV con las columnas FGJ).")
            else:
                st.warning("Por favor, sube un archivo ZIP primero.")
        else:
            st.markdown("**Análisis Global**")

    # --- Preparar contenido de descarga (Fila 1) ---
    zip_data_download = st.session_state.get('zip_mapa_global', None)
    download_zip_disabled = zip_data_download is None
        
    with col3:
        st.caption("Descarga de Resultados")
        st.download_button(
            label="Descargar Mapa",
            data=zip_data_download if zip_data_download else b'',
            file_name="mapa_procesado.zip",
            mime="application/zip",
            key="download_mapa_global",
            use_container_width=True,
            disabled=download_zip_disabled
        )
    with col4:
        st.caption("Descripción")
        st.markdown("**Descargar Mapa (ZIP)**")


    st.markdown("---")
    
    # =======================================================
    # FILA 2: DATOS TWITTER (CSV)
    # =======================================================
    st.subheader("2. Datos Twitter")
    
    col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 1, 2, 1, 2])
    
    with col1:
        st.caption("Carga de Datos")
        uploaded_file = st.file_uploader(
            "Cargar Datos Twitter",
            type=['csv'],
            key="upload_twitter",
            label_visibility="collapsed"
        )
    with col2:
        st.caption("Acción")
        if st.button(" Procesar y Generar Resultados", key="process_btn", use_container_width=True):
            if uploaded_file is not None:
                with st.spinner('Procesando datos (Clasificación, Sentimiento, Clustering)...'):
                    df_cluster_res, df_gauge_res = process_uploaded_csv(uploaded_file)
                
                if df_cluster_res is not None and df_gauge_res is not None:
                    st.session_state['df_cluster_res'] = df_cluster_res
                    st.session_state['df_gauge_res'] = df_gauge_res
                    st.success("✅ Procesamiento completado. Archivos listos para descargar.")
                    st.rerun() 
                else:
                    st.error("❌ Fallo el procesamiento. Verifique el formato de su CSV.")
            else:
                st.warning("Por favor, sube un archivo CSV de Twitter primero.")
        else:
            st.markdown("**Datos Twitter**")


    df_cluster_download = st.session_state.get('df_cluster_res', None)
    df_gauge_download = st.session_state.get('df_gauge_res', None)

    cluster_csv = get_csv_download_link(df_cluster_download, "resultados_cluster.csv")
    gauge_csv = get_csv_download_link(df_gauge_download, "resultados_sentiment.csv")
    
    download_disabled = df_cluster_download is None
    
    with col3:
        st.caption("Descarga 1 (Gauge)")
        st.download_button(
            label="Descargar Sentiment",
            data=gauge_csv,
            file_name="resultados_sentiment.csv",
            mime="text/csv",
            key="download_sentiment",
            use_container_width=True,
            disabled=download_disabled
        )
    with col4:
        st.caption("Descripción")
        st.markdown("**Descargar Sentiment**")
        
    with col5:
        st.caption("Descarga 2 (Cluster)")
        st.download_button(
            label="Descargar Cluster",
            data=cluster_csv,
            file_name="resultados_cluster.csv",
            mime="text/csv",
            key="download_cluster",
            use_container_width=True,
            disabled=download_disabled
        )
    with col6:
        st.caption("Descripción")
        st.markdown("**Descargar Cluster**")

    st.markdown("---")

    # =======================================================
    # FILA 3: DATOS TEMPORALES (ZIP / CSV)
    # =======================================================
    st.subheader("3. Datos Temporales (Modelo XGBoost)")
    
    col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 1, 2, 1, 2])
    
    # 3.1 Carga 
    with col1:
        st.caption("Carga de Datos")
        uploaded_temporal_file = st.file_uploader(
            "Cargar ZIP de Datos Temporales",
            type=['zip'],
            key="upload_temporal", 
            label_visibility="collapsed"
        )

    # 3.2 Acción
    should_rerun = False
    with col2:
        st.caption("Acción")
        process_temporal_button = st.button("Procesar XGBoost", key="process_temporal_btn", use_container_width=True)

        if process_temporal_button:
            if uploaded_temporal_file is not None:
                with st.spinner('Entrenando XGBoost y generando predicciones a 24 horas... 🧠'):
                    try:
                        # La función retorna zip_serie (ZIP buffer) y csv_mapa (CSV buffer)
                        zip_serie, csv_mapa_pred = temporal_processor.process_temporal_data(uploaded_temporal_file) 
                        
                        if zip_serie is not None and csv_mapa_pred is not None:
                            st.session_state['zip_serie_temporal'] = zip_serie
                            st.session_state['csv_mapa_temporal'] = csv_mapa_pred # <-- ALMACENA CSV BUFFER
                            st.success("✅ Predicciones Temporales completadas con XGBoost. Archivos listos para descargar.")
                            should_rerun = True
                        else:
                            st.error("❌ Falló el procesamiento temporal. Verifique el formato del CSV interno.")
                    except Exception as e:
                        st.error(f"❌ Error crítico en el procesamiento: {e}")
                        st.code(str(e))
            else:
                st.warning("Por favor, sube un archivo ZIP primero.")
        else:
            st.markdown("**Modelo XGBoost**")

    # 3.3 Descargas
    zip_serie_download = st.session_state.get('zip_serie_temporal', None)
    csv_mapa_download = st.session_state.get('csv_mapa_temporal', None)
    
    download_serie_disabled = zip_serie_download is None
    download_mapa_disabled = csv_mapa_download is None 
    
    with col3:
        st.caption("Descarga 1 (Serie)")
        st.download_button(
            label="Descargar Serie",
            data=zip_serie_download if zip_serie_download else b'',
            file_name="serie_temporal_delitos.zip",
            mime="application/zip",
            key="download_serie_xgb",
            use_container_width=True,
            disabled=download_serie_disabled
        )
    with col4: 
        st.caption("Descripción")
        st.markdown("**Descargar Serie (ZIP)**")
        
    with col5:
        st.caption("Descarga 2 (Mapa)")
        st.download_button(
            label="Descargar Mapa Predicciones",
            data=csv_mapa_download if csv_mapa_download else b'',
            file_name="mapa_predicciones_24h.csv", 
            mime="text/csv", 
            key="download_mapa_serie_xgb",
            use_container_width=True,
            disabled=download_mapa_disabled
        )
    with col6: 
        st.caption("Descripción")
        st.markdown("**Descargar Mapa Predicciones (CSV)**") 
    
    st.markdown("---")

    # =======================================================
    # FILA 4: VISUALIZACIÓN DE RESULTADOS (CARGA DE ARCHIVOS)
    # =======================================================
    st.subheader("4. Carga de Datos para Visualización")

    # Definimos las especificaciones de carga en un diccionario
    # 'target_key' es donde se almacenará el DataFrame final en st.session_state
    VIZ_SPECS = {
        "Graficar Mapa": {"type": "zip", "label": "incidencias_mapa_global", "target_key": "viz_mapa_global_df"},
        "Graficar Sentimiento": {"type": "csv", "label": "sentiment_data", "target_key": "viz_sentiment_df"},
        "Graficar Cluster": {"type": "csv", "label": "cluster_data", "target_key": "viz_cluster_df"},
        "Graficar Serie": {"type": "zip", "label": "serie_temporal_data", "target_key": "viz_serie_temporal_df"},
        "Graficar Mapa Serie": {"type": "csv", "label": "mapa_predicciones_data", "target_key": "viz_mapa_temporal_df"}
    }

    # Creamos las 5 columnas para la carga y acción
    cols_upload = st.columns(5)
    cols_action = st.columns(5)
    
    # Reiniciar la bandera si la ejecución previa no la activó
    if 'should_rerun_viz' not in st.session_state:
        st.session_state['should_rerun_viz'] = False

    st.markdown("---")

    for i, (btn_label, spec) in enumerate(VIZ_SPECS.items()):
        
        # --- Columna de Carga (Arriba) ---
        with cols_upload[i]:
            st.caption(f"Cargar {spec['type'].upper()} ({spec['label']})")
            uploaded_file = st.file_uploader(
                f"Subir Archivo para {btn_label}",
                type=[spec['type']],
                key=f"upload_{spec['target_key']}",
                label_visibility="collapsed"
            )
            
        # --- Columna de Acción (Abajo) ---
        with cols_action[i]:
            st.caption(f"Acción: {btn_label}")
            if st.button(
                f"Ver {btn_label.replace('Graficar ', '')}", 
                key=f"btn_{spec['target_key']}", 
                use_container_width=True
            ):
                if uploaded_file is not None:
                    with st.spinner(f"Cargando y procesando {uploaded_file.name}..."):
                        # Usar la función auxiliar para leer el buffer en un DataFrame
                        df_loaded = _load_file_to_dataframe(uploaded_file, spec['type'])
                        
                        if df_loaded is not None and not df_loaded.empty:
                            # Almacenar el DataFrame limpio en la sesión para que el dashboard lo use
                            st.session_state[spec['target_key']] = df_loaded
                            st.success(f"✅ Datos para '{btn_label.replace('Graficar ', '')}' cargados y listos.")
                            st.session_state['should_rerun_viz'] = True
                        else:
                            st.error("❌ Falló la lectura del DataFrame. Revise el contenido.")
                else:
                    st.warning("Por favor, sube un archivo antes de graficar.")
                    
    if st.session_state['should_rerun_viz']:
        st.session_state['should_rerun_viz'] = False
        st.rerun() 
        
    st.markdown("---")