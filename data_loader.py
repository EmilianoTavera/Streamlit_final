# data_loader.py

import pandas as pd
import streamlit as st
import geopandas as gpd
import zipfile
import io

## Funciones de Carga de Datos

@st.cache_data(show_spinner=False)
def load_gauge_data(path="resultados_sentiment.csv"):#
    """Carga los datos para los medidores."""
    try:
        # 1️ Leer el CSV
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error al cargar df_Gauge.csv: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_cluster_data(path="resultados_cluster.csv"):#
    """Carga los datos para el cluster 3D y el boxplot."""
    try:
        # 1️ Leer el CSV
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error al cargar df_Cluster_3D.csv: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_map_data():
    """Carga los datos de incidentes y los límites GeoJSON de la CDMX.
    MODIFICADO: Usa el archivo local 'limite-de-las-alcaldias.geojson' en lugar de la URL externa para evitar errores de conexión.
    """
    
    # 1️ Cargar Datos de incidentes (ASUME que la variable 'df' se carga aquí)
    # *** REEMPLAZA ESTA LÍNEA CON TU LÓGICA DE CARGA DE DATOS DE INCIDENTES (si no lo hace otra función) ***
    df = None 
    
    # 2️ Cargar GeoJSON desde el repositorio local
    # IMPORTANTE: Ya que tu archivo está en el repositorio, lo cargaremos directamente. 
    # Asegúrate de usar el nombre correcto. Lo cambié a .geojson para compatibilidad, 
    # pero si es .json úsalo.
    RUTA_GEOJSON_LOCAL = "limite-de-las-alcaldias.geojson" 
    
    try:
        # Cargar el GeoJSON directamente desde el archivo local
        gdf = gpd.read_file(RUTA_GEOJSON_LOCAL)
        
    except Exception as e:
        # Capturar errores si el archivo local no se encuentra (FileNotFoundError)
        st.error(f"Error al cargar el GeoJSON local '{RUTA_GEOJSON_LOCAL}'. Asegúrate de que el archivo exista en la raíz del repositorio. Detalle: {e}")
        return df, None # Retorna None para el GDF
        
    # Retorna tus datos de incidentes (df) y los límites geográficos (gdf)
    return df, gdf


@st.cache_data(show_spinner=False)
def load_serie_temporal_delitos(zip_path="serie_temporal_delitos.zip", csv_filename="serie_temporal_delitos.csv"):#
    """Carga los datos para las series temporal delitos desde un archivo ZIP."""
    df = pd.DataFrame()
    try:
        # 1️ Abrir y leer el archivo CSV dentro del ZIP
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Leer el contenido del CSV en memoria
            with z.open(csv_filename) as f:
                df = pd.read_csv(io.BytesIO(f.read()))

        # 2️ Convertir columna 'fecha' a datetime
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'])

        return df
    except FileNotFoundError:
        st.error(f"Error: El archivo ZIP '{zip_path}' no se encontró.")
        return pd.DataFrame()
    except KeyError:
        st.error(f"Error: El archivo CSV '{csv_filename}' no se encontró dentro del ZIP.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar o descomprimir df_serie_temporal_delitos.zip: {e}")
        return pd.DataFrame()

# Rutas de archivos (Ajusta si están en subcarpetas)
FILE_PRED_CSV = "mapa_predicciones_24h.csv"
URL_GEOJSON = "https://datos.cdmx.gob.mx/dataset/bae265a8-d1f6-4614-b399-4184bc93e027/resource/deb5c583-84e2-4e07-a706-1b3a0dbc99b0/download/limite-de-las-alcaldas.json"


@st.cache_data
def load_predict():
    """
    Carga el dataset de predicciones y el GeoJSON.
    """
    try:
        # 1. Cargar GeoJSON de Alcaldías
        gdf = gpd.read_file(URL_GEOJSON)

        # 2. Cargar CSV de Predicciones
        # Si no tienes el archivo real, crea un DataFrame vacío para que no rompa la app
        try:
            df = pd.read_csv(FILE_PRED_CSV)
        except FileNotFoundError:
            # Crea un dummy para evitar errores si falta el archivo
            return None, gdf

        return df, gdf

    except Exception as e:
        st.error(f"Error al cargar datos de predicción: {e}")
        return None, None
