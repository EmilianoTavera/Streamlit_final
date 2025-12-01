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
def load_map_data(zip_path="mapa_procesado.zip", csv_filename="mapa_procesado.csv"):#
    """
    Carga los datos del mapa desde un archivo ZIP.
    Asume que el archivo CSV está dentro del ZIP.
    """
    df = pd.DataFrame()

    try:
        # 1️ Abrir y leer el archivo CSV dentro del ZIP
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Leer el contenido del CSV en memoria
            with z.open(csv_filename) as f:
                # Usar io.BytesIO para que pd.read_csv pueda leer el archivo directamente
                df = pd.read_csv(io.BytesIO(f.read()))

    except FileNotFoundError:
        st.error(f"Error: El archivo ZIP '{zip_path}' no se encontró.")
        return pd.DataFrame(), None
    except KeyError:
        st.error(f"Error: El archivo CSV '{csv_filename}' no se encontró dentro del ZIP.")
        return pd.DataFrame(), None
    except Exception as e:
        st.error(f"Error al cargar o descomprimir los datos del mapa: {e}")
        return pd.DataFrame(), None

    # Limpieza básica de columnas
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Asegurar binarios
    for col in ['trabajo', 'centro', 'horario_laboral']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # 2️ Cargar GeoJSON
    RUTA_GEOJSON_LOCAL = "limite-de-las-alcaldias.json"
    url_geojson = "https://datos.cdmx.gob.mx/dataset/bae265a8-d1f6-4614-b399-4184bc93e027/resource/deb5c583-84e2-4e07-a706-1b3a0dbc99b0/download/limite-de-las-alcaldas.json"
    gdf = gpd.read_file(RUTA_GEOJSON_LOCAL)

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
        RUTA_GEOJSON_LOCAL = "limite-de-las-alcaldias.json"
        gdf = gpd.read_file(RUTA_GEOJSON_LOCAL)

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
