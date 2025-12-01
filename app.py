import streamlit as st
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import altair as alt
import pathlib 

# ⚠️ IMPORTANTE: Añadir la ruta para módulos, necesario para importar common, data_loader y views
sys.path.append('.') 

# --- Importar Lógica y Vistas ---
from common import cargar_estilos_globales # Asegúrate de importar la función
from common import USERS, logout
from common import create_gauge
from data_loader import load_gauge_data, load_cluster_data, load_map_data, load_serie_temporal_delitos, load_predict
from clusters_viz import mostrar_medidores, mostrar_cluster_3d, mostrar_boxplot_y_stats
from views import admin_entrenamiento, admin_configuracion, user_entrenamiento
from views import admin_chatbot # 🆕 IMPORTAR NUEVA VISTA CHATBOT
from map_viz import render_mapa_contexto
from time_series_viz import render_time_series
from predict_viz import render_mapa_predicciones


# --- 1. Configuración de la Página y Estados ---
st.set_page_config(
    page_title="Dashboard de Actividad",
    layout="wide"
)

# INMEDIATAMENTE DESPUÉS, CARGAS LOS ESTILOS
cargar_estilos_globales()

# Inicializar estados de sesión
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_type' not in st.session_state:
    st.session_state['user_type'] = None
if 'admin_page' not in st.session_state:
    st.session_state['admin_page'] = 'dashboard' 
if 'user_page' not in st.session_state:
    st.session_state['user_page'] = 'dashboard_2'

# AÑADIDO: Inicializar el estado mutable de usuarios y el nombre del usuario
if 'users' not in st.session_state:
    st.session_state['users'] = USERS.copy() # Copia del diccionario inicial
if 'username' not in st.session_state:
    st.session_state['username'] = None

# --- Cargar Datos Globalmente (Usa caché de Streamlit) ---
# Se mantiene la carga aquí ya que los DataFrames son necesarios para las visualizaciones por defecto
Gauge_data = load_gauge_data()
Cluster_data = load_cluster_data()
Map_data, Deleg_data= load_map_data()
TimeSeries_data = load_serie_temporal_delitos()
Pred_Data, Pred_GDF = load_predict()


# --- 2. Funciones de Autenticación ---
def login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Iniciamos el contenedor para centrar el texto
        st.markdown('<div class="login-container">', unsafe_allow_html=True) 

        # Título Centrado
        st.markdown(
            '<h2 style="text-align: center; font-size: 36px;" >Iniciar Sesión</h2>',
            unsafe_allow_html=True
        )

        # Descripción Centrada
        st.markdown('<p style="text-align: center; font-size: 22px;" >Por favor ingresa tus credenciales para acceder al sistema.</p>', unsafe_allow_html=True)

        # Cerramos el contenedor.
        st.markdown('</div>', unsafe_allow_html=True) 

        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")

        if st.button("Entrar", use_container_width=True):
            # AUTENTICACIÓN: Usamos la lista mutable en st.session_state
            if username in st.session_state['users'] and password == st.session_state['users'][username]["password"]:
                st.session_state['logged_in'] = True
                st.session_state['admin_page'] = 'dashboard' 
                st.session_state['user_page'] = 'dashboard_2' 
                st.session_state['user_type'] = st.session_state['users'][username]["role"]
                st.session_state['username'] = username # Guardamos el nombre del usuario
                st.rerun()
            else:
                st.error("Credenciales incorrectas")

        # Código de la imagen de logos_cdmx.png que tenías al final de login
        try:
            ruta_base = pathlib.Path(__file__).parent
            ruta_imagen = ruta_base / "logos_cdmx.png"
            st.image(
                str(ruta_imagen),
                use_container_width=True, 
                caption="Instituciones de la Ciudad de México"
            )
        except Exception as e:
            st.error(f"Error al cargar la imagen: Asegúrate de que 'logos_cdmx.png' esté en la ruta correcta. {e}")

def render_dashboard_content(gauge_df, cluster_df, ts_df, pred_df, pred_gdf, role):

    # --- Título y Fecha ---
    st.title(f"Tendencias de actividad ({role.capitalize()})")
    st.markdown("---")

    # Definimos dos columnas principales con una proporción 
    # [1, 1.5] o [1, 2] para que el mapa tenga más protagonismo visual
    col_analisis, col_mapa = st.columns([0.8,1.5], gap="small")

    # --- COLUMNA IZQUIERDA: Todo el análisis estadístico y clusters ---
    with col_analisis:

        # 1. Medidores (Gauges)
        # Nota: Al estar en una columna más angosta, sugeriría cambiar 
        # 'columnas_por_fila' a 2 para que se vean bien (2 arriba, 2 abajo).
        with st.container(border=True):
            st.caption("Sentimiento General")
            mostrar_medidores(gauge_df, columnas_por_fila=2) 

        # 2. Gráfico 3D
        with st.container(border=True):
            st.caption("Distribución 3D")
            mostrar_cluster_3d(cluster_df)

        # 3. Boxplot y Resumen Estadístico (Plegable)
        with st.container(border=True):
            st.caption("Variabilidad y Métricas")
            mostrar_boxplot_y_stats(cluster_df)

    # --- COLUMNA DERECHA: Mapa ---
    with col_mapa:
        # Usamos un contenedor con altura fija (height) para forzar que el mapa 
        # ocupe todo el espacio vertical disponible y no se vea pequeño.
        # Ajusta el valor 850 según la cantidad de datos en tu columna izquierda.
        with st.container(border=True, height=1010):
            st.subheader("Incidencias por alcaldía")
            # Asegúrate de que tu función de mapa tenga use_container_width=True internamente
            render_mapa_contexto(Map_data, Deleg_data)
        with st.container(border=True):
            st.caption("Evolución Temporal de Delitos")
            render_time_series(ts_df)

        with st.container(border=True):
            st.subheader("Mapa de Predicciones (Próximas 24h)")
            render_mapa_predicciones(pred_df, pred_gdf)

# --- 4. Componentes de Navegación (Menú Superior) ---

def mostrar_dashboard_admin_con_menu(gauge_df, cluster_df, ts_df, pred_df, pred_gdf):
    
    # 🆕 AGREGAR UNA COLUMNA MÁS PARA EL BOTÓN CHATBOT
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 1])

    with nav_col1:
        if st.button("Dashboard", key="nav_home_admin", use_container_width=True):
            st.session_state['admin_page'] = 'dashboard'
            st.rerun()
    with nav_col2:
        if st.button("Entrenamiento", key="nav_train_admin", use_container_width=True):
            st.session_state['admin_page'] = 'entrenamiento'
            st.rerun()
    with nav_col3:
        if st.button("Configuraciones", key="nav_conf_admin", use_container_width=True):
            st.session_state['admin_page'] = 'configuracion'
            st.rerun()
    # 🆕 NUEVO BOTÓN CHATBOT
    with nav_col4:
        if st.button("Chatbot", key="nav_chat_admin", use_container_width=True):
            st.session_state['admin_page'] = 'chatbot'
            st.rerun()


    st.markdown("---")

    # Router Admin
    if st.session_state['admin_page'] == 'entrenamiento':
        admin_entrenamiento.render()
    elif st.session_state['admin_page'] == 'configuracion':
        admin_configuracion.render()
    # 🆕 NUEVO ENRUTAMIENTO CHATBOT
    elif st.session_state['admin_page'] == 'chatbot':
        admin_chatbot.render()
    elif st.session_state['admin_page'] == 'dashboard':
        render_dashboard_content(gauge_df, cluster_df, ts_df, pred_df, pred_gdf, role="admin")

def mostrar_dashboard_user_con_menu(gauge_df, cluster_df, ts_df, pred_df, pred_gdf):
    # NOTA: Mantengo la navegación de usuario sin el botón de Chatbot y Configuración, 
    # asumiendo que solo el admin lo usará. Si el usuario también lo necesita, házmelo saber.

    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])

    with nav_col1:
        if st.button("Dashboard", key="nav_home_user", use_container_width=True):
            st.session_state['user_page'] = 'dashboard_2'
            st.rerun()
    with nav_col2:
        if st.button("Entrenamiento", key="nav_train_user", use_container_width=True):
            st.session_state['user_page'] = 'entrenamiento_2'
            st.rerun()
    with nav_col3:
        st.empty() # Espacio vacío

    st.markdown("---")

    # Router Usuario
    if st.session_state['user_page'] == 'entrenamiento_2':
        user_entrenamiento.render()
    elif st.session_state['user_page'] == 'dashboard_2':
        render_dashboard_content(gauge_df,cluster_df, ts_df, pred_df, pred_gdf, role="usuario")


# --- 5. ROUTER PRINCIPAL ---

if not st.session_state['logged_in']:
    login()
else:
    # Renderizar el menú y el contenido específico del rol
    if st.session_state['user_type'] == "admin":
        mostrar_dashboard_admin_con_menu(Gauge_data, Cluster_data, TimeSeries_data, Pred_Data, Pred_GDF)
    elif st.session_state['user_type'] == "user":
        mostrar_dashboard_user_con_menu(Gauge_data, Cluster_data, TimeSeries_data, Pred_Data, Pred_GDF)
    else:
        st.error("Error: Tipo de usuario desconocido")

    logout()