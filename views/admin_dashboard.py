# views/admin_dashboard.py
import streamlit as st
import sys
sys.path.append('.') # Permite importar common.py
from common import renderizar_graficos_dashboard

def render():
    renderizar_graficos_dashboard(titulo="Tendencias de actividad (Admin)")