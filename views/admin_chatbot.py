import streamlit as st
import sys
sys.path.append('.') 
from chatbot_module import render_chatbot_view # Importamos la función de renderizado

def render():
    st.title(" Asistente de IA")
    st.markdown("---")
    
    # Aquí se renderiza la lógica del chatbot
    render_chatbot_view()