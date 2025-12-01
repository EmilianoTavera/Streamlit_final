# views/admin_configuracion.py

import streamlit as st
import pandas as pd
import time

def render():
    st.title("Configuración de Usuarios")
    st.markdown("---")

    # 1. Mostrar lista de usuarios
    st.subheader("Usuarios Existentes")
    
    # Prepara los datos para mostrar en un DataFrame
    user_data = []
    # Accedemos a la versión mutable del diccionario de usuarios
    current_users_dict = st.session_state['users']
    
    for username, details in current_users_dict.items():
        user_data.append({
            "Usuario": username,
            "Contraseña": details["password"],
            "Rol": details["role"]
        })
        
    df = pd.DataFrame(user_data)
    # Muestra la tabla de usuarios
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")

    col_add, col_delete = st.columns(2)

    # --- COLUMNA 1: AGREGAR NUEVO USUARIO ---
    with col_add:
        st.subheader(" Agregar Nuevo Usuario")
        with st.form(key='add_user_form', clear_on_submit=True, border=True):
            new_username = st.text_input("Nuevo Nombre de Usuario")
            new_password = st.text_input("Nueva Contraseña", type="password")
            # Forzamos roles para evitar errores
            new_role = st.selectbox("Rol", ["user", "admin"])
            submit_button = st.form_submit_button(label='Añadir Usuario', use_container_width=True)

            if submit_button:
                if new_username and new_password:
                    if new_username in current_users_dict:
                        st.error(f"El usuario '{new_username}' ya existe.")
                    else:
                        # Actualizar el diccionario en st.session_state
                        st.session_state['users'][new_username] = {
                            "password": new_password,
                            "role": new_role
                        }
                        st.success(f"Usuario '{new_username}' ({new_role}) agregado con éxito.")
                        time.sleep(0.5) 
                        st.rerun()
                else:
                    st.warning("Por favor, rellena el nombre de usuario y la contraseña.")

    # --- COLUMNA 2: ELIMINAR USUARIO EXISTENTE ---
    with col_delete:
        st.subheader("Eliminar Usuario")
        
        # Obtenemos la lista de usuarios para el selectbox
        current_users_list = list(current_users_dict.keys())
        
        if len(current_users_list) > 1:
            user_to_delete = st.selectbox(
                "Selecciona el usuario a eliminar",
                options=current_users_list,
                key="delete_user_select"
            )
            
            delete_button = st.button("Eliminar Usuario Seleccionado", key="delete_user_button", use_container_width=True)
            
            if delete_button and user_to_delete:
                # Comprobación de seguridad: no permitir eliminar al usuario actualmente loggeado
                if user_to_delete == st.session_state['username']: 
                    st.error("No puedes eliminar al usuario actualmente loggeado.")
                else:
                    # Eliminar del diccionario en st.session_state
                    del st.session_state['users'][user_to_delete]
                    st.success(f"Usuario '{user_to_delete}' eliminado con éxito.")
                    time.sleep(0.5)
                    st.rerun()
        else:
            st.warning("Debe haber al menos un usuario en el sistema. No se puede eliminar.")
