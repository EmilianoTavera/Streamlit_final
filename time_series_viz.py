import streamlit as st
import altair as alt
import pandas as pd


def render_time_series(df_default):
    """
    Renderiza la evolución temporal de delitos.
    Prioriza los datos cargados por el administrador (viz_serie_temporal_df).
    """
    # Leer datos desde st.session_state si están disponibles
    df = st.session_state.get('viz_serie_temporal_df', df_default)

    if df is None or df.empty:
        st.warning("No hay datos de delitos disponibles.")
        return

    # Asegurar que 'fecha' es datetime para el filtrado
    if pd.api.types.is_datetime64_any_dtype(df['fecha']) == False:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df.dropna(subset=['fecha'], inplace=True)
    
    # --- 1. FILTROS ---
    # Usamos 3 columnas para que los filtros queden alineados horizontalmente
    c1, c2, c3 = st.columns([1, 1, 0.8])

    with c1:
        # Filtro de Alcaldía
        all_alcaldias = sorted(df['alcaldia'].unique())
        # Evitar errores si no hay alcaldías o si las preseleccionadas ya no existen
        default_alcaldia = [all_alcaldias[0]] if all_alcaldias else []
        selected_alcaldias = st.multiselect(
            "Filtrar Alcaldía:",
            options=all_alcaldias,
            default=default_alcaldia,
            key="ts_alcaldia_multiselect"
        )

    with c2:
        # Filtro de Fechas (Rango)
        min_date = df['fecha'].min().date()
        max_date = df['fecha'].max().date()

        fechas_selec = st.date_input(
            "Rango de Fechas:",
            value=(min_date, max_date) if min_date != max_date else (min_date, min_date),
            min_value=min_date,
            max_value=max_date,
            key="ts_date_input"
        )

    with c3:
        # Agrupación temporal
        granularity = st.selectbox(
            "Agrupar por:",
            ["Día", "Mes", "Año"],
            index=1,
            key="ts_granularity_select"
        )

    # --- 2. APLICAR FILTROS ---
    df_filtered = df.copy()

    # Filtro de Fechas
    # Verificamos que sea una tupla de 2 valores (Inicio, Fin) para evitar errores mientras el usuario selecciona
    if isinstance(fechas_selec, tuple) and len(fechas_selec) == 2:
        start_date, end_date = fechas_selec
        # Convertimos la columna fecha a .date para comparar
        mask_date = (df_filtered['fecha'].dt.date >= start_date) & (df_filtered['fecha'].dt.date <= end_date)
        df_filtered = df_filtered[mask_date]

    # Filtro de Alcaldías
    if selected_alcaldias:
        df_filtered = df_filtered[df_filtered['alcaldia'].isin(selected_alcaldias)]

    # --- 3. GRÁFICA ALTAIR ---
    if not df_filtered.empty:
        # Configuración dinámica del Eje X según la agrupación
        if granularity == "Mes":
            x_val = 'yearmonth(fecha):T'
            tooltip_x = 'yearmonth(fecha)'
            x_title = "Fecha (Mes)"
        elif granularity == "Año":
            x_val = 'year(fecha):O'
            tooltip_x = 'year(fecha)'
            x_title = "Año"
        else:  # Día
            x_val = 'fecha:T'
            tooltip_x = 'fecha'
            x_title = "Fecha (Día)"

        chart = alt.Chart(df_filtered).mark_line(point=True).encode(
            x=alt.X(x_val, title=x_title),
            y=alt.Y('sum(conteo):Q', title='Total Delitos'),
            color=alt.Color('alcaldia:N', legend=alt.Legend(title="Alcaldía", orient='bottom')),
            tooltip=[tooltip_x, 'alcaldia', alt.Tooltip('sum(conteo)', title="Total")]
        ).properties(
            # Aumentamos height a 450 para igualar el tamaño visual del Cluster 3D
            height=450
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No hay datos para los filtros seleccionados.")