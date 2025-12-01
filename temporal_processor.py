#temporal_processor.py
import pandas as pd
import numpy as np
import io
import zipfile
import re
import unicodedata
import warnings
import datetime
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

warnings.filterwarnings('ignore')

# ----------------------------------------------------
# 0. CONSTANTES Y MAPEOS
# ----------------------------------------------------

# Lista de delitos a mantener (tomada de tu código original)
DELITOS_A_MANTENER = ["DANO EN PROPIEDAD AJENA CULPOSA", "DANO EN PROPIEDAD AJENA CULPOSA POR TRANSITO VEHICULAR A AUTOMOVIL", "DANO EN PROPIEDAD AJENA CULPOSA POR TRANSITO VEHICULAR A BIENES INMUEBLES", "DANO EN PROPIEDAD AJENA CULPOSA POR TRANSITO VEHICULAR A VIAS DE COMUNICACION", "DANO EN PROPIEDAD AJENA INTENCIONAL", "DANO EN PROPIEDAD AJENA INTENCIONAL A AUTOMOVIL", "DANO EN PROPIEDAD AJENA INTENCIONAL A BIENES INMUEBLES", "DANO EN PROPIEDAD AJENA INTENCIONAL A CASA HABITACION", "DANO EN PROPIEDAD AJENA INTENCIONAL A NEGOCIO", "DANO EN PROPIEDAD AJENA INTENCIONAL A VIAS DE COMUNICACION", "DANO SUELO ACTIVIDAD INVASION O EXTRACCION", "HOMICIDIO CULPOSO", "HOMICIDIO CULPOSO CON EXCLUYENTES DE RESPONSABILIDAD", "HOMICIDIO CULPOSO CON EXCLUYENTES DE RESPONSABILIDAD", "HOMICIDIO CULPOSO FUERA DEL DF ATROPELLADO", "HOMICIDIO CULPOSO FUERA DEL DF COLISION", "HOMICIDIO CULPOSO ", "POR ARMA DE FUEGO", "HOMICIDIO CULPOSO POR INSTRUMENTO PUNZO CORTANTE", "HOMICIDIO CULPOSO POR TRANSITO VEHICULAR", "HOMICIDIO CULPOSO POR TRANSITO VEHICULAR ATROPELLADO", "HOMICIDIO CULPOSO POR TRANSITO VEHICULAR CAIDA", "HOMICIDIO CULPOSO POR TRANSITO VEHICULAR COLISION", "HOMICIDIO DOLOSO", "HOMICIDIO POR AHORCAMIENTO", "HOMICIDIO POR ARMA BLANCA", "HOMICIDIO POR ARMA DE FUEGO", "HOMICIDIO POR GOLPES", "HOMICIDIO POR INMERSION", "HOMICIDIOS INTENCIONALES OTROS", "ROBO A CASA HABITACION CON VIOLENCIA", "ROBO A CASA HABITACION SIN VIOLENCIA", "ROBO A CASA HABITACION Y VEHICULO CON VIOLENCIA", "ROBO A CASA HABITACION Y VEHICULO SIN VIOLENCIA", "ROBO A LOCALES SEMIFIJOS PUESTOS DE ALIMENTOSBEBIDAS ENSERES PERIODICOSLOTERIA OTROS", "ROBO A NEGOCIO CON VIOLENCIA", "ROBO A NEGOCIO CON VIOLENCIA POR FARDEROS TIENDAS DE AUTOSERVICIO", "ROBO A NEGOCIO CON VIOLENCIA POR FARDEROS TIENDAS DE CONVENIENCIA", "ROBO A ", "NEGOCIO NOMINA Y VEHICULO CON VIOLENCIA", "ROBO A NEGOCIO SIN VIOLENCIA", "ROBO A NEGOCIO SIN VIOLENCIA POR FARDEROS", "ROBO A NEGOCIO SIN VIOLENCIA POR FARDEROS TIENDAS DE AUTOSERVICIO", "ROBO A NEGOCIO SIN VIOLENCIA POR FARDEROS TIENDAS DE CONVENIENCIA", "ROBO A NEGOCIO Y VEHICULO CON VIOLENCIA", "ROBO A NEGOCIO Y VEHICULO SIN VIOLENCIA", "ROBO A OFICINA PUBLICA CON VIOLENCIA", "ROBO A OFICINA PUBLICA SIN VIOLENCIA", "ROBO A PASAJERO CONDUCTOR DE TAXI CON VIOLENCIA", "ROBO A PASAJERO CONDUCTOR DE VEHICULO CON VIOLENCIA", "ROBO A PASAJERO A BORDO DE CABLEBUS CON VIOLENCIA", "ROBO A PASAJERO A BORDO DE CABLEBUS SIN VIOLENCIA", "ROBO A PASAJERO A BORDO DE METRO CON VIOLENCIA", "ROBO A PASAJERO A BORDO ", "DE METRO SIN VIOLENCIA", "ROBO A PASAJERO A BORDO DE METROBUS CON VIOLENCIA", "ROBO A PASAJERO A BORDO DE METROBUS SIN VIOLENCIA", "ROBO A PASAJERO A BORDO DE PESERO COLECTIVO CON VIOLENCIA", "ROBO A PASAJERO A BORDO DE PESERO COLECTIVO SIN VIOLENCIA", "ROBO A PASAJERO A BORDO DE PESERO Y VEHICULO CON VIOLENCIA", "ROBO A PASAJERO A BORDO DE TAXI CON VIOLENCIA", "ROBO A PASAJERO A BORDO DE TAXI SIN VIOLENCIA", "ROBO A PASAJERO A BORDO DE TRANSPORTE PUBLICO CON VIOLENCIA", "ROBO A PASAJERO A BORDO DE TRANSPORTE PUBLICO SIN VIOLENCIA", "ROBO A PASAJERO EN AUTOBUS FORANEO CON VIOLENCIA", "ROBO A PASAJERO EN AUTOBUS FORANEO SIN VIOLENCIA", "ROBO A PASAJERO EN ECOBUS ", "CON VIOLENCIA", "ROBO A PASAJERO EN ECOBUS SIN VIOLENCIA", "ROBO A PASAJERO EN RTP CON VIOLENCIA", "ROBO A PASAJERO EN RTP SIN VIOLENCIA", "ROBO A PASAJERO EN TREN LIGERO CON VIOLENCIA", "ROBO A PASAJERO EN TREN LIGERO SIN VIOLENCIA", "ROBO A PASAJERO EN TREN SUBURBANO CON VIOLENCIA", "ROBO A PASAJERO EN TREN SUBURBANO SIN VIOLENCIA", "ROBO A PASAJERO EN TROLEBUS CON VIOLENCIA", "ROBO A PASAJERO EN TROLEBUS SIN VIOLENCIA", "ROBO A REPARTIDOR CON VIOLENCIA", "ROBO A REPARTIDOR SIN VIOLencia", "ROBO A REPARTIDOR Y VEHICULO CON VIOLENCIA", "ROBO A REPARTIDOR Y VEHICULO SIN VIOLENCIA", "ROBO A SUCURSAL BANCARIA ASALTO BANCARIO CON VIOLENCIA", "ROBO A SUCURSAL BANCARIA CON VIOLENCIA", "ROBO A SUCURSAL BANCARIA DENTRO DE TIENDAS DE ", "AUTOSERVICIO CON VIOLENCIA", "ROBO A SUCURSAL BANCARIA DENTRO DE TIENDAS DE AUTOSERVICIO SV", "ROBO A SUCURSAL BANCARIA SIN VIOLENCIA", "ROBO A SUCURSAL BANCARIA SUPERMERCADO CON VIOLENCIA", "ROBO A SUCURSAL BANCARIA SUPERMERCADO SIN VIOLENCIA", "ROBO A TRANSEUNTE A BORDO DE TAXI PUBLICO Y PRIVADO CON VIOLENCIA", "ROBO A TRANSEUNTE A BORDO DE TAXI PUBLICO Y PRIVADO SIN VIOLENCIA", "ROBO A TRANSEUNTE CONDUCTOR DE TAXI PUBLICO Y PRIVADO CON VIOLENCIA", "ROBO A TRANSEUNTE DE CELULAR CON VIOLENCIA", "ROBO A TRANSEUNTE DE CELULAR SIN VIOLENCIA", "ROBO A TRANSEUNTE EN CINE CON VIOLENCIA", "ROBO A TRANSEUNTE EN HOTEL CON VIOLENCIA", "ROBO A TRANSEUNTE EN NEGOCIO CON VIOLENCIA", "ROBO A TRANSEUNTE EN PARQUES Y MERCADOS CON VIOLENCIA", "ROBO ", "A TRANSEUNTE EN RESTAURANT CON VIOLENCIA", "ROBO A TRANSEUNTE EN TERMINAL DE PASAJEROS CON VIOLENCIA", "ROBO A TRANSEUNTE EN VIA PUBLICA CON VIOLENCIA", "ROBO A TRANSEUNte EN VIA PUBLICA NOMINA CON VIOLENCIA", "ROBO A TRANSEUNTE EN VIA PUBLICA NOMINA SIN VIOLENCIA", "ROBO A TRANSEUNTE EN VIA PUBLICA SIN VIOLENCIA", "ROBO A TRANSEUNTE SALIENDO DEL BANCO CON VIOLENCIA", "ROBO A TRANSEUNTE SALIENDO DEL CAJERO CON VIOLENCIA", "ROBO A TRANSEUNTE Y VEHICULO CON VIOLENCIA", "ROBO A TRANSPORTISTA Y VEHICULO PESADO CON VIOLENCIA", "ROBO A TRANSPORTISTA Y VEHICULO PESADO SIN VIOLENCIA", "ROBO DE ACCESORIOS DE AUTO", "ROBO DE ALHAJAS", "ROBO DE ANIMALES", "ROBO DE ARMA", "ROBO DE CONTENEDORES DE TRAILERS SV", "ROBO DE DINERO", "ROBO DE DOCUMENTOS", "ROBO DE FLUIDOS", "ROBO DE ", "INFANTE", "ROBO DE MAQUINARIA CON VIOLENCIA", "ROBO DE MAQUINARIA SIN VIOLENCIA", "ROBO DE MERCANCIA A TRANSPORTISTA CV", "ROBO DE MERCANCIA EN CONTENEDEROS EN AREAS FEDERALES", "ROBO DE MERCANCIA EN CONTENEDEROS EN AREAS FEDERALES", "ROBO DE MOTOCICLETA CON VIOLENCIA", "ROBO DE MOTOCICLETA SIN VIOLENCIA", "ROBO DE OBJETOS", "ROBO DE OBJETOS A ESCUELA", "ROBO DE OBJETOS DEL INTERIOR DE UN VEHICULO", "ROBO DE PLACA DE AUTOMOVIL", "ROBO DE VEHICULO DE PEDALES", "ROBO DE VEHICULO DE SERVICIO DE TRANSPORTE CON VIOLENCIA", "ROBO DE VEHICULO DE SERVICIO DE TRANSPORTE SIN VIOLENCIA", "ROBO DE VEHICULO DE SERVICIO OFICIAL CON VIOLENCIA", "ROBO DE VEHICULO DE SERVICIO OFICIAL SIN VIOLENCIA", "ROBO DE VEHICULO DE SERVICIO PARTICULAR CON VIOLENCIA", "ROBO DE VEHICULO DE SERVICIO PARTICULAR SIN ", "VIOLENCIA", "ROBO DE VEHICULO DE SERVICIO PUBLICO CON VIOLENCIA", "ROBO DE VEHICULO DE SERVICIO PUBLICO SIN VIOLENCIA", "ROBO DE VEHICULO ELECTRICO MOTOPATIN", "ROBO DE VEHICULO EN PENSION TALLER Y AGENCIAS CV", "ROBO DE VEHICULO EN PENSION TALLER Y AGENCIAS SV", "ROBO DE VEHICULO Y NOMINA CON VIOLENCIA", "ROBO DURANTE TRASLADO DE VALORES NOMINA CON VIOLENCIA", "ROBO DURANTE TRASLADO DE VALORES NOMINA SIN VIOLENCIA", "ROBO EN EVENTOS MASIVOS DEPORTIVOS CULTURALES RELIGIOSOS Y ARTISTICOS SV", "ROBO EN INTERIOR DE EMPRESA NOMINA CON VIOLENCIA", "ROBO EN INTERIOR DE EMPRESA NOMINA SIN VIOLENCIA", "ROBO SV DENTRO DE NEGOCIOS AUTOSERVICIOS CONVENIENCIA", "SECUESTRO", "SECUESTRO EXPRESS PARA COMETER ROBO O EXTORSION", "SECUESTRO EXTORSIVO", "VIOLACION"]

# Mapeo de meses (Tomado de tu código)
MONTH_MAP = {
    'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo',
    'April': 'Abril', 'May': 'Mayo', 'June': 'Junio',
    'July': 'Julio', 'August': 'Agosto', 'September': 'Septiembre',
    'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'
}

# ----------------------------------------------------
# 1. FUNCIONES AUXILIARES DE ARCHIVOS Y UTILIDADES
# ----------------------------------------------------

def _create_output_zip(df, filename):
    """
    Convierte un DataFrame en un archivo ZIP con un CSV dentro, listo para Streamlit.
    """
    if df is None or df.empty:
        return None
    
    # Convertir DataFrame a CSV en memoria
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8', float_format='%.4f')
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    
    # Crear el archivo ZIP en memoria
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, csv_bytes)
        
    return zip_buffer.getvalue()

def _create_output_csv_buffer(df):
    """
    Convierte un DataFrame directamente a un buffer de bytes CSV (no ZIP).
    """
    if df is None or df.empty:
        return None
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8', float_format='%.4f')
    return io.BytesIO(csv_buffer.getvalue().encode('utf-8')).getvalue()

def _calcular_metricas(y_true, y_pred, nombre_modelo="Modelo"):
    """Calcula métricas clave (RMSE, MAE, R2)."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {'Modelo': nombre_modelo, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
        
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    metricas = {
        'Modelo': nombre_modelo,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
    }
    return metricas

def _normalize_text(text):
    """Función de normalización de texto generalizada (quitando acentos, mayúsculas)."""
    if isinstance(text, str):
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        text = text.upper()
        text = re.sub(r'[^A-Z0-9\s]', '', text)
    return text

# ----------------------------------------------------
# 2. FUNCIONES DE LIMPIEZA Y FEATURE ENGINEERING
# ----------------------------------------------------

def _load_and_clean_data(df_raw):
    """
    Aplica la limpieza y normalización completa a los datos crudos.
    """
    df = df_raw.copy()

    # --- 1. Conversión de Tipos y Normalización de Meses ---
    if 'mes_inicio' in df.columns:
        df['mes_inicio'] = df['mes_inicio'].replace(MONTH_MAP)
    if 'mes_hecho' in df.columns:
        df['mes_hecho'] = df['mes_hecho'].replace(MONTH_MAP)

    if 'fecha_hecho' in df.columns:
        df['fecha_hecho'] = pd.to_datetime(df['fecha_hecho'], errors='coerce')
    
    # Normalización de columnas de tiempo
    if 'hora_hecho' in df.columns:
        df["hora_hecho"] = pd.to_datetime(df["hora_hecho"].astype(str).str.slice(0, 8), format="%H:%M:%S", errors="coerce").dt.time
    if 'hora_inicio' in df.columns:
        df["hora_inicio"] = pd.to_datetime(df["hora_inicio"].astype(str).str.slice(0, 8), format="%H:%M:%S", errors="coerce").dt.time
    
    # Normalización de columnas de texto
    cols_to_normalize = ['delito', 'fiscalia', 'agencia', 'unidad_investigacion', 'municipio_hecho', 'colonia_catalogo', 'categoria_delito', 'colonia_hecho', 'alcaldia_hecho']
    for col in cols_to_normalize:
        if col in df.columns:
            df[col] = df[col].apply(_normalize_text)

    # --- 2. Imputación y Filtrado de Tiempo ---
    
    # Imputación de hora_inicio
    if 'hora_inicio' in df.columns:
        mask_midnight = df['hora_inicio'].astype(str).str.slice(0, 8) == '00:00:00'
        df.loc[mask_midnight, 'hora_inicio'] = np.nan
        df.dropna(subset=['hora_inicio'], inplace=True)
    
    # Filtrado por rango de años
    if 'fecha_hecho' in df.columns:
        df = df[(df['fecha_hecho'].dt.year >= 2016) & (df['fecha_hecho'].dt.year <= 2024)]
    
    df.dropna(subset=['anio_hecho', 'fecha_hecho'], inplace=True)
    
    # Imputación de hora_hecho
    if 'hora_hecho' in df.columns:
        if 'fecha_hecho' in df.columns:
            df['hora_hecho'] = df['fecha_hecho'].dt.time
        mask_midnight_hecho = df['hora_hecho'].astype(str).str.slice(0, 8) == '00:00:00'
        df.loc[mask_midnight_hecho, 'hora_hecho'] = np.nan
        df['hora_hecho'] = df['hora_hecho'].fillna(df['hora_inicio'])
        
    df.dropna(subset=['fiscalia'], inplace=True)
    df.drop(columns=['competencia'], inplace=True, errors='ignore')

    # --- 3. Imputación Geográfica y Creación de Datetime ---

    valores_indeterminados = ['CDMX INDETERMINADA', 'CDMX (INDETERMINADA)', 'CDMX INDETERMINADA']
    if 'alcaldia_hecho' in df.columns and 'alcaldia_catalogo' in df.columns:
        df = df[~df['alcaldia_hecho'].isin(valores_indeterminados) & ~df['alcaldia_catalogo'].isin(valores_indeterminados)]
    
    if 'fiscalia' in df.columns and 'alcaldia_hecho' in df.columns:
        fiscalia_alcaldia_map = (
            df[df['alcaldia_hecho'].notna()]
            .groupby('fiscalia')['alcaldia_hecho']
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else None)
            .to_dict()
        )
        mask_nan = df['alcaldia_hecho'].isna()
        df.loc[mask_nan, 'alcaldia_hecho'] = df.loc[mask_nan, 'fiscalia'].map(fiscalia_alcaldia_map)
    
    df.dropna(subset=['colonia_hecho', 'latitud'], inplace=True)
    
    # CREACIÓN DE LA COLUMNA CRÍTICA 'datetime'
    if 'fecha_hecho' in df.columns and 'hora_hecho' in df.columns:
        df['datetime'] = pd.to_datetime(
            df['fecha_hecho'].astype(str) + ' ' + df['hora_hecho'].astype(str),
            errors='coerce'
        )
    else:
        # Esto debería fallar si no tiene las columnas correctas
        raise KeyError("Columnas fecha_hecho u hora_hecho faltantes para crear 'datetime'.")

    df = df.dropna(subset=['datetime'])

    # --- 4. Filtrado Final de Delitos ---
    # Esto replica la funcionalidad de 'datos_para_limpiar'
    df = df[df['delito'].isin(DELITOS_A_MANTENER)].copy()
    
    # Columnas finales requeridas para el pipeline de series de tiempo y mapa
    columns_final_req = ['datetime', 'alcaldia_hecho', 'delito', 'latitud', 'longitud']
    df = df[columns_final_req]

    return df

def _crear_features_temporales(df):
    """Crea las features temporales cíclicas y binarias."""
    df['anio'] = df['datetime'].dt.year
    df['mes'] = df['datetime'].dt.month
    df['dia'] = df['datetime'].dt.day
    df['dia_semana'] = df['datetime'].dt.dayofweek
    df['hora'] = df['datetime'].dt.hour
    
    # Features cíclicas (Sin y Cos)
    df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    
    # Features binarias
    df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)
    df['es_noche'] = ((df['hora'] >= 20) | (df['hora'] <= 6)).astype(int)
    
    return df

def _crear_serie_temporal_agregada(df, nivel='alcaldia_hecho', freq='24H'):
    """Agrega el conteo de delitos por día y por alcaldía."""
    df_temp = df.copy().dropna(subset=['datetime', nivel])
    df_temp = df_temp.sort_values('datetime')
    df_temp['periodo_24h'] = df_temp['datetime'].dt.floor('D')
    serie = df_temp.groupby([nivel, 'periodo_24h']).size().reset_index(name='conteo_delitos')
    serie.rename(columns={'periodo_24h': 'datetime'}, inplace=True)
    return serie

def _crear_features_rezagos_rolling(df, columna_objetivo='conteo_delitos',
                                  lags=[1, 7, 14, 28], ventanas=[7, 14, 28]):
    """Crea las features de rezagos y ventanas móviles."""
    df = df.copy()
    df = df.sort_values(['alcaldia_hecho', 'datetime'])
    
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('alcaldia_hecho')[columna_objetivo].shift(lag)
        
    for ventana in ventanas:
        df[f'rolling_mean_{ventana}'] = df.groupby('alcaldia_hecho')[columna_objetivo].transform(
            lambda x: x.rolling(window=ventana, min_periods=1).mean()
        )
        df[f'rolling_std_{ventana}'] = df.groupby('alcaldia_hecho')[columna_objetivo].transform(
            lambda x: x.rolling(window=ventana, min_periods=1).std()
        )

    df['diff_1'] = df.groupby('alcaldia_hecho')[columna_objetivo].diff(1)
    df['diff_7'] = df.groupby('alcaldia_hecho')[columna_objetivo].diff(7)
    df['tendencia'] = df.groupby('alcaldia_hecho')[columna_objetivo].transform(
        lambda x: x.rolling(window=90, min_periods=1).mean()
    )
    return df.dropna()

def _dividir_train_test_ts(df, test_size=0.2, fecha_col='datetime'):
    """Divide la serie temporal en conjuntos de entrenamiento y prueba."""
    df = df.sort_values(fecha_col)
    n = len(df)
    split_idx = int(n * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test

# ----------------------------------------------------
# 3. MODELADO XGBOOST
# ----------------------------------------------------

def _entrenar_xgboost_y_predecir(df_serie_clean, alcaldia_foco=None):
    """
    Entrena el modelo XGBoost en el conjunto de datos limpio y genera predicciones.
    """
    if alcaldia_foco is None:
        # Seleccionamos la alcaldía con más delitos como 'foco'
        top_alcaldia = df_serie_clean.groupby('alcaldia_hecho')['conteo_delitos'].sum().idxmax()
        alcaldia_foco = top_alcaldia
        
    df_alcaldia = df_serie_clean[df_serie_clean['alcaldia_hecho'] == alcaldia_foco].copy()
    df_alcaldia = df_alcaldia.sort_values('datetime')
    
    feature_cols = [col for col in df_alcaldia.columns
                    if col not in ['datetime', 'alcaldia_hecho', 'conteo_delitos', 'latitud', 'longitud']]
                    
    if len(df_alcaldia) < 20: 
        return None, None, None, None

    train, test = _dividir_train_test_ts(df_alcaldia, test_size=0.2)
    
    X_train = train[feature_cols]
    y_train = train['conteo_delitos']
    X_test = test[feature_cols]
    y_test = test['conteo_delitos']
    
    # Parámetros del modelo XGBoost (tomados de tu código)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    modelo = xgb.XGBRegressor(**params)
    modelo.fit(X_train, y_train, verbose=False)
    
    # Evaluar métricas (opcional, para verificación)
    y_pred_test = modelo.predict(X_test)
    metricas_test = _calcular_metricas(y_test, y_pred_test, "XGBoost (Test)")
    
    return modelo, df_alcaldia, metricas_test, feature_cols

def _generar_predicciones_24h_mapa(modelo, df_serie, feature_cols):
    """
    Genera la predicción de delitos para el próximo día, por cada alcaldía, 
    y construye el DataFrame de salida para el mapa.
    """
    alcaldias = df_serie['alcaldia_hecho'].unique()
    predicciones_list = []
    
    for alcaldia in alcaldias:
        df_alc = df_serie[df_serie['alcaldia_hecho'] == alcaldia].copy()
        df_alc = df_alc.sort_values('datetime')
        
        # Necesitamos al menos el rezago más largo (28) para calcular features
        if len(df_alc) < 28: continue

        ultima_fecha = df_alc['datetime'].max()
        proxima_fecha = ultima_fecha + timedelta(days=1)
        
        # Generar las features para la próxima fecha:
        df_future = pd.DataFrame({'datetime': [proxima_fecha], 'alcaldia_hecho': [alcaldia]})
        
        # 1. Agregar features de tiempo
        df_future['anio'] = df_future['datetime'].dt.year
        df_future['mes'] = df_future['datetime'].dt.month
        df_future['dia'] = df_future['datetime'].dt.day
        df_future['dia_semana'] = df_future['datetime'].dt.dayofweek
        df_future['hora'] = df_future['datetime'].dt.hour
        df_future['hora_sin'] = np.sin(2 * np.pi * df_future['hora'] / 24)
        df_future['hora_cos'] = np.cos(2 * np.pi * df_future['hora'] / 24)
        df_future['dia_semana_sin'] = np.sin(2 * np.pi * df_future['dia_semana'] / 7)
        df_future['dia_semana_cos'] = np.cos(2 * np.pi * df_future['dia_semana'] / 7)
        df_future['mes_sin'] = np.sin(2 * np.pi * df_future['mes'] / 12)
        df_future['mes_cos'] = np.cos(2 * np.pi * df_future['mes'] / 12)
        df_future['es_fin_semana'] = (df_future['dia_semana'] >= 5).astype(int)
        df_future['es_noche'] = ((df_future['hora'] >= 20) | (df_future['hora'] <= 6)).astype(int)
        
        # 2. Calcular features de rezago/rolling
        for col in feature_cols:
            if 'lag' in col:
                lag_num = int(col.split('_')[1])
                df_future[col] = df_alc['conteo_delitos'].iloc[-lag_num]
            elif 'rolling_mean' in col:
                window = int(col.split('_')[-1])
                df_future[col] = df_alc['conteo_delitos'].iloc[-window:].mean()
            elif 'rolling_std' in col:
                window = int(col.split('_')[-1])
                df_future[col] = df_alc['conteo_delitos'].iloc[-window:].std()
            elif col == 'diff_1':
                df_future[col] = df_alc['conteo_delitos'].iloc[-1] - df_alc['conteo_delitos'].iloc[-2]
            elif col == 'diff_7':
                df_future[col] = df_alc['conteo_delitos'].iloc[-1] - df_alc['conteo_delitos'].iloc[-8]
            elif col == 'tendencia':
                df_future[col] = df_alc['conteo_delitos'].iloc[-90:].mean()

        X_predict = df_future[feature_cols]

        try:
            prediccion = modelo.predict(X_predict)[0]
            prediccion = max(0, round(prediccion))
            
            # Obtener coordenadas de la última fila disponible
            df_coords = df_serie[df_serie['alcaldia_hecho'] == alcaldia].dropna(subset=['latitud', 'longitud'])
            lat_ejemplo = df_coords['latitud'].iloc[-1]
            lon_ejemplo = df_coords['longitud'].iloc[-1]
            
            predicciones_list.append({
                'alcaldia_hecho': alcaldia,
                'fecha_prediccion': proxima_fecha.strftime('%Y-%m-%d'),
                'latitud': lat_ejemplo,
                'longitud': lon_ejemplo,
                'conteo_delitos_predichos': prediccion
            })
        except Exception as e:
            # print(f"Error en {alcaldia}: {str(e)}")
            continue
            
    df_predicciones = pd.DataFrame(predicciones_list)
    return df_predicciones

def _generar_serie_temporal_salida(df_raw, df_predicciones_24h):
    """
    Combina la serie histórica (estandarizada) con la predicción a nivel de delito.
    
    MODIFICADO: Ahora devuelve el DataFrame de predicciones expandidas por separado.
    """
    df_limpio = df_raw.copy()
    
    # Estandarización de delitos
    df_limpio['delito'] = df_limpio['delito'].str.upper()
    df_limpio.loc[df_limpio['delito'].str.contains('ROBO', na=False), 'delito'] = 'ROBO'
    df_limpio.loc[df_limpio['delito'].str.contains('DANO EN PROPIEDAD AJENA|DANO SUELO ACTIVIDAD INVASION O EXTRACCION', na=False, regex=True), 'delito'] = 'DANO EN PROPIEDAD AJENA'
    df_limpio.loc[df_limpio['delito'].str.contains('SECUESTRO', na=False), 'delito'] = 'SECUESTRO'
    df_limpio.loc[df_limpio['delito'].str.contains('HOMICIDIO', na=False), 'delito'] = 'HOMICIDIO'
    df_limpio.loc[df_limpio['delito'].str.contains('VIOLACION', na=False), 'delito'] = 'VIOLACION'
    
    # 1. Histórico (Agregado por fecha, alcaldía, delito)
    df_historico = df_limpio.groupby([
        df_limpio['datetime'].dt.date, 'alcaldia_hecho', 'delito'
    ]).size().reset_index(name='conteo')
    df_historico.columns = ['fecha', 'alcaldia', 'delito', 'conteo']
    df_historico['fecha'] = pd.to_datetime(df_historico['fecha'])
    df_historico['tipo'] = 'H' # Histórico

    # 2. Predicción (Expandida a nivel de delito según la proporción histórica)
    predicciones_expandidas = []
    for _, row in df_predicciones_24h.iterrows():
        # Proporción de delitos en esa alcaldía
        delitos_alc = df_limpio[
            df_limpio['alcaldia_hecho'] == row['alcaldia_hecho']
        ]['delito'].value_counts(normalize=True)
        
        for delito, proporcion in delitos_alc.items():
            predicciones_expandidas.append({
                'fecha': pd.to_datetime(row['fecha_prediccion']),
                'alcaldia': row['alcaldia_hecho'],
                'delito': delito,
                'conteo': round(row['conteo_delitos_predichos'] * proporcion),
                'tipo': 'P' # Predicción
            })
            
    df_pred_expandido = pd.DataFrame(predicciones_expandidas)
    
    # 3. Combinación
    df_completo = pd.concat([df_historico, df_pred_expandido], ignore_index=True)
    df_completo = df_completo.sort_values(['alcaldia', 'delito', 'fecha'])
    
    # MODIFICACIÓN: Devolver el DF completo y el DF de predicciones expandidas.
    return df_completo[['fecha', 'alcaldia', 'delito', 'conteo', 'tipo']], df_pred_expandido

# ----------------------------------------------------
# 4. PIPELINE PRINCIPAL PARA STREAMLIT
# ----------------------------------------------------

def process_temporal_data(uploaded_zip_buffer):
    """
    Función principal que orquesta el pipeline completo de XGBoost.
    Retorna dos buffers: uno ZIP (Serie Temporal) y uno CSV (Mapa de Predicciones).
    """
    
    # 1. Leer el ZIP y encontrar el CSV
    try:
        zip_input = io.BytesIO(uploaded_zip_buffer.read())
        with zipfile.ZipFile(zip_input, 'r') as zf:
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError("No se encontró ningún archivo CSV dentro del ZIP.")
            with zf.open(csv_files[0]) as csv_file:
                # Leer el archivo CSV
                df_raw_in = pd.read_csv(io.TextIOWrapper(csv_file, encoding='utf-8'))
                
    except Exception as e:
        # Error en la lectura del ZIP o CSV
        return None, None
        
    # 2. Preprocesamiento, Feature Engineering y Agregación
    try:
        # Aplicamos la limpieza completa al archivo crudo
        df_limpio = _load_and_clean_data(df_raw_in)
    except KeyError as e:
        raise Exception(f"Error crítico en la columna de entrada. Faltan columnas clave como 'fecha_hecho' o 'hora_hecho'. Error: {e}")
    
    # No podemos continuar sin datos suficientes para los rezagos (mínimo 28 días)
    if len(df_limpio['datetime'].dt.date.unique()) < 28: 
        raise Exception("Error: El conjunto de datos es muy corto para el modelado temporal (requiere al menos 28 días de datos limpios).")
        
    df_features = _crear_features_temporales(df_limpio)
    df_serie_base = _crear_serie_temporal_agregada(df_features, nivel='alcaldia_hecho', freq='24H')
    df_serie_clean = _crear_features_rezagos_rolling(df_serie_base)
    
    # Unimos las coordenadas usando merge para evitar errores de alineación
    df_coords = df_limpio[['alcaldia_hecho', 'latitud', 'longitud']].drop_duplicates().groupby('alcaldia_hecho').first().reset_index()
    df_serie_clean = pd.merge(df_serie_clean, df_coords, on='alcaldia_hecho', how='left')
    
    # 3. Entrenamiento y Predicción
    result_modelo = _entrenar_xgboost_y_predecir(df_serie_clean)
    if result_modelo is None or result_modelo[0] is None:
        return None, None
        
    modelo_xgb, _, _, feature_cols = result_modelo
    
    # 4. Generación de DataFrames de Salida
    df_predicciones_24h_mapa = _generar_predicciones_24h_mapa(modelo_xgb, df_serie_clean, feature_cols)
    
    # Capturar ambos DataFrames de salida
    df_serie_completa_out, df_pred_expandido = _generar_serie_temporal_salida(df_limpio, df_predicciones_24h_mapa)


    # 5. Empaquetado de Resultados
    
    # Resultado 1: serie_temporal_delitos.csv (Sigue en ZIP)
    zip_serie = _create_output_zip(
        df_serie_completa_out, 
        "serie_temporal_delitos.csv"
    )
    
    # Preparar el DataFrame para el mapa (Formato: alcaldia, delito, count_delitos_predichos)
    df_mapa_pred_final = df_pred_expandido.copy()
    
    df_mapa_pred_final = df_mapa_pred_final.rename(columns={
        'conteo': 'count_delitos_predichos'
    })
    
    # Seleccionar solo las columnas requeridas (alcaldia, delito, count_delitos_predichos)
    df_mapa_pred_final = df_mapa_pred_final[['alcaldia', 'delito', 'count_delitos_predichos']]

    # Resultado 2: mapa_predicciones_24h.csv (MODIFICADO: Ahora es CSV plano)
    csv_mapa = _create_output_csv_buffer(df_mapa_pred_final)

    # Nota: La función principal debe devolver los dos buffers de salida
    return zip_serie, csv_mapa