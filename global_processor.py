# global_processor.py
import re
import pandas as pd
import numpy as np
import unicodedata
import zipfile
import io
import warnings
warnings.filterwarnings('ignore')
# ----------------------------------------------------
# 1. CONSTANTES Y DICCIONARIOS
# ----------------------------------------------------
# Estas son las coordenadas de todas las delegaciones que se usarán para imputar las mismas
COORDENADAS_REF = {
    "ALVARO OBREGON": {"latitud": 19.390806, "longitud": -99.2283},
    "AZCAPOTZALCO": {"latitud": 19.48, "longitud": -99.18436},
    "BENITO JUAREZ": {"latitud": 19.3984, "longitud": -99.15766},
    "COYOACAN": {"latitud": 19.3467, "longitud": -99.1617},
    "CUAJIMALPA DE MORELOS": {"latitud": 19.37444444, "longitud": -99.28472222},
    "CUAUHTEMOC": {"latitud": 19.44506, "longitud": -99.14612},
    "GUSTAVO A. MADERO": {"latitud": 19.48407, "longitud": -99.11144},
    "IZTACALCO": {"latitud": 19.395, "longitud": -99.09861111},
    "IZTAPALAPA": {"latitud": 19.35529, "longitud": -99.06224},
    "LA MAGDALENA CONTRERAS": {"latitud": 19.33212, "longitud": -99.21118},
    "MIGUEL HIDALGO": {"latitud": 19.43411, "longitud": -99.20024},
    "MILPA ALTA": {"latitud": 19.1919839, "longitud": -99.0228937},
    "TLAHUAC": {"latitud": 19.28689, "longitud": -99.00507},
    "TLALPAN": {"latitud": 19.22694444, "longitud": -99.20583333},
    "VENUSTIANO CARRANZA": {"latitud": 19.419261, "longitud": -99.113701},
    "XOCHIMILCO": {"latitud": 19.25465, "longitud": -99.10356}
}
# Listado de colonias consideradas centro y trabajo
COLONIA_CENTRO = ['ALTAVISTA', 'AMPLIACION GRANADA', 'ANAHUAC', 'ANAHUAC I SECCION', 'ANZURES', 'ARGENTINA ANTIGUA', 'ARGENTINA PONIENTE', 'BOSQUE DE CHAPULTEPEC I SECCION', 'BOSQUE DE CHAPULTEPEC II SECCION', 'BOSQUE DE CHAPULTEPEC III SECCION', 'BUENAVISTA','CENTRO', 'CENTRO AREA 1', 'CENTRO AREA 3', 'CENTRO AREA 5', 'CHIMALISTAC', 'CONDESA', 'CUAUHTEMOC', 'CUAUHTEMOC PENSIL', 'DEL VALLE CENTRO', 'DOCTORES', 'ESCANDON I SECCION', 'GRANADA', 'GUERRERO', 'HIPODROMO', 'HIPODROMO CONDESA', 'JUAREZ', 'LOMAS DE CHAPULTEPEC I SECCION', 'LOMAS DE CHAPULTEPEC IV SECCION LOMAS VIRREYES', 'LOMAS DE CUAUTEPEC', 'MIRAVALLE', 'MIXCOAC', 'MODERNA', 'MORELOS', 'NARVARTE', 'NATIVITAS', 'NATIVITAS PUEBLO SAN LUCAS XOCHIMANCA', 'NONOALCO TLATELOLCO', 'OBRERA', 'PENSIL NORTE', 'PENSIL SUR', 'POLANCO', 'POPOTLA', 'PORTALES ORIENTE', 'PORTALES SUR', 'ROMA NORTE', 'ROMA SUR', 'SAN ANGEL', 'SAN ANGEL INN', 'SAN MIGUEL 1 SECCION', 'SAN MIGUEL 2 SECCION', 'SAN MIGUEL CHAPULTEPEC', 'SAN PEDRO DE LOS PINOS', 'SAN RAFAEL', 'SAN RAFAEL TICOMAN', 'SAN SIMON TICUMAC', 'SANTA CRUZ ATOYAC', 'SANTA MARIA LA RIBERA', 'TABACALERA', 'TACUBAYA', 'VALLEJO', 'XOCO', 'ZONA CENTRO']
COLONIA_TRABAJO = ['AMPLIACION GRANADA','ANAHUAC','ANAHUAC I SECCION','BOSQUE DE CHAPULTEPEC I SECCION','BOSQUE DE CHAPULTEPEC II SECCION','BOSQUE DE CHAPULTEPEC III SECCION','CONDESA','CUAUHTEMOC','GRANADA','HIPODROMO','HIPODROMO CONDESA','JUAREZ','LOMAS DE CHAPULTEPEC I SECCION','LOMAS DE CHAPULTEPEC IV SECCION LOMAS VIRREYES','POLANCO','SAN MIGUEL CHAPULTEPEC','TACUBAYA','ZONA ESCOLAR','LINDAVISTA','TEPEYAC INSURGENTES','VALLEJO','SANTA MARIA LA RIBERA','SAN RAFAEL','TABACALERA','ROMA NORTE','ROMA SUR','DOCTORES','NARVARTE','DEL VALLE CENTRO','NAPOLES','INSURGENTES MIXCOAC','MIXCOAC','SAN JOSE INSURGENTES','GUADALUPE INN','AXOTLA','CHIMALISTAC','TLALPAN CENTRO I','TLALPAN CENTRO II','PUEBLO SANTA FE','SANTA FE','CONTADERO','LOCAXCO']
# Actualizaciones manuales a las coordenadas o alcaldías
CAMBIOS_MANUALES = [
    {
        "delito": "ROBO DE DINERO",
        "fecha_hora": "2017-12-15 09:30:00",
        "alcaldia": "CUAUHTEMOC",
        "latitud": 19.45631,
        "longitud": -99.161421
    },
    { "delito": "ROBO A PASAJERO EN ECOBUS SIN VIOLENCIA", "fecha_hora": "2021-06-18 21:00:00", "alcaldia": "GUSTAVO A. MADERO", "latitud": 19.4, "longitud": -99.07325169 },
    { "delito": "ROBO A PASAJERO A BORDO DE TRANSPORTE PÚBLICO CON VIOLENCIA", "fecha_hora": "2021-06-18 21:40:00", "alcaldia": "GUSTAVO A. MADERO", "latitud": 19.4, "longitud": -99.0732516942668 },
    { "delito": "EXTORSION", "fecha_hora": "2021-09-03 09:00:00", "alcaldia": "ALVARO OBREGON", "latitud": 19.390806, "longitud": -99.235012 },
    { "delito": "LESIONES CULPOSAS POR TRANSITO VEHICULAR", "fecha_hora": "2021-12-23 21:00:00", "alcaldia": "AZCAPOTZALCO", "latitud": 19.484102, "longitud": -99.152802 },
    { "delito": "LESIONES CULPOSAS POR TRANSITO VEHICULAR EN COLISION", "fecha_hora": "2023-02-25 15:00:00", "alcaldia": "ALVARO OBREGON", "latitud": 19.381942, "longitud": -99.211072 }
]
# Lista de delitos a mantener (la lista larga de tu código original)
DELITOS_A_MANTENER = ["DANO EN PROPIEDAD AJENA CULPOSA", "DANO EN PROPIEDAD AJENA CULPOSA POR TRANSITO VEHICULAR A AUTOMOVIL", "DANO EN PROPIEDAD AJENA CULPOSA POR TRANSITO VEHICULAR A BIENES INMUEBLES", "DANO EN PROPIEDAD AJENA CULPOSA POR TRANSITO VEHICULAR A VIAS DE COMUNICACION", "DANO EN PROPIEDAD AJENA INTENCIONAL", "DANO EN PROPIEDAD AJENA INTENCIONAL A AUTOMOVIL", "DANO EN PROPIEDAD AJENA INTENCIONAL A BIENES INMUEBLES", "DANO EN PROPIEDAD AJENA INTENCIONAL A CASA HABITACION", "DANO EN PROPIEDAD AJENA INTENCIONAL A NEGOCIO", "DANO EN PROPIEDAD AJENA INTENCIONAL A VIAS DE COMUNICACION", "DANO SUELO ACTIVIDAD INVASION O EXTRACCION", "HOMICIDIO CULPOSO", "HOMICIDIO CULPOSO CON EXCLUYENTES DE RESPONSABILIDAD", "HOMICIDIO CULPOSO CON EXCLUYENTES DE RESPONSABILIDAD", "HOMICIDIO CULPOSO FUERA DEL DF ATROPELLADO", "HOMICIDIO CULPOSO FUERA DEL DF COLISION", "HOMICIDIO CULPOSO ", "POR ARMA DE FUEGO", "HOMICIDIO CULPOSO POR INSTRUMENTO PUNZO CORTANTE", "HOMICIDIO CULPOSO POR TRANSITO VEHICULAR", "HOMICIDIO CULPOSO POR TRANSITO VEHICULAR ATROPELLADO", "HOMICIDIO CULPOSO POR TRANSITO VEHICULAR CAIDA", "HOMICIDIO CULPOSO POR TRANSITO VEHICULAR COLISION", "HOMICIDIO DOLOSO", "HOMICIDIO POR AHORCAMIENTO", "HOMICIDIO POR ARMA BLANCA", "HOMICIDIO POR ARMA DE FUEGO", "HOMICIDIO POR GOLPES", "HOMICIDIO POR INMERSION", "HOMICIDIOS INTENCIONALES OTROS", "ROBO A CASA HABITACION CON VIOLENCIA", "ROBO A CASA HABITACION SIN VIOLENCIA", "ROBO A CASA HABITACION Y VEHICULO CON VIOLENCIA", "ROBO A CASA HABITACION Y VEHICULO SIN VIOLENCIA", "ROBO A LOCALES SEMIFIJOS PUESTOS DE ALIMENTOSBEBIDAS ENSERES PERIODICOSLOTERIA OTROS", "ROBO A NEGOCIO CON VIOLENCIA", "ROBO A NEGOCIO CON VIOLENCIA POR FARDEROS TIENDAS DE AUTOSERVICIO", "ROBO A NEGOCIO CON VIOLENCIA POR FARDEROS TIENDAS DE CONVENIENCIA", "ROBO A ", "NEGOCIO NOMINA Y VEHICULO CON VIOLENCIA", "ROBO A NEGOCIO SIN VIOLENCIA", "ROBO A NEGOCIO SIN VIOLENCIA POR FARDEROS", "ROBO A NEGOCIO SIN VIOLENCIA POR FARDEROS TIENDAS DE AUTOSERVICIO", "ROBO A NEGOCIO SIN VIOLENCIA POR FARDEROS TIENDAS DE CONVENIENCIA", "ROBO A NEGOCIO Y VEHICULO CON VIOLENCIA", "ROBO A NEGOCIO Y VEHICULO SIN VIOLENCIA", "ROBO A OFICINA PUBLICA CON VIOLENCIA", "ROBO A OFICINA PUBLICA SIN VIOLENCIA", "ROBO A PASAJERO CONDUCTOR DE TAXI CON VIOLENCIA", "ROBO A PASAJERO CONDUCTOR DE VEHICULO CON VIOLENCIA", "ROBO A PASAJERO A BORDO DE CABLEBUS CON VIOLENCIA", "ROBO A PASAJERO A BORDO DE CABLEBUS SIN VIOLENCIA", "ROBO A PASAJERO A BORDO DE METRO CON VIOLENCIA", "ROBO A PASAJERO A BORDO ", "DE METRO SIN VIOLENCIA", "ROBO A PASAJERO A BORDO DE METROBUS CON VIOLENCIA", "ROBO A PASAJERO A BORDO DE METROBUS SIN VIOLENCIA", "ROBO A PASAJERO A BORDO DE PESERO COLECTIVO CON VIOLENCIA", "ROBO A PASAJERO A BORDO DE PESERO COLECTIVO SIN VIOLENCIA", "ROBO A PASAJERO A BORDO DE PESERO Y VEHICULO CON VIOLENCIA", "ROBO A PASAJERO A BORDO DE TAXI CON VIOLENCIA", "ROBO A PASAJERO A BORDO DE TAXI SIN VIOLENCIA", "ROBO A PASAJERO A BORDO DE TRANSPORTE PUBLICO CON VIOLENCIA", "ROBO A PASAJERO A BORDO DE TRANSPORTE PUBLICO SIN VIOLENCIA", "ROBO A PASAJERO EN AUTOBUS FORANEO CON VIOLENCIA", "ROBO A PASAJERO EN AUTOBUS FORANEO SIN VIOLENCIA", "ROBO A PASAJERO EN ECOBUS ", "CON VIOLENCIA", "ROBO A PASAJERO EN ECOBUS SIN VIOLENCIA", "ROBO A PASAJERO EN RTP CON VIOLENCIA", "ROBO A PASAJERO EN RTP SIN VIOLENCIA", "ROBO A PASAJERO EN TREN LIGERO CON VIOLENCIA", "ROBO A PASAJERO EN TREN LIGERO SIN VIOLENCIA", "ROBO A PASAJERO EN TREN SUBURBANO CON VIOLENCIA", "ROBO A PASAJERO EN TREN SUBURBANO SIN VIOLENCIA", "ROBO A PASAJERO EN TROLEBUS CON VIOLENCIA", "ROBO A PASAJERO EN TROLEBUS SIN VIOLENCIA", "ROBO A REPARTIDOR CON VIOLENCIA", "ROBO A REPARTIDOR SIN VIOLENCIA", "ROBO A REPARTIDOR Y VEHICULO CON VIOLENCIA", "ROBO A REPARTIDOR Y VEHICULO SIN VIOLENCIA", "ROBO A SUCURSAL BANCARIA ASALTO BANCARIO CON VIOLENCIA", "ROBO A SUCURSAL BANCARIA CON VIOLENCIA", "ROBO A SUCURSAL BANCARIA DENTRO DE TIENDAS DE ", "AUTOSERVICIO CON VIOLENCIA", "ROBO A SUCURSAL BANCARIA DENTRO DE TIENDAS DE AUTOSERVICIO SV", "ROBO A SUCURSAL BANCARIA SIN VIOLENCIA", "ROBO A SUCURSAL BANCARIA SUPERMERCADO CON VIOLENCIA", "ROBO A SUCURSAL BANCARIA SUPERMERCADO SIN VIOLENCIA", "ROBO A TRANSEUNTE A BORDO DE TAXI PUBLICO Y PRIVADO CON VIOLENCIA", "ROBO A TRANSEUNTE A BORDO DE TAXI PUBLICO Y PRIVADO SIN VIOLENCIA", "ROBO A TRANSEUNTE CONDUCTOR DE TAXI PUBLICO Y PRIVADO CON VIOLENCIA", "ROBO A TRANSEUNTE DE CELULAR CON VIOLENCIA", "ROBO A TRANSEUNTE DE CELULAR SIN VIOLENCIA", "ROBO A TRANSEUNTE EN CINE CON VIOLENCIA", "ROBO A TRANSEUNTE EN HOTEL CON VIOLENCIA", "ROBO A TRANSEUNTE EN NEGOCIO CON VIOLENCIA", "ROBO A TRANSEUNTE EN PARQUES Y MERCADOS CON VIOLENCIA", "ROBO ", "A TRANSEUNTE EN RESTAURANT CON VIOLENCIA", "ROBO A TRANSEUNTE EN TERMINAL DE PASAJEROS CON VIOLENCIA", "ROBO A TRANSEUNTE EN VIA PUBLICA CON VIOLENCIA", "ROBO A TRANSEUNTE EN VIA PUBLICA NOMINA CON VIOLENCIA", "ROBO A TRANSEUNTE EN VIA PUBLICA NOMINA SIN VIOLENCIA", "ROBO A TRANSEUNTE EN VIA PUBLICA SIN VIOLENCIA", "ROBO A TRANSEUNTE SALIENDO DEL BANCO CON VIOLENCIA", "ROBO A TRANSEUNTE SALIENDO DEL CAJERO CON VIOLENCIA", "ROBO A TRANSEUNTE Y VEHICULO CON VIOLENCIA", "ROBO A TRANSPORTISTA Y VEHICULO PESADO CON VIOLENCIA", "ROBO A TRANSPORTISTA Y VEHICULO PESADO SIN VIOLENCIA", "ROBO DE ACCESORIOS DE AUTO", "ROBO DE ALHAJAS", "ROBO DE ANIMALES", "ROBO DE ARMA", "ROBO DE CONTENEDORES DE TRAILERS SV", "ROBO DE DINERO", "ROBO DE DOCUMENTOS", "ROBO DE FLUIDOS", "ROBO DE ", "INFANTE", "ROBO DE MAQUINARIA CON VIOLENCIA", "ROBO DE MAQUINARIA SIN VIOLENCIA", "ROBO DE MERCANCIA A TRANSPORTISTA CV", "ROBO DE MERCANCIA EN CONTENEDEROS EN AREAS FEDERALES", "ROBO DE MERCANCIA EN CONTENEDEROS EN AREAS FEDERALES", "ROBO DE MOTOCICLETA CON VIOLENCIA", "ROBO DE MOTOCICLETA SIN VIOLENCIA", "ROBO DE OBJETOS", "ROBO DE OBJETOS A ESCUELA", "ROBO DE OBJETOS DEL INTERIOR DE UN VEHICULO", "ROBO DE PLACA DE AUTOMOVIL", "ROBO DE VEHICULO DE PEDALES", "ROBO DE VEHICULO DE SERVICIO DE TRANSPORTE CON VIOLENCIA", "ROBO DE VEHICULO DE SERVICIO DE TRANSPORTE SIN VIOLENCIA", "ROBO DE VEHICULO DE SERVICIO OFICIAL CON VIOLENCIA", "ROBO DE VEHICULO DE SERVICIO OFICIAL SIN VIOLENCIA", "ROBO DE VEHICULO DE SERVICIO PARTICULAR CON VIOLENCIA", "ROBO DE VEHICULO DE SERVICIO PARTICULAR SIN ", "VIOLENCIA", "ROBO DE VEHICULO DE SERVICIO PUBLICO CON VIOLENCIA", "ROBO DE VEHICULO DE SERVICIO PUBLICO SIN VIOLENCIA", "ROBO DE VEHICULO ELECTRICO MOTOPATIN", "ROBO DE VEHICULO EN PENSION TALLER Y AGENCIAS CV", "ROBO DE VEHICULO EN PENSION TALLER Y AGENCIAS SV", "ROBO DE VEHICULO Y NOMINA CON VIOLENCIA", "ROBO DURANTE TRASLADO DE VALORES NOMINA CON VIOLENCIA", "ROBO DURANTE TRASLADO DE VALORES NOMINA SIN VIOLENCIA", "ROBO EN EVENTOS MASIVOS DEPORTIVOS CULTURALES RELIGIOSOS Y ARTISTICOS SV", "ROBO EN INTERIOR DE EMPRESA NOMINA CON VIOLENCIA", "ROBO EN INTERIOR DE EMPRESA NOMINA SIN VIOLENCIA", "ROBO SV DENTRO DE NEGOCIOS AUTOSERVICIOS CONVENIENCIA", "SECUESTRO", "SECUESTRO EXPRESS PARA COMETER ROBO O EXTORSION", "SECUESTRO EXTORSIVO", "VIOLACION"]
# ----------------------------------------------------
# 2. FUNCIONES DE LIMPIEZA Y FEATURE ENGINEERIN
# ----------------------------------------------------
def normalize_month_names(month):
    month_map = {'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo','April': 'Abril', 'May': 'Mayo', 'June': 'Junio','July': 'Julio', 'August': 'Agosto', 'September': 'Septiembre','October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre' }

    return month_map.get(month, month)



def format_data_types(df):

   

    # 🆕 SOLUCIÓN: Limpiar fecha_hecho a formato estricto antes de la conversión

    if 'fecha_hecho' in df.columns:

        df['fecha_hecho'] = df['fecha_hecho'].astype(str).str.slice(0, 10)

   

    df['anio_hecho'] = df['anio_hecho'].astype('Int32')

    df['mes_hecho'] = df['mes_hecho'].astype('category')

   

    # Intentamos la conversión a datetime con el formato limpio

    df['fecha_hecho'] = pd.to_datetime(df['fecha_hecho'], format="%Y-%m-%d", errors='coerce')

   

    df['delito'] = df['delito'].astype('category')

    df['categoria_delito'] = df['categoria_delito'].astype('category')

    df['colonia_hecho'] = df['colonia_hecho'].astype('category')

    df['alcaldia_hecho'] = df['alcaldia_hecho'].astype('category')

    df['latitud'] = df['latitud'].astype('float32')

    df['longitud'] = df['longitud'].astype('float32')

   

    # Ajustar la hora también para evitar problemas

    df["hora_hecho"] = df["hora_hecho"].astype(str).str.slice(0, 8)

    df["hora_hecho"] = pd.to_datetime(df["hora_hecho"], format="%H:%M:%S", errors="coerce").dt.time

   

    return df



def fill_missing_data(df):

    df['anio_hecho'] = df['anio_hecho'].fillna(df['anio_inicio'])

    df['mes_hecho'] = df['mes_hecho'].fillna(df['mes_inicio'])

   

    fecha_inicio_dt_temp = pd.to_datetime(df['fecha_inicio'], errors='coerce')

    fecha_hecho_dt_temp = pd.to_datetime(df['fecha_hecho'], errors='coerce')

    tiempo_transcurrido_temp = fecha_inicio_dt_temp - fecha_hecho_dt_temp

    average_tiempo_transcurrido = tiempo_transcurrido_temp.mean()

   

    df['fecha_hecho'] = df['fecha_hecho'].fillna(pd.to_datetime(df['fecha_inicio'], errors='coerce') - average_tiempo_transcurrido)

    return df



def combine_date_and_time(df):

    fecha_str = df["fecha_hecho"].dt.strftime("%Y-%m-%d")

    hora_str = df["hora_hecho"].astype(str).str.slice(0,8)

    df["fecha_hora"] = pd.to_datetime(fecha_str + " " + hora_str, format="%Y-%m-%d %H:%M:%S", errors="coerce")

    return df



def fill_missing_fecha_hora(df):

    df['fecha_inicio_dt'] = pd.to_datetime(df['fecha_inicio'], errors='coerce')

    missing_fecha_hora_mask = df['fecha_hora'].isna()

   

    fecha_inicio_dt_temp = pd.to_datetime(df['fecha_inicio'], errors='coerce')

    fecha_hecho_dt_temp = pd.to_datetime(df['fecha_hecho'], errors='coerce')

    tiempo_transcurrido_temp = fecha_inicio_dt_temp - fecha_hecho_dt_temp

    average_tiempo_transcurrido = tiempo_transcurrido_temp.mean()

   

    estimated_fecha_hora = df.loc[missing_fecha_hora_mask, 'fecha_inicio_dt'] - average_tiempo_transcurrido

    df.loc[missing_fecha_hora_mask, 'fecha_hora'] = estimated_fecha_hora

    return df



def actualizar_registros(data, cambios):

    for cambio in cambios:

        target_datetime = pd.to_datetime(cambio["fecha_hora"])

        condition = ((data["delito"] == cambio["delito"]) & (data["fecha_hora"] == target_datetime))

        data.loc[condition, "alcaldia_hecho"] = cambio["alcaldia"]

        data.loc[condition, "latitud"] = cambio["latitud"]

        data.loc[condition, "longitud"] = cambio["longitud"]

    return data



def rellenar_coordenadas(data):

    for alcaldia, coords in COORDENADAS_REF.items():

        condition = data["longitud"].isna() & (data["alcaldia_hecho"] == alcaldia)

        data.loc[condition, "latitud"] = coords["latitud"]

        data.loc[condition, "longitud"] = coords["longitud"]

    return data



def normalize_colonia(colonia):

    if isinstance(colonia, str):

        colonia = ''.join(c for c in unicodedata.normalize('NFD', colonia) if unicodedata.category(c) != 'Mn')

        colonia = colonia.upper()

        colonia = re.sub(r'[^A-Z0-9\s]', '', colonia)

    return colonia



def normalize_text(text):

    if isinstance(text, str):

        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

        text = text.upper()

        text = re.sub(r'[^A-Z0-9\s]', '', text)

    return text



def assign_centro(row):

    if row['colonia_hecho'] in COLONIA_CENTRO or row['alcaldia_hecho'] == 'CUAUHTEMOC':

        if row['alcaldia_hecho'] in ['GUSTAVO A. MADERO', 'CUAJIMALPA', 'ALVARO OBREGON', 'MAGDALENA CONTRERAS', 'AZCAPOTZALCO', 'IZTAPALAPA','TLAHUAC','MILPA ALTA', 'XOCHIMILCO']:

             return 0

        return 1

    return 0



def assign_trabajo(row):

    if row['colonia_hecho'] in COLONIA_TRABAJO:

        return 1

    return 0



def estandarizar_delitos(df):

    df['delito'] = df['delito'].str.upper()

    df.loc[df['delito'].str.contains('ROBO'), 'delito'] = 'ROBO'

    df.loc[df['delito'].str.contains('DANO EN PROPIEDAD AJENA'), 'delito'] = 'DANO EN PROPIEDAD AJENA'

    df.loc[df['delito'].str.contains('DANO SUELO ACTIVIDAD INVASION O EXTRACCION'), 'delito'] = 'DANO EN PROPIEDAD AJENA'

    df.loc[df['delito'].str.contains('SECUESTRO'), 'delito'] = 'SECUESTRO'

    df.loc[df['delito'].str.contains('HOMICIDIO'), 'delito'] = 'HOMICIDIO'

    df.loc[df['delito'].str.contains('VIOLACION'), 'delito'] = 'VIOLACION'

    df.loc[df['delito'].str.contains('FEMINICIDIO'), 'delito'] = 'FEMINICIDIO'

    df.loc[df['delito'].str.contains('EXTORSION'), 'delito'] = 'EXTORSION'

    return df



def crear_horario_laboral(df):

    df['hora_hecho'] = df['hora_hecho'].astype(str)

    df['hora_int'] = pd.to_numeric(df['hora_hecho'].str.split(':').str[0], errors='coerce')

    df.dropna(subset=['hora_int'], inplace=True)

    df['hora_int'] = df['hora_int'].astype(int)



    df['horario_laboral'] = ((df['hora_int'] >= 8) & (df['hora_int'] < 18)).astype(int)

   

    df = df.drop(columns=['hora_int'], errors='ignore')

    return df



# ----------------------------------------------------

# 3. PIPELINE PRINCIPAL Y MANEJO DE ARCHIVOS

# ----------------------------------------------------



def process_global_csv(uploaded_zip_buffer):

    """

    Función principal que toma un buffer ZIP, procesa el CSV interno y devuelve

    un DataFrame limpio y con features de tiempo/espacio.

    """

   

    # 1. Leer el ZIP y encontrar el CSV

    try:

        # Usamos BytesIO para manejar el buffer del archivo subido

        zip_input = io.BytesIO(uploaded_zip_buffer.read())

       

        with zipfile.ZipFile(zip_input, 'r') as zf:

            # Asumimos que el CSV es el único o el principal archivo dentro del ZIP

            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]

            if not csv_files:

                raise ValueError("No se encontró ningún archivo CSV dentro del ZIP.")

           

            # Leer el primer archivo CSV encontrado

            with zf.open(csv_files[0]) as csv_file:

                # El archivo puede estar codificado, intentamos leerlo como texto

                data = pd.read_csv(io.TextIOWrapper(csv_file, encoding='utf-8'))

               

    except Exception as e:

        print(f"Error al procesar el archivo ZIP/CSV: {e}")

        return None



    # 2. Ejecución del Pipeline de Limpieza

   

    # Traducción de meses

    data['mes_hecho'] = data['mes_hecho'].apply(normalize_month_names)



    # Formatear columnas a tipos deseados

    data = format_data_types(data)



    # Imputación de días, meses y años

    data = fill_missing_data(data)



    # Combinación de columnas fecha y hora

    data = combine_date_and_time(data)



    # Imputación de valores faltantes de hora y fecha_hora

    data = fill_missing_fecha_hora(data)

   

    # --- Eliminación de columnas no deseadas ---

    data = data.drop(['anio_hecho','mes_hecho','fecha_hecho','anio_inicio', 'mes_inicio','fecha_inicio','hora_inicio','competencia','fiscalia','agencia','unidad_investigacion','colonia_catalogo','alcaldia_catalogo','municipio_hecho','fecha_inicio_dt'], axis=1, errors='ignore')



    # --- Filtrado de Filas Nulas/Indeterminadas ---

    data = data[data['alcaldia_hecho'] != 'CDMX (indeterminada)']

    data = data[data['colonia_hecho'] != ' ']

    data = data.dropna(subset=['colonia_hecho', 'alcaldia_hecho', 'fecha_hora'])



    # --- Imputación y Normalización Geográfica/Temporal ---

    data = actualizar_registros(data, CAMBIOS_MANUALES) # Actualizar registros fijos

    data = rellenar_coordenadas(data) # Rellenar coordenadas faltantes por alcaldía



    # --- Normalización de Texto ---

    data['colonia_hecho_normalized'] = data['colonia_hecho'].apply(normalize_colonia)

    data['colonia_hecho'] = data['colonia_hecho_normalized'].astype('category')



    data['delito_normalized'] = data['delito'].apply(normalize_text)

    data['categoria_delito_normalized'] = data['categoria_delito'].apply(normalize_text)

    data['delito'] = data['delito_normalized'].astype('category')

    data['categoria_delito'] = data['categoria_delito_normalized'].astype('category')

   

    # --- Creación de Features Binarias ---

    data['centro'] = data.apply(assign_centro, axis=1)

    data['trabajo'] = data.apply(assign_trabajo, axis=1)

   

    # --- Estandarización Final de Delitos y Horario ---

    data = estandarizar_delitos(data)

    data = crear_horario_laboral(data)

   

    # --- Limpieza Final ---

    data = data[data['fecha_hora'].dt.year != 222]

    data = data.drop(columns=['colonia_hecho_normalized','delito_normalized','categoria_delito_normalized'], axis=1, errors='ignore')

    data = data.drop_duplicates()

   

    # Filtrar solo los delitos que queremos mantener

    data = data[data['delito'].isin(DELITOS_A_MANTENER)].copy()


    # 3. Seleccionar columnas finales para el mapa (salida)

    columns_final = ['hora_hecho', 'delito', 'alcaldia_hecho', 'latitud', 'longitud', 'centro', 'trabajo', 'horario_laboral']

   

    return data[columns_final]





def create_zip_from_dataframe(df, filename="df_mapa_procesado.csv"):

    """

    Convierte el DataFrame en un archivo ZIP con un CSV dentro, listo para descargar.

    """

    if df is None or df.empty:

        return None

   

    # Convertir DataFrame a CSV en memoria

    csv_buffer = io.StringIO()

    df.to_csv(csv_buffer, index=False, encoding='utf-8')

    csv_bytes = csv_buffer.getvalue().encode('utf-8')

   

    # Crear el archivo ZIP en memoria

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:

        zf.writestr(filename, csv_bytes)

       

    return zip_buffer.getvalue()