# processor.py

import re
import nltk
import asyncio
import numpy as np
import nest_asyncio
import pandas as pd
import streamlit as st # Necesario para st.error
from io import StringIO # Para manejar el buffer de archivos subidos

from googletrans import Translator
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# --- INICIALIZACIÓN Y DESCARGA DE NLTK DATA ---
try:
    # Intenta verificar si el recurso VADER ya está instalado
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    # Si no está, lo descarga (esto se ejecuta al inicio en Streamlit Cloud)
    print("Descargando Vader Lexicon para análisis de sentimiento...")
    nltk.download('vader_lexicon')
# ---------------------------------------------


# Diccionario de delitos (Parte esencial del preprocesamiento)
CRIMENES_KEYWORDS = {
    "Homicidio": [
        "asesinan", "asesinado", "mataron", "ejecutan", "ejecutado", "asesina", "encuentran cuerpo", 
        "hallan cadáver", "sin vida", "balacera", "disparan contra", "asesinato", "homicidio", 
        "ultimaron", "asesinados", "baleado", "balearon", "acribillado", "sicarios", 
        "hallan muerto", "tiroteo", "ataque armado", "impactos de bala", "#Homicidio"
    ],
    "Robo": [
        "roba", "robaron", "asaltan", "asaltaron", "me robaron", "robo", "asalto", "atraco", 
        "atracaron", "robo con violencia", "cristalazo", "ratero", "rateros", "ladrón", 
        "ladrones", "me quitaron", "se llevaron mi", "robo en metro", "robo en metrobus",
        "despojada", "despojado", "me encañonaron", "con arma", "armados", "roban", 
        "asaltó", "#robo", "#asalto"
    ],
    "Feminicidio": ["Asesinada", "feminicidio", "matan a mujer", "asesinan mujer", "feminicida", "la mató", "mujer asesinada", "hallan mujer", "encuentran mujer sin vida", "asesinada", "fue asesinada", "violencia de género", "#Feminicidio", "#NiUnaMenos", "la asesinaron", "la mataron", "muerta", "cadáver de mujer", "#JusticiaPara", "víctima de feminicidio", "estrangulada"],
    "Secuestro": ["secuestran", "secuestraron", "secuestro", "secuestro de", "levantón", "levantaron", "secuestrado", "se lo llevaron", "privación de libertad", "desaparecido", "secuestró", "secuestrada", "desaparecida", "rescate", "piden rescate", "fue secuestrado", "secuestrado", "#Secuestro"],
    "Extorsión": ["extorsión", "extorsionan", "cobro de piso", "extorsionado", "amenaza", "amenazas", "amenazan", "llamada extorsiva", "falso secuestro", "extorsionando", "secuestro virtual", "fraude telefónico", "amenazaron", "#Extorsion"],
    "Violación": ["violación", "violaron", "abuso sexual", "abusaron", "violado", "agresión sexual", "acoso sexual", "tocamientos", "#MeToo", "denuncia por violación", "#AbusoSexual", "violó", "abuso", "maltrato", "AGRESIÓN SEXUAL", "agresion sexual", "agresiones sexuales", "agresor"],
    "Vandalismo": ["vandalizan", "vandalizaron", "destrozos", "grafiti", "rayan", "vandalizando", "#Vandalismo", "rayaron", "destruyeron", "quemaron", "incendiaron", "rayando", "queman", "quemó", "#DañoaPropiedadPrivada"]
}


# --- 2. FUNCIONES DE PREPROCESAMIENTO ---

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+"
    )
    return emoji_pattern.sub(r'', text)

def normalize_censored_text(text, keywords_dict):
    all_keywords_lower = set()
    for category_keywords in keywords_dict.values():
        for kw in category_keywords:
            all_keywords_lower.add(kw.lower())
    sorted_clean_keywords = sorted(list(all_keywords_lower), key=len, reverse=True)
    numeric_substitutions = {
        'a': '4', 'e': '3', 'i': '1', 'o': '0',
        's': '5', 'l': '1', 't': '7',
    }
    modified_text = text
    for clean_kw in sorted_clean_keywords:
        pattern_parts = []
        for i, char in enumerate(clean_kw):
            char_lower = char.lower()
            char_options_set = {char.lower(), char.upper()}
            if char_lower in numeric_substitutions:
                char_options_set.add(numeric_substitutions[char_lower])
            char_options_set.update({'*', 'x', 'X', '_'})
            char_class_str = ''.join(sorted(list(char_options_set)))
            char_pattern_segment = f"[{char_class_str}]"
            pattern_parts.append(char_pattern_segment)
            if i < len(clean_kw) - 1:
                pattern_parts.append(r'[*xX_\d]*?')
            final_regex_pattern = "".join(pattern_parts)
            if final_regex_pattern:
                modified_text = re.sub(
                    rf'(?:(?<=\W)|(?<=^)){final_regex_pattern}(?:(?=\W)|(?=$))',
                    clean_kw,
                    modified_text,
                    flags=re.IGNORECASE
                )
    return modified_text

def formateo_texto(text):
    text_without_emojis = remove_emojis(text)
    text_lower = text_without_emojis.lower()
    normalized_text = normalize_censored_text(text_lower, CRIMENES_KEYWORDS)
    return normalized_text

def clean_and_format_dataframe(df):
    df.loc[:, 'text'] = df['text'].apply(lambda x: formateo_texto(x))
    df = df.drop_duplicates(subset='text')
    df = df.rename(columns={'categorias': 'categorias'})
    df = df[~df['text'].str.contains(r'#\w+', na=False)]
    return df

def apply_tfidf_vectorization(df):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
    df.loc[:, 'textotfidf'] = pd.Series([vec for vec in tfidf_matrix.toarray()], index=df.index)
    return df

def populate_categories_from_keywords(df):
    for category, keywords in CRIMENES_KEYWORDS.items():
        pattern = r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b'
        df.loc[
            df['text'].str.contains(pattern, case=False, na=False) &
            df['categorias'].isna(),
            'categorias'
        ] = category
    return df

def preprocess_features(df):
    # Asegura que las columnas esenciales existan
    if 'text' not in df.columns or 'public_metrics.impression_count' not in df.columns:
        st.error("El CSV debe contener las columnas 'text' y 'public_metrics.impression_count'.")
        return None

    df = populate_categories_from_keywords(df.copy())
    df['categorias'] = df['categorias'].fillna('desconocido')
    
    # Vectorización TF-IDF
    df = apply_tfidf_vectorization(df)
    
    df_selected = df[['text', 'textotfidf', 'public_metrics.impression_count', 'categorias']].copy()
    return df_selected


# --- 3. CLASIFICACIÓN (DECISION TREE) ---

def split_data_for_training(df):
    X_all = np.vstack(df['textotfidf'].values)
    known_category_mask = (df['categorias'] != 'desconocido')
    
    # Manejar el caso donde no hay suficientes categorías conocidas para entrenar
    if known_category_mask.sum() == 0:
        st.warning("No hay suficientes datos etiquetados para entrenar el clasificador.")
        return None, None, X_all
        
    X_filtered_for_training = np.vstack(df.loc[known_category_mask, 'textotfidf'].values)
    y_filtered_for_training = df.loc[known_category_mask, 'categorias']
    
    # División de datos (solo necesitamos X_train y y_train para entrenar el modelo)
    X_train, _, y_train, _ = train_test_split(
        X_filtered_for_training, y_filtered_for_training, test_size=0.2, random_state=42
    )
    return X_train, y_train, X_all

def train_and_predict_decision_tree(X_train, y_train, X_all, df_to_update):
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    
    all_predictions = dt_classifier.predict(X_all)
    df_to_update.loc[:, 'prediction'] = all_predictions
    
    # Rellenar 'desconocido' con la predicción
    df_to_update.loc[df_to_update['categorias'] == 'desconocido', 'categorias'] = df_to_update['prediction']
    
    return df_to_update.drop(columns=['prediction'])

def run_decision_tree_process(df):
    X_train, y_train, X_all = split_data_for_training(df.copy())
    
    if X_train is None:
        return df # Retorna el DataFrame sin clasificar si no hay datos para entrenar

    df_updated = train_and_predict_decision_tree(X_train, y_train, X_all, df.copy())
    return df_updated


# --- 4. ANÁLISIS DE SENTIMIENTO (VADER) ---

async def _translate_task_async(text):
    """Tarea asíncrona para la traducción."""
    local_translator = Translator()
    # Usamos try/except para manejar posibles fallos de conexión o límites de API
    try:
        translation = await local_translator.translate(text, src='es', dest='en')
        return translation.text
    except Exception:
        return text # Devolver original si falla la traducción

def translate_to_english(text):
    """Ejecuta la traducción de forma síncrona usando nest_asyncio."""
    if isinstance(text, str) and len(text.strip()) > 0:
        nest_asyncio.apply()
        translated_text = asyncio.run(_translate_task_async(text))
        return translated_text
    return text

def perform_sentiment_analysis(df):
    sid = SentimentIntensityAnalyzer()
    
    def get_sentiment_scores(text):
        if isinstance(text, str):
            return sid.polarity_scores(text) 
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    
    df.loc[:, 'sentiment_scores'] = df['text_en'].apply(get_sentiment_scores)
    df.loc[:, 'sentiment_compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])
    
    df.loc[:, 'sentiment_label'] = df['sentiment_compound'].apply(
        lambda score: 'Positive' if score >= 0.05 else ('Negative' if score <= -0.05 else 'Neutral')
    )
    return df

def run_sentiment_analysis_process(df):
    # Aseguramos que la columna 'text_en' exista, traduciendo solo si es necesario
    if 'text_en' not in df.columns:
        df['text_en'] = df['text'].apply(translate_to_english) 
    
    df = perform_sentiment_analysis(df.copy())
    return df


# --- 5. CLUSTERING (K-MEANS) ---

def prepare_features_for_clustering(df):
    # Transformación logarítmica a impresiones
    df['impresiones_log'] = np.log1p(df['public_metrics.impression_count'])
    
    # One-hot encoding de categorías
    df_features_procesado = pd.get_dummies(df, columns=['categorias'], prefix='cat')
    
    # Escalamiento (StandardScaler)
    continuous_cols = ['sentiment_compound', 'impresiones_log']
    binary_cols = [col for col in df_features_procesado.columns if col.startswith('cat_')]
    
    scaler = StandardScaler()
    X_scaled_continuous = scaler.fit_transform(df_features_procesado[continuous_cols])
    X_scaled_continuous_df = pd.DataFrame(X_scaled_continuous, columns=continuous_cols, index=df_features_procesado.index)
    X_binary_df = df_features_procesado[binary_cols]
    
    X_for_kmeans = pd.concat([X_scaled_continuous_df, X_binary_df], axis=1)
    return X_for_kmeans

def perform_kmeans_clustering(df_to_cluster, X_for_kmeans, k=5):
    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_model.fit(X_for_kmeans)
    df_to_cluster.loc[:, 'cluster_label'] = kmeans_model.labels_
    df_to_cluster['cluster_label'] = df_to_cluster['cluster_label'].astype(str)
    return df_to_cluster

def run_full_clustering_process(df_filtered_data):
    df_temp = df_filtered_data.copy()
    
    # Preparar features
    X_for_kmeans = prepare_features_for_clustering(df_temp)
    
    # Ejecutar clustering
    df_final_clusters = perform_kmeans_clustering(df_temp, X_for_kmeans, k=5)
    
    return df_final_clusters


# --- 6. FUNCIÓN DE EXPORTACIÓN Y PIPELINE PRINCIPAL ---

def create_gauge_data(df_final_clusters):
    """Genera el DataFrame con el sentimiento promedio por categoría (df_Gauge)."""
    
    sentiment_por_categoria = df_final_clusters.groupby('categorias')['sentiment_compound'].mean().reset_index()
    
    def classify_sentiment_label(score):
        if score >= 0.05:
            return 'Positivo'
        elif score <= -0.05:
            return 'Negativo'
        return 'Neutro'
        
    sentiment_por_categoria['sentiment_label'] = sentiment_por_categoria['sentiment_compound'].apply(classify_sentiment_label)
    
    sentiment_por_categoria = sentiment_por_categoria.rename(columns={'categorias': 'categorias'})
    
    return sentiment_por_categoria


def process_uploaded_csv(uploaded_file_buffer):
    """
    Función principal que ejecuta todo el pipeline de procesamiento de texto.
    """
    
    # 1. Cargar el DataFrame desde el buffer
    try:
        # Usamos el buffer (io.BytesIO) que viene de st.file_uploader
        df_initial = pd.read_csv(uploaded_file_buffer)
        
        # Necesitamos una columna 'categorias' inicializada, si no existe
        if 'categorias' not in df_initial.columns:
             df_initial['categorias'] = pd.NA
             
    except Exception as e:
        # st.error() solo se usa para notificar el error en la interfaz de Streamlit
        st.error(f"Error al leer el archivo CSV. Asegúrese de que contenga datos válidos y las columnas requeridas (text, public_metrics.impression_count). Error: {e}")
        return None, None

    # --- PIPELINE ---
    
    # A. Limpieza y Formateo
    df_cleaned = clean_and_format_dataframe(df_initial.copy())
    
    # B. Preprocesamiento de Features (TF-IDF y llenado de categorías con keywords)
    df_preprocessed = preprocess_features(df_cleaned.copy())
    
    if df_preprocessed is None:
        return None, None
    
    # C. Clasificación (Decision Tree para rellenar 'desconocido')
    df_classified = run_decision_tree_process(df_preprocessed.copy())
    
    # D. Análisis de Sentimiento (Traducción + VADER)
    df_sentiment = run_sentiment_analysis_process(df_classified.copy())
    
    # E. Clustering (K-Means)
    df_final_clusters = run_full_clustering_process(df_sentiment.copy())

    # --- GENERACIÓN DE RESULTADOS ---
    
    # Resultado 1: Dataframe de Clusters (para descarga "cluster")
    df_clusters_export = df_final_clusters[[
        'sentiment_compound', 
        'categorias', 
        'impresiones_log', 
        'cluster_label'
    ]].copy()
    
    # Resultado 2: Dataframe de Sentimiento por Categoría (para descarga "sentiment")
    df_gauge_data = create_gauge_data(df_final_clusters)

    return df_clusters_export, df_gauge_data