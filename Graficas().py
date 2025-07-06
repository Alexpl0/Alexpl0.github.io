# -*- coding: utf-8 -*-
"""
Script completo para el análisis de características de audio,
generación de gráficas exploratorias y de reducción de dimensionalidad.
"""

import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from glob import glob
import kagglehub

# --- Configuración y Carga de Datos ---

# Configuración de carpetas para guardar gráficas
os.makedirs('graficas/pitch', exist_ok=True)
os.makedirs('graficas/mfcc', exist_ok=True)
os.makedirs('graficas/pca', exist_ok=True)

def verificar_o_descargar(nombre, kaggle_id):
    """Verifica si un dataset existe, si no, lo descarga de Kaggle."""
    if not os.path.exists(nombre):
        print(f"Descargando {nombre} desde Kaggle...")
        # Esta función asume que tienes la API de Kaggle configurada
        try:
            path = kagglehub.dataset_download(kaggle_id, path=nombre)
            print(f"Descarga completada en: {path}")
        except Exception as e:
            print(f"Error al descargar {nombre}: {e}")
            print("Por favor, asegúrate de tener 'kaggle.json' configurado.")
            return None
    else:
        print(f"El dataset {nombre} ya existe, no se descarga de nuevo.")
        path = nombre
    return path

# Descarga o verificación de datasets
mesd_path = verificar_o_descargar("mexican-emotional-speech-database-mesd", "saurabhshahane/mexican-emotional-speech-database-mesd")
ravdess_path = verificar_o_descargar("ravdess-emotional-speech-audio", "uwrfkaggler/ravdess-emotional-speech-audio")
ser_path = verificar_o_descargar("speech-emotion-recognition-en", "dmitrybabko/speech-emotion-recognition-en")

# --- Extracción de Características ---

def extraer_caracteristicas(audio_path):
    """Extrae Pitch y MFCC de un archivo de audio."""
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        if len(y) == 0: return 0, np.zeros(13)
        # Pitch (F0)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = pitches[magnitudes > np.median(magnitudes)].mean() if np.any(magnitudes > 0) else 0
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)
        return pitch, mfcc_mean
    except Exception as e:
        print(f"Error procesando {audio_path}: {e}")
        return 0, np.zeros(13)

def cargar_dataset(base_path, emociones_map, ext="wav"):
    """Carga archivos de un dataset y extrae sus características y etiquetas."""
    if base_path is None:
        return pd.DataFrame() # Devuelve un DataFrame vacío si la ruta no es válida
    archivos = glob(os.path.join(base_path, "**", f"*.{ext}"), recursive=True)
    data = []
    print(f"Procesando {len(archivos)} archivos de {base_path}...")
    for archivo in archivos:
        nombre_base = os.path.basename(archivo).lower()
        emocion_encontrada = None
        for emo, clave in emociones_map.items():
            if clave in nombre_base:
                emocion_encontrada = emo
                break
        if emocion_encontrada:
            pitch, mfcc = extraer_caracteristicas(archivo)
            if pitch > 0: # Solo añadir si se pudo extraer algo
                data.append({
                    "archivo": archivo,
                    "emocion": emocion_encontrada,
                    "pitch": pitch,
                    **{f"mfcc_{i+1}": mfcc[i] for i in range(13)}
                })
    return pd.DataFrame(data)

# Mapas de emociones
mesd_emociones = {"alegría": "alegria", "tristeza": "tristeza", "enojo": "enojo", "miedo": "miedo", "sorpresa": "sorpresa", "disgusto": "disgusto", "neutralidad": "neutral"}
ravdess_emociones = {"alegría": "happy", "tristeza": "sad", "enojo": "angry", "miedo": "fear", "sorpresa": "surprise", "disgusto": "disgust", "neutralidad": "neutral"}
ser_emociones = {"alegría": "happy", "tristeza": "sad", "enojo": "angry", "miedo": "fear", "sorpresa": "surprise", "disgusto": "disgust", "neutralidad": "neutral"}

# Cargar y unir datasets
df_mesd = cargar_dataset(mesd_path, mesd_emociones)
df_ravdess = cargar_dataset(ravdess_path, ravdess_emociones)
df_ser = cargar_dataset(ser_path, ser_emociones)
df = pd.concat([df_mesd, df_ravdess, df_ser], ignore_index=True).dropna()

print(f"Dataset combinado creado con {len(df)} muestras.")
print("Distribución de emociones:\n", df['emocion'].value_counts())

# --- Análisis Exploratorio de Datos (EDA) y Gráficas ---

# Gráfica 1: Histograma de Pitch por Emoción
plt.figure(figsize=(12, 7))
sns.kdeplot(data=df, x='pitch', hue='emocion', fill=True, common_norm=False, palette="tab10")
plt.title("Distribución del Pitch Fundamental (Hz) por Emoción", fontsize=16)
plt.xlabel("Pitch (Hz)", fontsize=12)
plt.ylabel("Densidad", fontsize=12)
plt.legend(title="Emoción")
plt.tight_layout()
plt.savefig("graficas/pitch/histograma_pitch_por_emocion.png")
plt.close()
print("Gráfica de distribución de pitch guardada.")

# Gráfica 2: Mapa de Calor de Correlaciones MFCC
df_corr = df.copy()
df_corr['emocion_code'] = df_corr['emocion'].astype('category').cat.codes
mfcc_cols = [f"mfcc_{i+1}" for i in range(13)]
correlations = df_corr[mfcc_cols + ['emocion_code']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlations, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación de Características Acústicas", fontsize=16)
plt.tight_layout()
plt.savefig("graficas/mfcc/heatmap_correlacion_mfcc.png")
plt.close()
print("Gráfica de mapa de calor de correlaciones guardada.")

# --- Implementación Básica de Modelo ---

features = [f'mfcc_{i+1}' for i in range(13)] + ['pitch']
X = df[features]
y = df['emocion']

# 1. División de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2. Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Entrenamiento, predicción y evaluación de un modelo simple
simple_model = LogisticRegression(max_iter=1000, random_state=42)
simple_model.fit(X_train_scaled, y_train)
y_pred = simple_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Implementación Básica del Modelo ---")
print(f"Shape de X_train: {X_train.shape}, Shape de X_test: {X_test.shape}")
print(f"Accuracy del modelo de Regresión Logística: {accuracy:.2f}")

# --- Reducción de Dimensionalidad y Visualización ---

# Escalado de todos los datos para visualización
X_scaled_full = scaler.fit_transform(X)

# Gráfica 3: PCA de Características Acústicas
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_full)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="emocion", palette="tab10", alpha=0.7)
plt.title("Proyección PCA de Características Acústicas", fontsize=16)
plt.xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)", fontsize=12)
plt.ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)", fontsize=12)
plt.legend(title="Emoción")
plt.tight_layout()
plt.savefig("graficas/pca/pca_emociones.png")
plt.close()
print("Gráfica de PCA guardada.")

# Gráfica 4: LDA de Características Acústicas
n_components_lda = len(df['emocion'].unique()) - 1
lda = LDA(n_components=n_components_lda)
X_lda = lda.fit_transform(X_scaled_full, y)

df_lda = pd.DataFrame(data=X_lda, columns=[f'LD{i+1}' for i in range(n_components_lda)])
df_lda['emocion'] = y.values

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_lda, x="LD1", y="LD2", hue="emocion", palette="tab10", alpha=0.8)
plt.title("Proyección LDA de Características Acústicas", fontsize=16)
plt.xlabel("Componente Discriminante Lineal 1", fontsize=12)
plt.ylabel("Componente Discriminante Lineal 2", fontsize=12)
plt.legend(title="Emoción")
plt.tight_layout()
plt.savefig("graficas/pca/lda_emociones.png")
plt.close()
print("Gráfica de LDA guardada.")

print("\n¡Proceso completado! Todas las gráficas han sido generadas y guardadas.")
