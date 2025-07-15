# =============================================================================
# SCRIPT PARA GENERAR VISUALIZACIONES DE ANÁLISIS EXPLORATORIO (EDA) - VERSIÓN MEJORADA
# =============================================================================
# Este script ha sido modificado para crear visualizaciones más detalladas y
# autoexplicativas, ideales para presentaciones e informes. Se han añadido
# anotaciones, subtítulos y un estilo visual mejorado.

print("--- Instalando y cargando librerías ---")
# Instalar las librerías necesarias de forma silenciosa
!pip install numpy pandas librosa matplotlib seaborn tqdm kaggle -q

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from google.colab import files
import kaggle

# --- 1. CONFIGURACIÓN Y DESCARGA DE DATOS (Sin cambios) ---

def setup_kaggle_api():
    """Configura la API de Kaggle para descargar datasets."""
    print("\n--- Configurando la API de Kaggle ---")
    if not os.path.exists("/root/.kaggle/kaggle.json"):
        print("Por favor, sube tu archivo 'kaggle.json'.")
        os.makedirs("/root/.kaggle", exist_ok=True)
        try:
            uploaded = files.upload()
            if "kaggle.json" in uploaded:
                os.rename("kaggle.json", "/root/.kaggle/kaggle.json")
                os.chmod("/root/.kaggle/kaggle.json", 600)
                print("\nArchivo 'kaggle.json' configurado.")
            else:
                print("\nADVERTENCIA: No se subió 'kaggle.json'. La descarga podría fallar.")
        except Exception as e:
            print(f"No se pudo subir el archivo: {e}")
    else:
        print("\nUsando el archivo 'kaggle.json' ya existente.")

def download_and_extract_dataset(dataset_id, path):
    """Descarga y descomprime un dataset de Kaggle si no existe."""
    if not os.path.exists(path):
        print(f"Descargando y descomprimiendo '{dataset_id}'...")
        try:
            kaggle.api.dataset_download_files(dataset_id, path=path, unzip=True, quiet=True)
            print(f"Dataset '{path}' listo.")
        except Exception as e:
            print(f"ERROR al descargar '{path}': {e}")
            return False
    else:
        print(f"El dataset '{path}' ya existe.")
    return True

# Configurar Kaggle y descargar datos
setup_kaggle_api()
dataset_path = "RAVDESS_DATA"
download_and_extract_dataset("uwrfkaggler/ravdess-emotional-speech-audio", dataset_path)


# --- 2. EXTRACCIÓN DE CARACTERÍSTICAS (Sin cambios) ---

ravdess_emotions_map = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
}

def extract_features_for_eda(file_path):
    """Extrae Pitch y MFCCs de un archivo de audio."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        # Extracción de Pitch (F0)
        pitch_track = librosa.pyin(y=y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))[0]
        pitch_track = pitch_track[~np.isnan(pitch_track)]
        mean_pitch = np.mean(pitch_track) if len(pitch_track) > 0 else 0
        # Extracción de MFCCs
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        return mean_pitch, mfccs
    except Exception as e:
        print(f"Error procesando {os.path.basename(file_path)}: {e}")
        return None, None

print("\n--- Procesando archivos de audio para extraer características ---")
audio_files = librosa.util.find_files(dataset_path, ext=['wav'])
data = []

for file in tqdm(audio_files, desc="Extrayendo Características"):
    basename = os.path.basename(file)
    try:
        emotion_code = int(basename.split('-')[2])
        emotion = ravdess_emotions_map.get(emotion_code)
        if emotion and emotion != 'calm':
            mean_pitch, mfccs = extract_features_for_eda(file)
            if mean_pitch is not None and mfccs is not None:
                for i, mfcc_val in enumerate(mfccs):
                    data.append([emotion, mean_pitch, f'MFCC_{i+1}', mfcc_val]) # MFCCs de 1 a 13
    except:
        continue

df = pd.DataFrame(data, columns=['emotion', 'pitch', 'mfcc_feature', 'mfcc_value'])
print(f"\nProcesamiento completado. Se generaron {len(df)} filas de datos.")


# --- 3. GENERACIÓN Y GUARDADO DE GRÁFICAS (VERSIÓN MEJORADA) ---

output_dir = "graficas_eda_mejoradas"
os.makedirs(output_dir, exist_ok=True)
print(f"\nLas gráficas mejoradas se guardarán en: '{output_dir}'")

# --- Gráfica 1: Distribución de Pitch con Anotaciones ---
print("Generando Gráfica 1: Histograma de Pitch Detallado...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(16, 9))

pitch_df = df[['emotion', 'pitch']].drop_duplicates()
emotions = sorted(pitch_df['emotion'].unique())
palette = sns.color_palette("viridis", n_colors=len(emotions))

# Título principal y subtítulo explicativo
fig.suptitle(
    'Análisis de Pitch: Distribución de Frecuencia Fundamental (F0) por Emoción',
    fontsize=22, weight='bold'
)
ax.set_title(
    'El pitch o tono de voz es un indicador clave del estado emocional. Valores más altos suelen asociarse con emociones de alta excitación (ej. sorpresa, felicidad).',
    fontsize=14, pad=20
)

# Graficar las distribuciones KDE
for i, emotion in enumerate(emotions):
    subset = pitch_df[pitch_df['emotion'] == emotion]
    sns.kdeplot(data=subset, x='pitch',
                label=emotion.capitalize(),
                color=palette[i],
                fill=True,
                alpha=0.5,
                linewidth=2,
                ax=ax)

    # Calcular y dibujar la media para cada emoción
    mean_pitch = subset['pitch'].mean()
    ax.axvline(x=mean_pitch, color=palette[i], linestyle='--', linewidth=1.5, alpha=0.8)
    # Añadir anotación de texto para la media
    ax.text(mean_pitch + 5, ax.get_ylim()[1] * (0.8 - i*0.07), f'Media ({emotion.capitalize()}): {mean_pitch:.0f} Hz',
            color=palette[i], weight='bold', ha='left')


ax.set_xlabel('Frecuencia Fundamental (Pitch) en Hertz (Hz)', fontsize=15, labelpad=10)
ax.set_ylabel('Densidad de Probabilidad', fontsize=15, labelpad=10)
ax.legend(title='Emoción', fontsize=12, title_fontsize='13', shadow=True, frameon=True, loc='upper right')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlim(left=50) # Ajustar el límite para una mejor visualización
plt.tight_layout(rect=[0, 0, 1, 0.95]) # Ajustar para que el suptitle no se solape

# Guardar la gráfica
pitch_hist_path = os.path.join(output_dir, "distribucion_pitch_detallada.png")
plt.savefig(pitch_hist_path, dpi=300, bbox_inches='tight')
print(f"Gráfica de Pitch guardada en: {pitch_hist_path}")
plt.show() # Mostrar la gráfica en Colab
plt.close()


# --- Gráfica 2: Gráficos de Violín para MFCCs con Mejoras ---
print("Generando Gráfica 2: Gráficos de Violín de MFCCs Detallados...")
fig, ax = plt.subplots(figsize=(20, 11))

# Paleta de colores
palette_mfcc = sns.color_palette("magma", n_colors=df['emotion'].nunique())

# Título principal y subtítulo explicativo
fig.suptitle(
    'Análisis Espectral: Distribución de los Primeros 13 MFCCs por Emoción',
    fontsize=22, weight='bold'
)
ax.set_title(
    'Los MFCCs describen la forma del espectro de una señal de audio, capturando características tímbricas únicas de la voz.',
    fontsize=14, pad=20
)

# Crear el gráfico de violín
sns.violinplot(x='mfcc_feature', y='mfcc_value', hue='emotion', data=df,
               palette=palette_mfcc, inner='quartile', ax=ax, cut=0,
               hue_order=sorted(df['emotion'].unique()),
               linewidth=1.5)

ax.set_xlabel('Coeficientes Cepstrales en la Frecuencia Mel (MFCCs)', fontsize=15, labelpad=10)
ax.set_ylabel('Rango de Valores del Coeficiente', fontsize=15, labelpad=10)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Mejorar la leyenda
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, [label.capitalize() for label in labels],
          title='Emoción', fontsize=12, title_fontsize='13',
          shadow=True, frameon=True, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.95]) # Ajustar para el suptitle

# Guardar la gráfica
mfcc_violin_path = os.path.join(output_dir, "distribucion_mfcc_detallada.png")
plt.savefig(mfcc_violin_path, dpi=300, bbox_inches='tight')
print(f"Gráfica de MFCCs guardada en: {mfcc_violin_path}")
plt.show() # Mostrar la gráfica en Colab
plt.close()

print("\n¡Proceso de generación de gráficas EDA mejoradas completado!")

