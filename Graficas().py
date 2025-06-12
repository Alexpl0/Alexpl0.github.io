# ...código de notebook...

import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from glob import glob
import kagglehub

# Configuración de carpetas para guardar gráficas
os.makedirs('graficas/pitch', exist_ok=True)
os.makedirs('graficas/mfcc', exist_ok=True)
os.makedirs('graficas/pca', exist_ok=True)

# Función para extraer pitch y MFCC de un archivo de audio
def extraer_caracteristicas(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    # Pitch (F0)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[magnitudes > np.median(magnitudes)].mean() if np.any(magnitudes > 0) else 0
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    return pitch, mfcc_mean

def verificar_o_descargar(nombre, kaggle_id):
    if not os.path.exists(nombre):
        print(f"Descargando {nombre} desde Kaggle...")
        path = kagglehub.dataset_download(kaggle_id)
    else:
        print(f"El dataset {nombre} ya existe, no se descarga de nuevo.")
        path = nombre
    return path

# Descarga o verificación de datasets
mesd_path = verificar_o_descargar("mexican-emotional-speech-database-mesd", "saurabhshahane/mexican-emotional-speech-database-mesd")
ravdess_path = verificar_o_descargar("ravdess-emotional-speech-audio", "uwrfkaggler/ravdess-emotional-speech-audio")
ser_path = verificar_o_descargar("speech-emotion-recognition-en", "dmitrybabko/speech-emotion-recognition-en")

# Función para cargar archivos y etiquetas (ajusta según estructura real de carpetas)
def cargar_dataset(base_path, emociones_map, ext="wav"):
    archivos = glob(os.path.join(base_path, "**", f"*.{ext}"), recursive=True)
    data = []
    for archivo in archivos:
        # Extraer etiqueta desde el nombre del archivo o carpeta (ajusta según dataset)
        nombre = os.path.basename(archivo).lower()
        for emo, clave in emociones_map.items():
            if clave in nombre:
                pitch, mfcc = extraer_caracteristicas(archivo)
                data.append({
                    "archivo": archivo,
                    "emocion": emo,
                    "pitch": pitch,
                    **{f"mfcc_{i+1}": mfcc[i] for i in range(13)}
                })
                break
    return pd.DataFrame(data)

# Mapas de emociones (ajusta según los nombres reales en los datasets)
mesd_emociones = {
    "alegría": "alegria",
    "tristeza": "tristeza",
    "enojo": "enojo",
    "miedo": "miedo",
    "sorpresa": "sorpresa",
    "disgusto": "disgusto",
    "neutralidad": "neutral"
}
ravdess_emociones = {
    "alegría": "happy",
    "tristeza": "sad",
    "enojo": "angry",
    "miedo": "fear",
    "sorpresa": "surprise",
    "disgusto": "disgust",
    "neutralidad": "neutral"
}
ser_emociones = {
    "alegría": "happy",
    "tristeza": "sad",
    "enojo": "angry",
    "miedo": "fear",
    "sorpresa": "surprise",
    "disgusto": "disgust",
    "neutralidad": "neutral"
}

# Cargar y unir datasets (esto puede tardar varios minutos)
df_mesd = cargar_dataset(mesd_path, mesd_emociones)
df_ravdess = cargar_dataset(ravdess_path, ravdess_emociones)
df_ser = cargar_dataset(ser_path, ser_emociones)
df = pd.concat([df_mesd, df_ravdess, df_ser], ignore_index=True)

# --- Gráfica 1: Histograma de Pitch por Emoción ---
plt.figure(figsize=(10, 6))
for emo in df['emocion'].unique():
    sns.kdeplot(df[df['emocion'] == emo]['pitch'], label=emo, fill=True)
plt.title("Distribución del Pitch Fundamental (Hz) Segmentada por Categoría Emocional")
plt.xlabel("Pitch fundamental (Hz)")
plt.ylabel("Densidad de probabilidad")
plt.legend()
plt.tight_layout()
plt.savefig("graficas/pitch/histograma_pitch_por_emocion.png")
plt.close()

# --- Gráfica 2: Mapa de Calor de Correlaciones MFCC-Emoción ---
# Convertir emociones a números
df['emocion_num'] = df['emocion'].astype('category').cat.codes
mfcc_cols = [f"mfcc_{i+1}" for i in range(13)]
corrs = []
for mfcc in mfcc_cols:
    corrs.append(df[[mfcc, 'emocion_num']].corr().iloc[0,1])
plt.figure(figsize=(8, 4))
sns.heatmap(np.array(corrs).reshape(1, -1), annot=True, cmap="coolwarm", cbar=True,
            xticklabels=mfcc_cols, yticklabels=["Correlación"])
plt.title("Matriz de Correlación entre Coeficientes MFCC y Categorías Emocionales")
plt.tight_layout()
plt.savefig("graficas/mfcc/heatmap_correlacion_mfcc_emocion.png")
plt.close()

# --- Gráfica 3: PCA de Características Acústicas ---
X = df[mfcc_cols].values
y = df['emocion'].values
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df['PC1'] = X_pca[:,0]
df['PC2'] = X_pca[:,1]
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="emocion", palette="tab10", alpha=0.7)
plt.title("Proyección PCA de Características Acústicas Coloreadas por Emoción")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
plt.legend()
plt.tight_layout()
plt.savefig("graficas/pca/pca_emociones.png")
plt.close()

print("¡Gráficas generadas y guardadas en las carpetas correspondientes!")