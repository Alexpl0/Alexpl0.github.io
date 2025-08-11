# =============================================================================
# CELDA 1: INSTALACIÓN Y CONFIGURACIÓN
# =============================================================================
print("--- Instalando librerías necesarias ---")
# Se instalan todas las librerías necesarias al principio.
# NOTA: Se removió 'resampy' ya que no es necesaria para este proyecto
!pip install kaggle plotly pandas numpy scikit-learn librosa tensorflow -q

# Instalar numba específicamente para evitar problemas con librosa
!pip install numba -q

print("Librerías instaladas.")

# Importamos solo lo necesario para la configuración.
import os
import json
from google.colab import files # type: ignore

# Configuración de la API de Kaggle usando variables de entorno
print("\n--- Configurando la API de Kaggle ---")
# Solo pedimos subir si no existe en la sesión actual de Colab
if not os.path.exists("/content/kaggle.json"):
    print("Por favor, sube tu archivo 'kaggle.json'.")
    uploaded = files.upload()
    if "kaggle.json" in uploaded:
        print("\nArchivo 'kaggle.json' recibido.")
    else:
        print("\nADVERTENCIA: No se subió 'kaggle.json'. La descarga fallará.")
else:
    print("\nUsando el archivo 'kaggle.json' ya existente.")

# Leemos el archivo y configuramos las variables de entorno
try:
    with open('kaggle.json', 'r') as f:
        credentials = json.load(f)
    os.environ['KAGGLE_USERNAME'] = credentials['username']
    os.environ['KAGGLE_KEY'] = credentials['key']
    print("\n¡Variables de entorno de Kaggle configuradas exitosamente!")
except FileNotFoundError:
    print("\nERROR: No se encontró 'kaggle.json'. Por favor, ejecuta la celda de nuevo y sube el archivo.")
except Exception as e:
    print(f"\nERROR al procesar 'kaggle.json': {e}")
# =============================================================================
# CELDA 2: IMPORTACIONES PRINCIPALES Y DEFINICIÓN DE FUNCIONES
# =============================================================================
print("--- Importando librerías principales y definiendo funciones ---")
# Ahora que la configuración está lista, importamos el resto de las librerías.
import librosa
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from glob import glob

import kaggle # La importación de Kaggle se hace aquí, de forma segura.
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

def verificar_o_descargar_api(nombre_dir, kaggle_id):
    """Descarga y descomprime un dataset usando la API de Kaggle."""
    if not os.path.exists(nombre_dir):
        print(f"Descargando y descomprimiendo '{nombre_dir}'...")
        try:
            kaggle.api.authenticate() # La autenticación usará las variables de entorno.
            kaggle.api.dataset_download_files(kaggle_id, path=nombre_dir, unzip=True)
            print(f"Dataset '{nombre_dir}' listo.")
        except Exception as e:
            print(f"ERROR al descargar '{nombre_dir}': {e}")
            return None
    else:
        print(f"El dataset '{nombre_dir}' ya existe.")
    return nombre_dir

def extract_features(file_name):
    """Extrae un conjunto de 180 características de un archivo de audio."""
    try:
        # Cambio importante: especificar sr=None para usar la frecuencia original
        # y cambiar res_type a 'scipy' que es más estable
        audio, sample_rate = librosa.load(file_name, sr=None, res_type='scipy')

        # Extraer características MFCC
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)

        # Extraer características Chroma
        stft = np.abs(librosa.stft(audio))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

        # Extraer características Mel-spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)

        return np.hstack((mfccs, chroma, mel))
    except Exception as e:
        print(f"Error procesando {os.path.basename(file_name)}: {e}")
        return None

def cargar_y_procesar(dataset_name, base_path, emociones_map):
    """Orquesta la carga y procesamiento de un dataset."""
    if base_path is None:
        return pd.DataFrame()

    search_path = os.path.join(base_path, "**", "*.wav")
    archivos = glob(search_path, recursive=True)

    if not archivos:
        return pd.DataFrame()

    data = []
    print(f"Procesando {len(archivos)} archivos de '{dataset_name}'...")

    for archivo in archivos:
        nombre_base = os.path.basename(archivo)
        emocion = None

        if dataset_name == "RAVDESS":
            try:
                emotion_code = int(nombre_base.split('-')[2])
                emocion = emociones_map.get(emotion_code)
            except:
                continue

        if emocion:
            features = extract_features(archivo)
            if features is not None:
                data.append([features, emocion])

    return pd.DataFrame(data, columns=['features', 'emotion'])

print("Funciones listas para ser usadas.")
# =============================================================================
# CELDA 3: EJECUCIÓN PRINCIPAL
# =============================================================================
print("\n--- Iniciando el proceso principal ---")

# Descargar datasets
ravdess_path = verificar_o_descargar_api("RAVDESS_DATA", "uwrfkaggler/ravdess-emotional-speech-audio")

# Mapa de emociones para RAVDESS
ravdess_emociones_map = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
}

# Cargar y procesar el dataset principal
df = cargar_y_procesar("RAVDESS", ravdess_path, ravdess_emociones_map)
df = df[df['emotion'] != 'calm'] # Filtrar emociones no deseadas

if df.empty:
    print("\nERROR CRÍTICO: El DataFrame está vacío.")
else:
    print(f"\nDataset final creado con {len(df)} muestras.")
    print("Distribución de emociones:\n", df['emotion'].value_counts())

    # Preparación de datos
    X = np.array(df['features'].tolist())
    y = np.array(df['emotion'].tolist())

    encoder = OneHotEncoder()
    y_encoded = encoder.fit_transform(y.reshape(-1, 1)).toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_cnn = np.expand_dims(X_train_scaled, axis=2)
    X_test_cnn = np.expand_dims(X_test_scaled, axis=2)

    # Definición y entrenamiento del modelo
    print("\n--- Definiendo y Entrenando el Modelo CNN 1D ---")
    model = Sequential([
        Conv1D(256, 5, padding='same', activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
        MaxPooling1D(pool_size=5),
        Dropout(0.2),
        Conv1D(128, 5, padding='same', activation='relu'),
        MaxPooling1D(pool_size=5),
        Dropout(0.2),
        Flatten(),
        Dense(y_encoded.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train_cnn, y_train, epochs=100, batch_size=64,
        validation_split=0.2, callbacks=[early_stopping]
    )

    loss, accuracy = model.evaluate(X_test_cnn, y_test)
    print(f"\nPrecisión del modelo en datos de prueba: {accuracy*100:.2f}%")

    # Generación de gráficas 3D
    print("\n--- Generando Gráficas 3D ---")
    X_for_viz = scaler.transform(X)
    y_for_viz = y

    output_dir = "graficas/plots_3d"
    os.makedirs(output_dir, exist_ok=True)

    # PCA 3D
    pca = PCA(n_components=3)
    X_pca_3d = pca.fit_transform(X_for_viz)
    df_pca_3d = pd.DataFrame(X_pca_3d, columns=['PC1', 'PC2', 'PC3'])
    df_pca_3d['emocion'] = y_for_viz

    fig_pca = px.scatter_3d(
        df_pca_3d, x='PC1', y='PC2', z='PC3',
        color='emocion', title='Visualización 3D de PCA'
    )
    fig_pca.write_html(os.path.join(output_dir, "pca_3d_plot.html"))
    print(f"Gráfica 3D de PCA guardada en: {os.path.join(output_dir, 'pca_3d_plot.html')}")
# 
    # LDA 3D
    n_components_lda = min(len(np.unique(y_for_viz)) - 1, X_for_viz.shape[1])
    if n_components_lda >= 3:
        lda = LDA(n_components=3)
        X_lda_3d = lda.fit_transform(X_for_viz, y_for_viz)
        df_lda_3d = pd.DataFrame(X_lda_3d, columns=['LD1', 'LD2', 'LD3'])
        df_lda_3d['emocion'] = y_for_viz

        fig_lda = px.scatter_3d(
            df_lda_3d, x='LD1', y='LD2', z='LD3',
            color='emocion', title='Visualización 3D de LDA'
        )
        fig_lda.write_html(os.path.join(output_dir, "lda_3d_plot.html"))
        print(f"Gráfica 3D de LDA guardada en: {os.path.join(output_dir, 'lda_3d_plot.html')}")
    else:
        print(f"LDA no puede generar 3 componentes (solo {n_components_lda}). Omitiendo gráfico 3D de LDA.")

    print("\n¡Proceso completado!")
