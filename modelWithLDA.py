# =============================================================================
# ENTRENAMIENTO AVANZADO DE CNN 1D CON LDA PARA RECONOCIMIENTO DE EMOCIONES EN VOZ
# Versión corregida y optimizada para Google Colab con análisis LDA
# =============================================================================

print("🚀 Iniciando entrenamiento avanzado de CNN 1D con LDA para reconocimiento emocional")
print("=" * 80)

# =============================================================================
# PARTE 1: INSTALACIÓN Y CONFIGURACIÓN AVANZADA (CORREGIDA)
# =============================================================================

print("📦 Instalando dependencias optimizadas para GPU...")
# CORRECCIÓN: Agregar resampy y otras dependencias faltantes
!pip install -q kaggle librosa resampy soundfile numpy pandas scikit-learn tensorflow plotly seaborn tqdm
!pip install -q tensorboard numba==0.56.4

# Verificar GPU
import tensorflow as tf
print(f"🔧 TensorFlow versión: {tf.__version__}")
print(f"🎮 GPU disponible: {tf.config.list_physical_devices('GPU')}")

# Configurar GPU para crecimiento dinámico de memoria
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Configuración de GPU optimizada")
        print(f"💾 GPU: {tf.test.gpu_device_name()}")
    except RuntimeError as e:
        print(f"⚠️ Error configurando GPU: {e}")
else:
    print("⚠️ No se detectó GPU, usando CPU")

# Importaciones principales
import os
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Dropout, Dense,
                                   BatchNormalization, GlobalAveragePooling1D, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                       ModelCheckpoint, TensorBoard)
from tensorflow.keras.utils import to_categorical
from google.colab import files
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

# Verificar instalación de resampy
try:
    import resampy
    print("✅ resampy instalado correctamente")
except ImportError:
    print("❌ Error: resampy no está instalado")
    !pip install resampy
    import resampy
    print("✅ resampy instalado y importado")

# =============================================================================
# PARTE 2: CONFIGURACIÓN DE KAGGLE Y DESCARGA DE DATASETS
# =============================================================================

print("\n📂 Configurando acceso a Kaggle...")

def setup_kaggle_credentials():
    """Configura las credenciales de Kaggle para acceso a datasets"""
    kaggle_dir = "/root/.kaggle"
    kaggle_file = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(kaggle_file):
        print("🔑 Por favor, sube tu archivo kaggle.json:")
        uploaded = files.upload()

        if "kaggle.json" in uploaded:
            os.makedirs(kaggle_dir, exist_ok=True)
            with open(kaggle_file, "w") as f:
                f.write(uploaded["kaggle.json"].decode())
            os.chmod(kaggle_file, 0o600)
            print("✅ Credenciales de Kaggle configuradas")
        else:
            raise Exception("❌ Archivo kaggle.json no encontrado")
    else:
        print("✅ Credenciales de Kaggle ya configuradas")

def download_datasets():
    """Descarga todos los datasets necesarios"""
    datasets = {
        "RAVDESS": "uwrfkaggler/ravdess-emotional-speech-audio",
        "TESS": "ejlok1/toronto-emotional-speech-set-tess",
        "MESD": "ejlok1/mexican-emotional-speech-database"
    }

    downloaded_paths = {}

    for name, dataset_id in datasets.items():
        path = f"/content/{name}_data"

        if not os.path.exists(path):
            print(f"⬇️ Descargando {name}...")
            try:
                os.system(f"kaggle datasets download -d {dataset_id} -p {path} --unzip -q")
                print(f"✅ {name} descargado exitosamente")
                downloaded_paths[name] = path
            except Exception as e:
                print(f"⚠️ Error descargando {name}: {e}")
                downloaded_paths[name] = None
        else:
            print(f"✅ {name} ya existe")
            downloaded_paths[name] = path

    return downloaded_paths

# Configurar y descargar datasets
setup_kaggle_credentials()
dataset_paths = download_datasets()

# =============================================================================
# PARTE 3: EXTRACCIÓN AVANZADA DE CARACTERÍSTICAS (CORREGIDA)
# =============================================================================

print("\n🎵 Configurando extracción avanzada de características...")

class AdvancedFeatureExtractor:
    """Extractor avanzado de características de audio para reconocimiento emocional"""

    def __init__(self,
                 sr=22050,
                 n_mfcc=40,
                 n_chroma=12,
                 n_mel=128,
                 hop_length=512,
                 win_length=2048):

        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_mel = n_mel
        self.hop_length = hop_length
        self.win_length = win_length

        print(f"🔧 Configuración del extractor:")
        print(f"   - Frecuencia de muestreo: {sr} Hz")
        print(f"   - MFCCs: {n_mfcc}")
        print(f"   - Características Chroma: {n_chroma}")
        print(f"   - Mel-spectrogram bins: {n_mel}")

    def extract_features(self, audio_path):
        """Extrae características completas de un archivo de audio"""
        try:
            # CORRECCIÓN: Usar res_type='soxr_hq' para mejor compatibilidad
            # o 'polyphase' si 'soxr_hq' no está disponible
            try:
                # Intentar con soxr_hq primero
                audio, sr = librosa.load(audio_path, sr=self.sr, res_type='soxr_hq')
            except:
                try:
                    # Fallback a polyphase
                    audio, sr = librosa.load(audio_path, sr=self.sr, res_type='polyphase')
                except:
                    # Último recurso: usar scipy
                    audio, sr = librosa.load(audio_path, sr=self.sr, res_type='scipy')

            # Verificar que el audio no esté vacío
            if len(audio) == 0:
                print(f"⚠️ Archivo de audio vacío: {os.path.basename(audio_path)}")
                return None

            # Normalizar audio
            audio = librosa.util.normalize(audio)

            # Asegurar longitud mínima
            min_length = self.hop_length * 2
            if len(audio) < min_length:
                audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')

            # 1. MFCCs (40 características)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc,
                                       hop_length=self.hop_length,
                                       win_length=self.win_length)
            mfccs_mean = np.mean(mfccs.T, axis=0)

            # 2. Características Chroma (12 características)
            stft = np.abs(librosa.stft(audio, hop_length=self.hop_length,
                                     win_length=self.win_length))
            chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
            chroma_mean = np.mean(chroma.T, axis=0)

            # 3. Mel-spectrogram (128 características)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mel,
                                                    hop_length=self.hop_length,
                                                    win_length=self.win_length)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_mean = np.mean(mel_spec_db.T, axis=0)

            # Verificar que no haya NaN o valores infinitos
            mfccs_mean = np.nan_to_num(mfccs_mean, nan=0.0, posinf=0.0, neginf=0.0)
            chroma_mean = np.nan_to_num(chroma_mean, nan=0.0, posinf=0.0, neginf=0.0)
            mel_mean = np.nan_to_num(mel_mean, nan=0.0, posinf=0.0, neginf=0.0)

            # Concatenar todas las características (180 total)
            features = np.hstack([mfccs_mean, chroma_mean, mel_mean])

            assert len(features) == 180, f"Error: {len(features)} características esperadas 180"

            return features

        except Exception as e:
            print(f"❌ Error procesando {os.path.basename(audio_path)}: {e}")
            return None

    def batch_extract(self, audio_files, emotions, dataset_name):
        """Extrae características de múltiples archivos con barra de progreso"""
        print(f"\n🎯 Procesando {len(audio_files)} archivos de {dataset_name}...")

        features_list = []
        emotions_list = []
        failed_files = []

        for audio_file, emotion in tqdm(zip(audio_files, emotions),
                                      total=len(audio_files),
                                      desc=f"Procesando {dataset_name}"):
            features = self.extract_features(audio_file)
            if features is not None:
                features_list.append(features)
                emotions_list.append(emotion)
            else:
                failed_files.append(audio_file)

        print(f"✅ Procesados {len(features_list)}/{len(audio_files)} archivos exitosamente")
        if failed_files:
            print(f"⚠️ {len(failed_files)} archivos fallaron")

        return np.array(features_list), np.array(emotions_list)

# =============================================================================
# PARTE 4: CARGA Y PROCESAMIENTO DE DATASETS (MEJORADA)
# =============================================================================

def load_dataset_files(dataset_path, dataset_name):
    """Carga archivos de audio y extrae etiquetas de emociones"""
    if dataset_path is None or not os.path.exists(dataset_path):
        print(f"⚠️ {dataset_name} no disponible")
        return [], []

    audio_files = []
    emotions = []

    # Buscar archivos de audio recursivamente
    valid_extensions = ['.wav', '.mp3', '.flac', '.ogg']

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                file_path = os.path.join(root, file)

                # Verificar que el archivo existe y no está corrupto
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    emotion = extract_emotion_from_filename(file, dataset_name)
                    if emotion:
                        audio_files.append(file_path)
                        emotions.append(emotion)

    print(f"📊 {dataset_name}: {len(audio_files)} archivos encontrados")
    if emotions:
        emotion_counts = pd.Series(emotions).value_counts()
        print(f"   Distribución: {dict(emotion_counts)}")

    return audio_files, emotions

def extract_emotion_from_filename(filename, dataset_name):
    """Extrae la emoción del nombre del archivo según el dataset"""
    filename_lower = filename.lower()

    if dataset_name == "RAVDESS":
        try:
            # Formato: 03-01-06-01-02-01-12.wav
            parts = filename.split('-')
            if len(parts) >= 3:
                emotion_code = int(parts[2])
                emotion_map = {
                    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
                }
                emotion = emotion_map.get(emotion_code)
                # Excluir 'calm' para simplificar
                return emotion if emotion and emotion != 'calm' else None
        except:
            return None

    elif dataset_name == "TESS":
        # Buscar emoción en el nombre del archivo
        emotion_keywords = {
            'angry': 'angry',
            'disgust': 'disgust',
            'fear': 'fearful',
            'happy': 'happy',
            'neutral': 'neutral',
            'sad': 'sad',
            'surprise': 'surprised'
        }

        for keyword, emotion in emotion_keywords.items():
            if keyword in filename_lower:
                return emotion

    elif dataset_name == "MESD":
        # Para MESD, buscar patrones comunes
        emotion_patterns = {
            'angry': 'angry',
            'enojo': 'angry',
            'feliz': 'happy',
            'happy': 'happy',
            'triste': 'sad',
            'sad': 'sad',
            'neutral': 'neutral',
            'miedo': 'fearful',
            'fear': 'fearful',
            'sorpresa': 'surprised',
            'surprise': 'surprised'
        }

        for pattern, emotion in emotion_patterns.items():
            if pattern in filename_lower:
                return emotion

    return None

# Cargar todos los datasets
print("\n📁 Cargando archivos de datasets...")
all_audio_files = []
all_emotions = []

for dataset_name, path in dataset_paths.items():
    files, emotions = load_dataset_files(path, dataset_name)
    all_audio_files.extend(files)
    all_emotions.extend(emotions)

print(f"\n📈 Total de archivos cargados: {len(all_audio_files)}")
if all_emotions:
    print(f"🎭 Distribución final de emociones:")
    emotion_distribution = pd.Series(all_emotions).value_counts()
    print(emotion_distribution)
else:
    print("⚠️ No se encontraron archivos de audio válidos")
    print("Verificar que los datasets estén descargados correctamente")

# =============================================================================
# PARTE 5: EXTRACCIÓN MASIVA DE CARACTERÍSTICAS (ROBUSTA)
# =============================================================================

if len(all_audio_files) > 0:
    print("\n🔬 Iniciando extracción masiva de características...")

    # Inicializar extractor
    extractor = AdvancedFeatureExtractor()

    # Extraer características de todos los archivos
    X, y = extractor.batch_extract(all_audio_files, all_emotions, "Todos los datasets")

    print(f"\n📊 Características extraídas:")
    print(f"   - Forma de X: {X.shape}")
    print(f"   - Forma de y: {y.shape}")
    print(f"   - Emociones únicas: {np.unique(y)}")

    # Verificar que tenemos suficientes datos
    if len(X) < 50:
        print("⚠️ Pocos datos disponibles. Considerar:")
        print("   - Verificar que los datasets estén descargados")
        print("   - Ajustar la función de extracción de emociones")
        print("   - Usar datasets locales si están disponibles")
else:
    print("❌ No se encontraron archivos de audio para procesar")
    print("Por favor, verificar la configuración de datasets")

# =============================================================================
# PARTE 6: PREPROCESAMIENTO AVANZADO
# =============================================================================

# Solo continuar si tenemos datos suficientes
if 'X' in locals() and len(X) > 50:

    print("\n⚙️ Aplicando preprocesamiento avanzado...")

    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    emotion_classes = label_encoder.classes_
    n_classes = len(emotion_classes)

    print(f"🏷️ Clases codificadas: {emotion_classes}")
    print(f"📝 Número de clases: {n_classes}")

    # Verificar que tenemos suficientes clases
    if n_classes < 2:
        print("❌ Error: Se necesitan al menos 2 clases para entrenar")
        exit()

    # División estratificada de datos ANTES de la estandarización
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
        X_train, y_train_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_train_encoded
    )

    print(f"🔢 División de datos:")
    print(f"   - Entrenamiento: {X_train.shape[0]} muestras")
    print(f"   - Validación: {X_val.shape[0]} muestras")
    print(f"   - Prueba: {X_test.shape[0]} muestras")

    # Estandarización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("✅ Estandarización completada")

    # =============================================================================
    # PARTE 6.5: ANÁLISIS LDA (NUEVO)
    # =============================================================================

    print("\n🎯 Aplicando Análisis Discriminante Lineal (LDA)...")

    # Calcular el número máximo de componentes LDA
    max_lda_components = min(n_classes - 1, X_train_scaled.shape[1])
    print(f"📐 Componentes LDA máximos posibles: {max_lda_components}")

    # Crear y ajustar LDA
    lda = LinearDiscriminantAnalysis(n_components=max_lda_components)
    X_train_lda = lda.fit_transform(X_train_scaled, y_train_encoded)
    X_val_lda = lda.transform(X_val_scaled)
    X_test_lda = lda.transform(X_test_scaled)

    print(f"✅ LDA aplicado exitosamente")
    print(f"   - Forma original: {X_train_scaled.shape}")
    print(f"   - Forma LDA: {X_train_lda.shape}")
    print(f"   - Varianza explicada por componente: {lda.explained_variance_ratio_}")
    print(f"   - Varianza total explicada: {np.sum(lda.explained_variance_ratio_):.4f}")

    # =============================================================================
    # CONFIGURACIÓN DE CARPETAS DE RESULTADOS
    # =============================================================================

    print("\n📁 Configurando carpetas de resultados...")

    # Crear estructura de carpetas
    results_dir = "/content/Results"
    model_dir = os.path.join(results_dir, "Model")
    lda_dir = os.path.join(results_dir, "LDA") 
    plots_dir = os.path.join(results_dir, "Plots")

    for directory in [results_dir, model_dir, lda_dir, plots_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Carpeta creada: {directory}")

    # =============================================================================
    # VISUALIZACIÓN AVANZADA DEL LDA
    # =============================================================================

    print("\n📊 Creando visualizaciones LDA...")

    # Configurar estilo de matplotlib
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Crear colores únicos para cada emoción
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    emotion_colors = {emotion: colors[i] for i, emotion in enumerate(emotion_classes)}

    # 1. Gráfico 2D de las primeras dos componentes LDA
    if X_train_lda.shape[1] >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis Discriminante Lineal (LDA) - Visualizaciones', fontsize=16, fontweight='bold')

        # Subplot 1: Scatter plot 2D
        ax1 = axes[0, 0]
        for i, emotion in enumerate(emotion_classes):
            mask = y_train_encoded == i
            ax1.scatter(X_train_lda[mask, 0], X_train_lda[mask, 1], 
                       c=[emotion_colors[emotion]], label=emotion, alpha=0.7, s=50)

        ax1.set_xlabel('Primera Componente LDA', fontsize=12)
        ax1.set_ylabel('Segunda Componente LDA', fontsize=12)
        ax1.set_title('Distribución de Emociones en Espacio LDA (2D)', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Histograma de la primera componente
        ax2 = axes[0, 1]
        for i, emotion in enumerate(emotion_classes):
            mask = y_train_encoded == i
            ax2.hist(X_train_lda[mask, 0], alpha=0.6, label=emotion, bins=20,
                    color=emotion_colors[emotion])

        ax2.set_xlabel('Primera Componente LDA', fontsize=12)
        ax2.set_ylabel('Frecuencia', fontsize=12)
        ax2.set_title('Distribución de la Primera Componente LDA', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Boxplot de componentes LDA
        ax3 = axes[1, 0]
        lda_df = pd.DataFrame(X_train_lda[:, :min(4, X_train_lda.shape[1])], 
                             columns=[f'LDA_{i+1}' for i in range(min(4, X_train_lda.shape[1]))])
        lda_df['Emotion'] = [emotion_classes[i] for i in y_train_encoded]

        # Crear boxplot
        lda_melted = lda_df.melt(id_vars=['Emotion'], var_name='Componente', value_name='Valor')
        sns.boxplot(data=lda_melted, x='Componente', y='Valor', hue='Emotion', ax=ax3)
        ax3.set_title('Distribución de Componentes LDA por Emoción', fontweight='bold')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Subplot 4: Varianza explicada
        ax4 = axes[1, 1]
        component_range = range(1, len(lda.explained_variance_ratio_) + 1)
        ax4.bar(component_range, lda.explained_variance_ratio_, alpha=0.7, color='skyblue')
        ax4.plot(component_range, np.cumsum(lda.explained_variance_ratio_), 
                color='red', marker='o', linewidth=2, markersize=6)
        ax4.set_xlabel('Componente LDA', fontsize=12)
        ax4.set_ylabel('Varianza Explicada', fontsize=12)
        ax4.set_title('Varianza Explicada por Componente LDA', fontweight='bold')
        ax4.set_xticks(component_range)
        ax4.grid(True, alpha=0.3)

        # Agregar texto con varianza total
        total_variance = np.sum(lda.explained_variance_ratio_)
        ax4.text(0.7, 0.95, f'Varianza Total: {total_variance:.3f}', 
                transform=ax4.transAxes, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        plt.tight_layout()
        
        # Guardar figura LDA principal
        lda_main_path = os.path.join(lda_dir, "lda_analysis_main.png")
        plt.savefig(lda_main_path, dpi=300, bbox_inches='tight')
        print(f"💾 Guardado: {lda_main_path}")
        plt.show()

    # 2. Visualización 3D (si hay al menos 3 componentes)
    if X_train_lda.shape[1] >= 3:
        print("📊 Creando visualización 3D con Plotly...")

        # Crear DataFrame para Plotly
        lda_3d_df = pd.DataFrame({
            'LDA1': X_train_lda[:, 0],
            'LDA2': X_train_lda[:, 1],
            'LDA3': X_train_lda[:, 2],
            'Emotion': [emotion_classes[i] for i in y_train_encoded]
        })

        # Crear gráfico 3D interactivo
        fig_3d = px.scatter_3d(lda_3d_df, x='LDA1', y='LDA2', z='LDA3', 
                              color='Emotion', 
                              title='Análisis LDA - Visualización 3D Interactiva',
                              labels={'LDA1': 'Primera Componente LDA',
                                     'LDA2': 'Segunda Componente LDA', 
                                     'LDA3': 'Tercera Componente LDA'})

        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Primera Componente LDA',
                yaxis_title='Segunda Componente LDA',
                zaxis_title='Tercera Componente LDA'
            ),
            font=dict(size=12),
            width=800,
            height=600
        )

        # Guardar gráfico 3D
        lda_3d_path = os.path.join(lda_dir, "lda_3d_visualization.html")
        fig_3d.write_html(lda_3d_path)
        print(f"💾 Guardado: {lda_3d_path}")
        
        fig_3d.show()

    # 3. Matriz de correlación de componentes LDA
    print("📊 Analizando correlaciones entre componentes LDA...")

    if X_train_lda.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        
        # Crear DataFrame con componentes LDA
        lda_components_df = pd.DataFrame(
            X_train_lda, 
            columns=[f'LDA_Comp_{i+1}' for i in range(X_train_lda.shape[1])]
        )
        
        # Calcular matriz de correlación
        correlation_matrix = lda_components_df.corr()
        
        # Crear heatmap
        mask = np.triu(np.ones_like(correlation_matrix))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Matriz de Correlación - Componentes LDA', fontsize=16, fontweight='bold')
        
        # Guardar matriz de correlación
        correlation_path = os.path.join(lda_dir, "lda_correlation_matrix.png")
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        print(f"💾 Guardado: {correlation_path}")
        
        plt.tight_layout()
        plt.show()

    # 4. Análisis de separabilidad por pares de emociones
    print("📊 Analizando separabilidad entre emociones...")

    # Calcular distancias entre centroides de clases en espacio LDA
    centroids = []
    for i in range(n_classes):
        mask = y_train_encoded == i
        centroid = np.mean(X_train_lda[mask], axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    # Calcular matriz de distancias entre centroides
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(centroids))

    plt.figure(figsize=(10, 8))
    sns.heatmap(distances, annot=True, xticklabels=emotion_classes, 
               yticklabels=emotion_classes, cmap='viridis', fmt='.2f')
    plt.title('Matriz de Distancias entre Centroides de Emociones (Espacio LDA)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Emociones', fontsize=12)
    plt.ylabel('Emociones', fontsize=12)
    
    # Guardar matriz de distancias
    distances_path = os.path.join(lda_dir, "emotion_centroids_distances.png")
    plt.savefig(distances_path, dpi=300, bbox_inches='tight')
    print(f"💾 Guardado: {distances_path}")
    
    plt.tight_layout()
    
    # Guardar historial de entrenamiento
    training_history_path = os.path.join(model_dir, "training_history.png")
    plt.savefig(training_history_path, dpi=300, bbox_inches='tight')
    print(f"💾 Guardado: {training_history_path}")
    
    plt.show()

    # Reporte de LDA
    print(f"\n📋 Reporte de LDA:")
    print(f"   - Dimensiones originales: {X_train_scaled.shape[1]}")
    print(f"   - Dimensiones después de LDA: {X_train_lda.shape[1]}")
    print(f"   - Reducción de dimensionalidad: {((X_train_scaled.shape[1] - X_train_lda.shape[1]) / X_train_scaled.shape[1] * 100):.1f}%")
    print(f"   - Varianza retenida: {np.sum(lda.explained_variance_ratio_) * 100:.1f}%")

    # =============================================================================
    # GRÁFICOS ADICIONALES DE ANÁLISIS
    # =============================================================================

    print("\n📊 Creando gráficos adicionales de análisis...")

    # 1. Distribución de emociones en el dataset
    plt.figure(figsize=(12, 6))
    emotion_counts = pd.Series(y).value_counts()
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(emotion_counts.index, emotion_counts.values, 
                   color=[emotion_colors[emotion] for emotion in emotion_counts.index])
    plt.title('Distribución de Emociones en el Dataset', fontweight='bold')
    plt.xlabel('Emociones')
    plt.ylabel('Número de Muestras')
    plt.xticks(rotation=45)
    
    # Agregar valores en las barras
    for bar, value in zip(bars, emotion_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(value), ha='center', va='bottom')
    
    plt.subplot(1, 2, 2)
    plt.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%',
            colors=[emotion_colors[emotion] for emotion in emotion_counts.index])
    plt.title('Distribución Porcentual de Emociones', fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar distribución de emociones
    emotion_dist_path = os.path.join(plots_dir, "emotion_distribution.png")
    plt.savefig(emotion_dist_path, dpi=300, bbox_inches='tight')
    print(f"💾 Guardado: {emotion_dist_path}")
    plt.show()

    # 2. Comparación de características antes y después de LDA
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparación: Características Originales vs LDA', fontsize=16, fontweight='bold')

    # Antes de LDA - Primera componente principal
    pca_temp = PCA(n_components=2)
    X_pca = pca_temp.fit_transform(X_train_scaled)
    
    ax1 = axes[0, 0]
    for i, emotion in enumerate(emotion_classes):
        mask = y_train_encoded == i
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[emotion_colors[emotion]], label=emotion, alpha=0.6, s=30)
    ax1.set_title('Características Originales (PCA)', fontweight='bold')
    ax1.set_xlabel('Primera Componente PCA')
    ax1.set_ylabel('Segunda Componente PCA')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Después de LDA
    ax2 = axes[0, 1]
    for i, emotion in enumerate(emotion_classes):
        mask = y_train_encoded == i
        if X_train_lda.shape[1] >= 2:
            ax2.scatter(X_train_lda[mask, 0], X_train_lda[mask, 1], 
                       c=[emotion_colors[emotion]], label=emotion, alpha=0.6, s=30)
        else:
            ax2.scatter(X_train_lda[mask, 0], np.zeros(np.sum(mask)), 
                       c=[emotion_colors[emotion]], label=emotion, alpha=0.6, s=30)
    ax2.set_title('Características LDA', fontweight='bold')
    ax2.set_xlabel('Primera Componente LDA')
    ax2.set_ylabel('Segunda Componente LDA' if X_train_lda.shape[1] >= 2 else 'Valor Constante')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Comparación de varianza explicada
    ax3 = axes[1, 0]
    pca_variance = pca_temp.explained_variance_ratio_
    lda_variance = lda.explained_variance_ratio_
    
    x_pos = np.arange(max(len(pca_variance), len(lda_variance)))
    width = 0.35
    
    ax3.bar(x_pos[:len(pca_variance)] - width/2, pca_variance, width, 
           label='PCA', alpha=0.7, color='lightblue')
    ax3.bar(x_pos[:len(lda_variance)] + width/2, lda_variance, width, 
           label='LDA', alpha=0.7, color='lightcoral')
    
    ax3.set_title('Comparación: Varianza Explicada PCA vs LDA', fontweight='bold')
    ax3.set_xlabel('Componente')
    ax3.set_ylabel('Varianza Explicada')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Métricas de separabilidad
    ax4 = axes[1, 1]
    
    # Calcular métricas de separabilidad
    from sklearn.metrics import silhouette_score
    
    try:
        silhouette_original = silhouette_score(X_train_scaled[:1000], y_train_encoded[:1000])  # Muestra para velocidad
        silhouette_lda = silhouette_score(X_train_lda[:1000], y_train_encoded[:1000])
        
        metrics = ['Silhouette Score']
        original_scores = [silhouette_original]
        lda_scores = [silhouette_lda]
        
        x_pos = np.arange(len(metrics))
        ax4.bar(x_pos - width/2, original_scores, width, label='Original', alpha=0.7, color='lightblue')
        ax4.bar(x_pos + width/2, lda_scores, width, label='LDA', alpha=0.7, color='lightcoral')
        
        ax4.set_title('Métricas de Separabilidad', fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        print(f"📊 Métricas de separabilidad:")
        print(f"   - Silhouette Score (Original): {silhouette_original:.3f}")
        print(f"   - Silhouette Score (LDA): {silhouette_lda:.3f}")
        print(f"   - Mejora: {((silhouette_lda - silhouette_original) / silhouette_original * 100):.1f}%")
        
    except Exception as e:
        print(f"⚠️ No se pudieron calcular métricas de separabilidad: {e}")
        ax4.text(0.5, 0.5, 'Métricas no disponibles', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Métricas de Separabilidad', fontweight='bold')

    plt.tight_layout()
    
    # Guardar comparación
    comparison_path = os.path.join(lda_dir, "pca_vs_lda_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"💾 Guardado: {comparison_path}")
    plt.show()

    # =============================================================================
    # PARTE 7: ARQUITECTURA AVANZADA DE CNN 1D (MODIFICADA PARA LDA)
    # =============================================================================

    print("\n🏗️ Construyendo arquitectura avanzada de CNN 1D con características LDA...")

    # Convertir etiquetas a categorical
    y_train_categorical = to_categorical(y_train_encoded, num_classes=n_classes)
    y_val_categorical = to_categorical(y_val_encoded, num_classes=n_classes)
    y_test_categorical = to_categorical(y_test_encoded, num_classes=n_classes)

    # Reformatear datos LDA para CNN 1D
    X_train_cnn = np.expand_dims(X_train_lda, axis=2)
    X_val_cnn = np.expand_dims(X_val_lda, axis=2)
    X_test_cnn = np.expand_dims(X_test_lda, axis=2)

    print(f"🎯 Datos preparados para CNN 1D con LDA:")
    print(f"   - Forma de entrada: {X_train_cnn.shape}")

    def create_advanced_cnn_model_lda(input_shape, num_classes,
                                    conv1_filters=64, conv2_filters=32,
                                    kernel_size=3, dropout_rate=0.3,
                                    use_batch_norm=True):
        """Crea un modelo CNN 1D optimizado para características LDA"""

        model = Sequential([
            # Capa de entrada
            Input(shape=input_shape, name='input_lda_features'),

            # Primer bloque convolucional (ajustado para LDA)
            Conv1D(filters=conv1_filters,
                   kernel_size=kernel_size,
                   padding='same',
                   activation='relu',
                   name='conv1d_1'),

            BatchNormalization(name='batch_norm_1') if use_batch_norm else tf.keras.layers.Lambda(lambda x: x),

            MaxPooling1D(pool_size=2, name='maxpool_1'),
            Dropout(dropout_rate, name='dropout_1'),

            # Segundo bloque convolucional
            Conv1D(filters=conv2_filters,
                   kernel_size=kernel_size,
                   padding='same',
                   activation='relu',
                   name='conv1d_2'),

            BatchNormalization(name='batch_norm_2') if use_batch_norm else tf.keras.layers.Lambda(lambda x: x),

            MaxPooling1D(pool_size=2, name='maxpool_2'),
            Dropout(dropout_rate, name='dropout_2'),

            # Global Average Pooling
            GlobalAveragePooling1D(name='global_avg_pool'),

            # Capa densa intermedia (reducida para LDA)
            Dense(32, activation='relu', name='dense_intermediate'),
            Dropout(dropout_rate * 0.5, name='dropout_final'),

            # Capa de salida
            Dense(num_classes, activation='softmax', name='output_emotions')
        ])

        return model

    # Crear modelo optimizado para LDA
    model = create_advanced_cnn_model_lda(
        input_shape=(X_train_lda.shape[1], 1),
        num_classes=n_classes,
        conv1_filters=64,
        conv2_filters=32,
        kernel_size=3,
        dropout_rate=0.3,
        use_batch_norm=True
    )

    print("🔍 Resumen del modelo CNN 1D con LDA:")
    model.summary()

    # Calcular parámetros
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

    print(f"\n📊 Análisis de parámetros:")
    print(f"   - Parámetros totales: {total_params:,}")
    print(f"   - Parámetros entrenables: {trainable_params:,}")
    print(f"   - Reducción vs modelo original: ~{((128*180 - total_params) / (128*180) * 100):.1f}%")

    # =============================================================================
    # PARTE 8: CONFIGURACIÓN DEL OPTIMIZADOR
    # =============================================================================

    print("\n⚡ Configurando optimizador ADAM...")

    adam_optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    # Compilar modelo
    model.compile(
        optimizer=adam_optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("✅ Modelo compilado exitosamente")

    # =============================================================================
    # PARTE 9: CALLBACKS
    # =============================================================================

    print("\n📋 Configurando callbacks...")

    callbacks_list = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
    ]

    # =============================================================================
    # PARTE 10: ENTRENAMIENTO
    # =============================================================================

    print("\n🚀 Iniciando entrenamiento con características LDA...")

    EPOCHS = 100
    BATCH_SIZE = 32

    print(f"⚙️ Configuración de entrenamiento:")
    print(f"   - Épocas: {EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Características de entrada: {X_train_lda.shape[1]} (LDA)")

    history = model.fit(
        X_train_cnn, y_train_categorical,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val_cnn, y_val_categorical),
        callbacks=callbacks_list,
        verbose=1
    )

    # =============================================================================
    # PARTE 11: EVALUACIÓN COMPLETA
    # =============================================================================

    print("\n📈 Evaluando modelo...")

    # Evaluación en conjunto de prueba
    test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_categorical, verbose=0)
    print(f"📊 Resultados finales:")
    print(f"   - Pérdida en prueba: {test_loss:.4f}")
    print(f"   - Precisión en prueba: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Predicciones detalladas
    y_pred_proba = model.predict(X_test_cnn, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = y_test_encoded

    # Reporte de clasificación
    print(f"\n📋 Reporte de clasificación detallado:")
    print(classification_report(y_true, y_pred, target_names=emotion_classes))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_classes, yticklabels=emotion_classes)
    plt.title('Matriz de Confusión - Modelo CNN 1D con LDA', fontsize=16, fontweight='bold')
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    
    # Guardar matriz de confusión
    confusion_matrix_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    print(f"💾 Guardado: {confusion_matrix_path}")
    
    plt.tight_layout()
    plt.show()

    # Visualizar historial de entrenamiento
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Precisión
    axes[0].plot(history.history['accuracy'], label='Entrenamiento', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validación', linewidth=2)
    axes[0].set_title('Precisión del Modelo', fontweight='bold')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Precisión')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Pérdida
    axes[1].plot(history.history['loss'], label='Entrenamiento', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validación', linewidth=2)
    axes[1].set_title('Pérdida del Modelo', fontweight='bold')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Pérdida')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =============================================================================
    # PARTE 12: GUARDADO DE MODELOS Y RESULTADOS
    # =============================================================================

    print("\n💾 Guardando modelos y resultados...")

    # Guardar modelo CNN en carpeta Model (formato nativo Keras)
    model_path = os.path.join(model_dir, "emotion_cnn_lda_model.keras")
    model.save(model_path)
    print(f"✅ Modelo CNN guardado: {model_path}")
    
    # También guardar en formato H5 para compatibilidad
    model_h5_path = os.path.join(model_dir, "emotion_cnn_lda_model.h5")
    model.save(model_h5_path)
    print(f"✅ Modelo CNN (H5) guardado: {model_h5_path}")

    # Guardar objetos de preprocesamiento en carpeta Model
    import pickle

    preprocessing_objects = {
        'scaler': scaler,
        'lda': lda,
        'label_encoder': label_encoder,
        'emotion_classes': emotion_classes,
        'feature_extractor_config': {
            'sr': extractor.sr,
            'n_mfcc': extractor.n_mfcc,
            'n_chroma': extractor.n_chroma,
            'n_mel': extractor.n_mel,
            'hop_length': extractor.hop_length,
            'win_length': extractor.win_length
        }
    }

    preprocessing_path = os.path.join(model_dir, "preprocessing_objects.pkl")
    with open(preprocessing_path, 'wb') as f:
        pickle.dump(preprocessing_objects, f)
    print(f"✅ Objetos de preprocesamiento guardados: {preprocessing_path}")

    # Función helper para convertir numpy types a tipos nativos de Python
    def convert_numpy_types(obj):
        """Convierte recursivamente tipos numpy a tipos nativos de Python para JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        else:
            return obj

    # Guardar resultados de evaluación en carpeta Model (con conversión numpy)
    results = {
        'test_accuracy': float(test_accuracy),  # Convertir explícitamente
        'test_loss': float(test_loss),
        'classification_report': classification_report(y_true, y_pred, target_names=emotion_classes, output_dict=True),
        'confusion_matrix': cm.tolist(),  # Convertir array numpy a lista
        'lda_explained_variance': lda.explained_variance_ratio_.tolist(),
        'lda_total_variance': float(np.sum(lda.explained_variance_ratio_)),
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        },
        'model_summary': {
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'input_shape': int(X_train_lda.shape[1]),
            'output_classes': int(n_classes),
            'epochs_trained': len(history.history['loss']),
            'best_val_accuracy': float(max(history.history['val_accuracy']))
        }
    }

    # Aplicar conversión recursiva para asegurar compatibilidad JSON
    results = convert_numpy_types(results)

    results_path = os.path.join(model_dir, "model_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Resultados guardados: {results_path}")

    # Guardar información detallada de LDA en carpeta LDA (con conversión numpy)
    lda_results = {
        'original_dimensions': int(X_train_scaled.shape[1]),
        'lda_dimensions': int(X_train_lda.shape[1]),
        'dimension_reduction_percentage': float((X_train_scaled.shape[1] - X_train_lda.shape[1]) / X_train_scaled.shape[1] * 100),
        'explained_variance_ratio': lda.explained_variance_ratio_.tolist(),
        'total_variance_explained': float(np.sum(lda.explained_variance_ratio_)),
        'emotion_classes': emotion_classes.tolist(),
        'centroids_distances': distances.tolist()
    }

    # Aplicar conversión recursiva
    lda_results = convert_numpy_types(lda_results)

    lda_results_path = os.path.join(lda_dir, "lda_analysis_results.json")
    with open(lda_results_path, 'w') as f:
        json.dump(lda_results, f, indent=2)
    print(f"✅ Resultados LDA guardados: {lda_results_path}")

    # =============================================================================
    # VISUALIZACIONES ADICIONALES DEL MODELO CNN
    # =============================================================================

    print("\n🔍 Creando visualizaciones del modelo CNN...")

    # 1. Visualización de filtros aprendidos
    def plot_learned_filters(model, layer_name, save_path):
        """Visualiza los filtros aprendidos de una capa convolucional"""
        try:
            # Obtener pesos de la capa
            for layer in model.layers:
                if layer.name == layer_name and 'conv' in layer.name.lower():
                    weights = layer.get_weights()[0]  # [kernel_size, input_channels, filters]
                    
                    # Configurar subplot
                    n_filters = min(weights.shape[-1], 16)  # Máximo 16 filtros
                    cols = 4
                    rows = (n_filters + cols - 1) // cols
                    
                    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
                    fig.suptitle(f'Filtros Aprendidos - {layer_name}', fontsize=14, fontweight='bold')
                    
                    if rows == 1:
                        axes = axes.reshape(1, -1)
                    
                    for i in range(n_filters):
                        row = i // cols
                        col = i % cols
                        
                        # Visualizar filtro
                        filter_weights = weights[:, 0, i]  # [kernel_size]
                        axes[row, col].plot(filter_weights, linewidth=2)
                        axes[row, col].set_title(f'Filtro {i+1}')
                        axes[row, col].grid(True, alpha=0.3)
                        axes[row, col].set_xlabel('Posición')
                        axes[row, col].set_ylabel('Peso')
                    
                    # Ocultar subplots vacíos
                    for i in range(n_filters, rows * cols):
                        row = i // cols
                        col = i % cols
                        axes[row, col].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"💾 Guardado: {save_path}")
                    plt.show()
                    break
        except Exception as e:
            print(f"⚠️ Error visualizando filtros de {layer_name}: {e}")

    # Visualizar filtros de ambas capas convolucionales
    conv1_filters_path = os.path.join(model_dir, "conv1d_filters_layer1.png")
    plot_learned_filters(model, 'conv1d_1', conv1_filters_path)
    
    conv2_filters_path = os.path.join(model_dir, "conv1d_filters_layer2.png")
    plot_learned_filters(model, 'conv1d_2', conv2_filters_path)

    # 2. Análisis de activaciones
    def analyze_model_activations(model, X_sample, y_sample, save_dir):
        """Analiza las activaciones del modelo para muestras de ejemplo"""
        try:
            # Crear modelo para extraer activaciones intermedias
            layer_outputs = [layer.output for layer in model.layers if 'conv1d' in layer.name or 'dense' in layer.name]
            if layer_outputs:
                activation_model = Model(inputs=model.input, outputs=layer_outputs)
                
                # Seleccionar una muestra de cada clase
                sample_indices = []
                for class_idx in range(n_classes):
                    class_samples = np.where(y_sample == class_idx)[0]
                    if len(class_samples) > 0:
                        sample_indices.append(class_samples[0])
                
                if sample_indices:
                    X_samples = X_sample[sample_indices]
                    y_samples = y_sample[sample_indices]
                    
                    # Obtener activaciones
                    activations = activation_model.predict(X_samples, verbose=0)
                    
                    # Visualizar activaciones de capas convolucionales
                    for layer_idx, activation in enumerate(activations):
                        if len(activation.shape) == 3:  # Conv1D layers
                            fig, axes = plt.subplots(len(sample_indices), 1, figsize=(15, 2*len(sample_indices)))
                            if len(sample_indices) == 1:
                                axes = [axes]
                            
                            fig.suptitle(f'Activaciones - Capa {layer_idx+1}', fontsize=14, fontweight='bold')
                            
                            for sample_idx, (sample_activation, sample_class) in enumerate(zip(activation, y_samples)):
                                # Visualizar activación como heatmap
                                axes[sample_idx].imshow(sample_activation.T, cmap='viridis', aspect='auto')
                                axes[sample_idx].set_title(f'Clase: {emotion_classes[sample_class]}')
                                axes[sample_idx].set_xlabel('Posición Temporal')
                                axes[sample_idx].set_ylabel('Canal de Activación')
                            
                            plt.tight_layout()
                            activation_path = os.path.join(save_dir, f"activations_layer_{layer_idx+1}.png")
                            plt.savefig(activation_path, dpi=300, bbox_inches='tight')
                            print(f"💾 Guardado: {activation_path}")
                            plt.show()
                            
        except Exception as e:
            print(f"⚠️ Error analizando activaciones: {e}")

    # Analizar activaciones con muestras de prueba
    analyze_model_activations(model, X_test_cnn[:min(len(X_test_cnn), 20)], 
                            y_test_encoded[:min(len(y_test_encoded), 20)], model_dir)

    # 3. Matriz de predicciones con confianza
    plt.figure(figsize=(12, 8))
    
    # Calcular métricas por clase
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Crear subplot con métricas
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis Detallado del Rendimiento del Modelo', fontsize=16, fontweight='bold')
    
    # Subplot 1: Métricas por clase
    ax1 = axes[0, 0]
    x_pos = np.arange(len(emotion_classes))
    width = 0.25
    
    ax1.bar(x_pos - width, precision, width, label='Precisión', alpha=0.8)
    ax1.bar(x_pos, recall, width, label='Recall', alpha=0.8)
    ax1.bar(x_pos + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax1.set_xlabel('Emociones')
    ax1.set_ylabel('Score')
    ax1.set_title('Métricas por Clase')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(emotion_classes, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Distribución de confianza de predicciones
    ax2 = axes[0, 1]
    confidence_scores = np.max(y_pred_proba, axis=1)
    correct_predictions = (y_pred == y_true)
    
    ax2.hist(confidence_scores[correct_predictions], bins=20, alpha=0.7, 
            label='Predicciones Correctas', color='green')
    ax2.hist(confidence_scores[~correct_predictions], bins=20, alpha=0.7, 
            label='Predicciones Incorrectas', color='red')
    ax2.set_xlabel('Confianza de Predicción')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Confianza')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Curva de precisión vs recall
    ax3 = axes[1, 0]
    for i, emotion in enumerate(emotion_classes):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        from sklearn.metrics import precision_recall_curve, auc
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, y_score)
        auc_score = auc(recall_curve, precision_curve)
        
        ax3.plot(recall_curve, precision_curve, 
                label=f'{emotion} (AUC={auc_score:.2f})', alpha=0.8)
    
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precisión')
    ax3.set_title('Curvas Precisión-Recall por Clase')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Errores por emoción
    ax4 = axes[1, 1]
    error_counts = []
    for i in range(n_classes):
        errors = np.sum((y_true == i) & (y_pred != i))
        error_counts.append(errors)
    
    bars = ax4.bar(emotion_classes, error_counts, color='lightcoral', alpha=0.7)
    ax4.set_xlabel('Emociones')
    ax4.set_ylabel('Número de Errores')
    ax4.set_title('Errores de Clasificación por Emoción')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bar, value in zip(bars, error_counts):
        if value > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(value), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Guardar análisis de rendimiento
    performance_path = os.path.join(model_dir, "detailed_performance_analysis.png")
    plt.savefig(performance_path, dpi=300, bbox_inches='tight')
    print(f"💾 Guardado: {performance_path}")
    plt.show()

    # Crear reporte de texto completo
    report_content = f"""
    REPORTE COMPLETO - RECONOCIMIENTO DE EMOCIONES CNN 1D + LDA
    ===========================================================
    
    CONFIGURACIÓN DEL MODELO:
    - Arquitectura: CNN 1D con características LDA
    - Características de entrada: {X_train_lda.shape[1]} (después de LDA)
    - Características originales: {X_train_scaled.shape[1]}
    - Reducción dimensional: {((X_train_scaled.shape[1] - X_train_lda.shape[1]) / X_train_scaled.shape[1] * 100):.1f}%
    - Número de clases: {n_classes}
    - Clases: {', '.join(emotion_classes)}
    
    ANÁLISIS LDA:
    - Componentes LDA: {X_train_lda.shape[1]}
    - Varianza explicada total: {np.sum(lda.explained_variance_ratio_)*100:.2f}%
    - Varianza por componente: {[f'{var:.3f}' for var in lda.explained_variance_ratio_]}
    
    ARQUITECTURA DEL MODELO:
    - Parámetros totales: {total_params:,}
    - Parámetros entrenables: {trainable_params:,}
    - Capas convolucionales: 2 (64 y 32 filtros)
    - Kernel size: 3
    - Dropout rate: 0.3
    - Global Average Pooling + Dense(32) + Dense({n_classes})
    
    ENTRENAMIENTO:
    - Épocas totales configuradas: {EPOCHS}
    - Épocas realmente entrenadas: {len(history.history['loss'])}
    - Batch size: {BATCH_SIZE}
    - Optimizador: Adam (lr=0.001)
    - Early stopping: Activado (patience=15)
    - Reduce LR: Activado (patience=7)
    
    RESULTADOS FINALES:
    - Precisión en prueba: {test_accuracy*100:.2f}%
    - Pérdida en prueba: {test_loss:.4f}
    - Mejor precisión en validación: {max(history.history['val_accuracy'])*100:.2f}%
    
    MÉTRICAS POR CLASE:
    {classification_report(y_true, y_pred, target_names=emotion_classes)}
    
    DISTRIBUCIÓN DE DATOS:
    - Total de muestras procesadas: {len(X)}
    - Entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.1f}%)
    - Validación: {X_val.shape[0]} muestras ({X_val.shape[0]/len(X)*100:.1f}%)
    - Prueba: {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.1f}%)
    
    DATASETS UTILIZADOS:
    {chr(10).join([f"    - {name}: {('✅ Cargado' if path else '❌ No disponible')}" for name, path in dataset_paths.items()])}
    
    ARCHIVOS GENERADOS:
    ==================
    
    📁 Results/
    ├── 📁 Model/
    │   ├── emotion_cnn_lda_model.keras (Modelo entrenado - formato nativo - {os.path.getsize(model_path)/(1024*1024):.1f} MB)
    │   ├── emotion_cnn_lda_model.h5 (Modelo entrenado - formato legacy - {os.path.getsize(model_h5_path)/(1024*1024):.1f} MB)
    │   ├── preprocessing_objects.pkl (Objetos de preprocesamiento)
    │   ├── model_results.json (Resultados detallados)
    │   ├── confusion_matrix.png (Matriz de confusión)
    │   ├── training_history.png (Historial de entrenamiento)
    │   ├── conv1d_filters_layer1.png (Filtros primera capa)
    │   ├── conv1d_filters_layer2.png (Filtros segunda capa)
    │   ├── activations_layer_*.png (Activaciones por capa)
    │   └── detailed_performance_analysis.png (Análisis detallado)
    │
    ├── 📁 LDA/
    │   ├── lda_analysis_main.png (Análisis principal LDA)
    │   ├── lda_3d_visualization.html (Visualización 3D interactiva)
    │   ├── lda_correlation_matrix.png (Matriz de correlación)
    │   ├── emotion_centroids_distances.png (Distancias entre centroides)
    │   ├── pca_vs_lda_comparison.png (Comparación PCA vs LDA)
    │   └── lda_analysis_results.json (Resultados LDA detallados)
    │
    ├── 📁 Plots/
    │   └── emotion_distribution.png (Distribución de emociones)
    │
    ├── complete_report.txt (Este reporte)
    └── emotion_recognition_results.zip (Archivo comprimido con todo)
    
    INSTRUCCIONES DE USO DEL MODELO:
    ================================
    
    Para usar el modelo entrenado:
    
    1. Cargar objetos de preprocesamiento:
       import pickle
       with open('preprocessing_objects.pkl', 'rb') as f:
           prep_objects = pickle.load(f)
       
    2. Cargar modelo (formato nativo recomendado):
       from tensorflow.keras.models import load_model
       model = load_model('emotion_cnn_lda_model.keras')
       
       # O usar formato legacy si es necesario:
       model = load_model('emotion_cnn_lda_model.h5')
       
    3. Procesar nuevo audio:
       # Extraer características usando la misma configuración
       # Aplicar StandardScaler y LDA en el mismo orden
       # Hacer predicción con el modelo
    
    NOTAS TÉCNICAS:
    ==============
    - El modelo espera características LDA de dimensión {X_train_lda.shape[1]}
    - Las características originales deben ser estandarizadas antes de aplicar LDA
    - El LDA debe ser aplicado con los mismos parámetros guardados
    - La extracción de características debe usar la configuración guardada
    
    FECHA DE GENERACIÓN: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    report_path = os.path.join(results_dir, "complete_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"✅ Reporte completo guardado: {report_path}")

    # =============================================================================
    # PARTE 13: DESCARGA AUTOMÁTICA DE ARCHIVOS
    # =============================================================================

    print("\n📥 Preparando descarga automática de archivos...")

    # Crear archivo ZIP con todos los resultados
    import zipfile
    import glob

    zip_path = "/content/emotion_recognition_results.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Agregar todos los archivos de la carpeta Results recursivamente
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Crear ruta relativa desde Results para mantener estructura
                archive_name = os.path.relpath(file_path, "/content")
                zipf.write(file_path, archive_name)
    
    print(f"✅ Archivo ZIP creado: {zip_path}")

    # Mostrar estructura de archivos creados
    print(f"\n📂 Estructura de archivos generados:")
    for root, dirs, files in os.walk(results_dir):
        level = root.replace(results_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_size = os.path.getsize(os.path.join(root, file))
            if file_size > 1024*1024:  # MB
                size_str = f"({file_size/(1024*1024):.1f} MB)"
            elif file_size > 1024:  # KB
                size_str = f"({file_size/1024:.1f} KB)"
            else:
                size_str = f"({file_size} B)"
            print(f"{subindent}{file} {size_str}")

    # Función de descarga automática
    def download_all_results():
        """Descarga automáticamente todos los archivos generados"""
        print("\n🚀 Iniciando descarga automática...")
        print("=" * 50)
        
        try:
            # Descargar ZIP principal
            print("📦 Descargando archivo comprimido con todos los resultados...")
            files.download(zip_path)
            print("✅ Descarga del ZIP completada")
            
            # Descargar archivos individuales importantes
            important_files = [
                (model_path, "Modelo CNN entrenado (.keras - formato nativo)"),
                (model_h5_path, "Modelo CNN entrenado (.h5 - formato legacy)"),
                (os.path.join(model_dir, "model_results.json"), "Resultados del modelo (.json)"),
                (os.path.join(lda_dir, "lda_analysis_results.json"), "Resultados LDA (.json)"),
                (report_path, "Reporte completo (.txt)")
            ]
            
            print("\n📄 Descargando archivos individuales importantes...")
            for file_path, description in important_files:
                if os.path.exists(file_path):
                    print(f"⬇️ {description}")
                    files.download(file_path)
                else:
                    print(f"⚠️ No encontrado: {description}")
                    
            print("\n✅ ¡Descarga automática completada!")
            print("📁 Revisa tu carpeta de descargas para encontrar todos los archivos")
            
        except Exception as e:
            print(f"❌ Error en descarga automática: {e}")
            print("💡 Puedes descargar manualmente usando los comandos mostrados arriba")

    # Función de descarga por categorías
    def download_model_files():
        """Descarga solo archivos relacionados con el modelo"""
        print("📦 Descargando archivos del modelo...")
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                files.download(file_path)
        print("✅ Archivos del modelo descargados")

    def download_lda_files():
        """Descarga solo archivos relacionados con LDA"""
        print("📊 Descargando archivos de LDA...")
        for file in os.listdir(lda_dir):
            file_path = os.path.join(lda_dir, file)
            if os.path.isfile(file_path):
                files.download(file_path)
        print("✅ Archivos de LDA descargados")

    def download_plots():
        """Descarga solo gráficos y visualizaciones"""
        print("📈 Descargando gráficos...")
        for file in os.listdir(plots_dir):
            file_path = os.path.join(plots_dir, file)
            if os.path.isfile(file_path):
                files.download(file_path)
        print("✅ Gráficos descargados")

    # Mostrar opciones de descarga
    print(f"\n📥 OPCIONES DE DESCARGA:")
    print(f"=" * 60)
    print(f"")
    print(f"🎯 OPCIÓN RECOMENDADA - Descarga automática completa:")
    print(f"   download_all_results()")
    print(f"")
    print(f"📦 DESCARGAS POR CATEGORÍA:")
    print(f"   download_model_files()     # Solo archivos del modelo")
    print(f"   download_lda_files()       # Solo archivos de LDA")
    print(f"   download_plots()           # Solo gráficos adicionales")
    print(f"")
    print(f"📄 DESCARGAS INDIVIDUALES:")
    print(f"   # ZIP completo:")
    print(f"   files.download('{zip_path}')")
    print(f"")
    print(f"   # Modelo entrenado (formato nativo):")
    print(f"   files.download('{model_path}')")
    print(f"")
    print(f"   # Modelo entrenado (formato legacy):")
    print(f"   files.download('{model_h5_path}')")
    print(f"")
    print(f"   # Reporte completo:")
    print(f"   files.download('{report_path}')")
    print(f"")
    print(f"🔧 COMANDOS MANUALES COMPLETOS:")
    print(f"   import glob")
    print(f"   # Todos los archivos:")
    print(f"   for file in glob.glob('{results_dir}/**/*', recursive=True):")
    print(f"       if os.path.isfile(file): files.download(file)")

    # Crear script de descarga
    download_script = f'''# Script de descarga automática - Reconocimiento de Emociones CNN+LDA
from google.colab import files
import os
import glob

print("🚀 Iniciando descarga automática...")

# Descargar ZIP completo
print("📦 Descargando archivo ZIP completo...")
files.download('{zip_path}')

# Descargar archivos importantes individualmente
print("📄 Descargando archivos importantes...")
important_files = [
    '{model_path}',  # Modelo formato nativo (.keras)
    '{model_h5_path}',  # Modelo formato legacy (.h5)
    '{os.path.join(model_dir, "model_results.json")}',
    '{os.path.join(lda_dir, "lda_analysis_results.json")}',
    '{report_path}'
]

for file_path in important_files:
    if os.path.exists(file_path):
        print(f"⬇️ Descargando: {{os.path.basename(file_path)}}")
        files.download(file_path)
    else:
        print(f"⚠️ No encontrado: {{file_path}}")

print("✅ Descarga completada!")
print("📁 Revisa tu carpeta de descargas")
'''

    script_path = os.path.join(results_dir, "download_script.py")
    with open(script_path, 'w') as f:
        f.write(download_script)
    print(f"📝 Script de descarga creado: {script_path}")

    # Hacer las funciones disponibles globalmente
    globals()['download_all_results'] = download_all_results
    globals()['download_model_files'] = download_model_files
    globals()['download_lda_files'] = download_lda_files
    globals()['download_plots'] = download_plots

    # Resumen final
    print(f"\n🎉 ¡Entrenamiento completado exitosamente!")
    print(f"{'='*80}")
    print(f"📊 RESUMEN FINAL:")
    print(f"   - Precisión final: {test_accuracy*100:.2f}%")
    print(f"   - Reducción dimensional LDA: {X_train_scaled.shape[1]} → {X_train_lda.shape[1]} características")
    print(f"   - Varianza retenida por LDA: {np.sum(lda.explained_variance_ratio_)*100:.1f}%")
    print(f"   - Número de parámetros del modelo: {total_params:,}")
    print(f"   - Clases reconocidas: {', '.join(emotion_classes)}")
    print(f"{'='*80}")

else:
    print("❌ No hay suficientes datos para entrenar el modelo")
    print("Por favor, verificar la configuración de los datasets")