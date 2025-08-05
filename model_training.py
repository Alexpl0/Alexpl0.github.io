# =============================================================================
# ENTRENAMIENTO AVANZADO DE CNN 1D PARA RECONOCIMIENTO DE EMOCIONES EN VOZ
# Optimizado para Google Colab con análisis técnico detallado
# =============================================================================

print("🚀 Iniciando entrenamiento avanzado de CNN 1D para reconocimiento emocional")
print("=" * 80)

# =============================================================================
# PARTE 1: INSTALACIÓN Y CONFIGURACIÓN AVANZADA
# =============================================================================

print("📦 Instalando dependencias optimizadas para GPU...")
!pip install -q kaggle librosa numpy pandas scikit-learn tensorflow-gpu plotly seaborn tqdm
!pip install -q tensorboard tensorboard-plugin-profile
!pip install -q keras-tuner  # Para optimización de hiperparámetros

# Verificar GPU
import tensorflow as tf
print(f"🔧 TensorFlow versión: {tf.__version__}")
print(f"🎮 GPU disponible: {tf.config.list_physical_devices('GPU')}")
print(f"💾 Memoria GPU: {tf.config.experimental.get_memory_info('GPU:0') if tf.config.list_physical_devices('GPU') else 'No GPU'}")

# Configurar GPU para crecimiento dinámico de memoria
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Configuración de GPU optimizada")
    except RuntimeError as e:
        print(f"⚠️ Error configurando GPU: {e}")

# Importaciones principales
import os
import json
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Dropout, Flatten, Dense, 
                                   BatchNormalization, GlobalAveragePooling1D, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                       ModelCheckpoint, TensorBoard)
from tensorflow.keras.utils import to_categorical
from google.colab import files, drive
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

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
# PARTE 3: EXTRACCIÓN AVANZADA DE CARACTERÍSTICAS
# =============================================================================

print("\n🎵 Configurando extracción avanzada de características...")

class AdvancedFeatureExtractor:
    """Extractor avanzado de características de audio para reconocimiento emocional"""
    
    def __init__(self, 
                 sr=22050,           # Frecuencia de muestreo
                 n_mfcc=40,          # Número de MFCCs
                 n_chroma=12,        # Características Chroma
                 n_mel=128,          # Mel-spectrogram
                 hop_length=512,     # Hop length para STFT
                 win_length=2048):   # Window length para STFT
        
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
        """
        Extrae características completas de un archivo de audio
        
        Returns:
            numpy.array: Vector de características de tamaño 180
        """
        try:
            # Cargar audio con manejo de errores robusto
            audio, sr = librosa.load(audio_path, sr=self.sr, res_type='scipy')
            
            # Normalizar audio
            audio = librosa.util.normalize(audio)
            
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
            
            # Concatenar todas las características (40 + 12 + 128 = 180)
            features = np.hstack([mfccs_mean, chroma_mean, mel_mean])
            
            # Verificar que tenemos exactamente 180 características
            assert len(features) == 180, f"Error: {len(features)} características en lugar de 180"
            
            return features
            
        except Exception as e:
            print(f"❌ Error procesando {os.path.basename(audio_path)}: {e}")
            return None
    
    def batch_extract(self, audio_files, emotions, dataset_name):
        """Extrae características de múltiples archivos con barra de progreso"""
        print(f"\n🎯 Procesando {len(audio_files)} archivos de {dataset_name}...")
        
        features_list = []
        emotions_list = []
        
        for audio_file, emotion in tqdm(zip(audio_files, emotions), 
                                      total=len(audio_files),
                                      desc=f"Procesando {dataset_name}"):
            features = self.extract_features(audio_file)
            if features is not None:
                features_list.append(features)
                emotions_list.append(emotion)
        
        print(f"✅ Procesados {len(features_list)}/{len(audio_files)} archivos exitosamente")
        return np.array(features_list), np.array(emotions_list)

# =============================================================================
# PARTE 4: CARGA Y PROCESAMIENTO DE DATASETS
# =============================================================================

def load_dataset_files(dataset_path, dataset_name):
    """Carga archivos de audio y extrae etiquetas de emociones"""
    if dataset_path is None or not os.path.exists(dataset_path):
        print(f"⚠️ {dataset_name} no disponible")
        return [], []
    
    audio_files = []
    emotions = []
    
    if dataset_name == "RAVDESS":
        # Mapeo de códigos RAVDESS a emociones
        emotion_map = {
            1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
            5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
        }
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    try:
                        # Formato: 03-01-06-01-02-01-12.wav
                        # Tercer número es la emoción
                        emotion_code = int(file.split('-')[2])
                        emotion = emotion_map.get(emotion_code)
                        
                        if emotion and emotion != 'calm':  # Filtrar 'calm'
                            audio_files.append(os.path.join(root, file))
                            emotions.append(emotion)
                    except:
                        continue
    
    elif dataset_name == "TESS":
        # TESS tiene carpetas por emoción
        for emotion_folder in os.listdir(dataset_path):
            emotion_path = os.path.join(dataset_path, emotion_folder)
            if os.path.isdir(emotion_path):
                # Normalizar nombres de emociones
                emotion = emotion_folder.lower().replace('_', '').replace(' ', '')
                if emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
                    # Mapear 'fear' -> 'fearful', 'surprise' -> 'surprised'
                    if emotion == 'fear':
                        emotion = 'fearful'
                    elif emotion == 'surprise':
                        emotion = 'surprised'
                    
                    for file in os.listdir(emotion_path):
                        if file.endswith('.wav'):
                            audio_files.append(os.path.join(emotion_path, file))
                            emotions.append(emotion)
    
    elif dataset_name == "MESD":
        # Procesamiento similar adaptado para MESD
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    # Extraer emoción del nombre del archivo o carpeta
                    # Implementar según estructura específica de MESD
                    pass
    
    print(f"📊 {dataset_name}: {len(audio_files)} archivos encontrados")
    if emotions:
        emotion_counts = pd.Series(emotions).value_counts()
        print(f"   Distribución: {dict(emotion_counts)}")
    
    return audio_files, emotions

# Cargar todos los datasets
print("\n📁 Cargando archivos de datasets...")
all_audio_files = []
all_emotions = []

for dataset_name, path in dataset_paths.items():
    files, emotions = load_dataset_files(path, dataset_name)
    all_audio_files.extend(files)
    all_emotions.extend(emotions)

print(f"\n📈 Total de archivos cargados: {len(all_audio_files)}")
print(f"🎭 Distribución final de emociones:")
emotion_distribution = pd.Series(all_emotions).value_counts()
print(emotion_distribution)

# =============================================================================
# PARTE 5: EXTRACCIÓN MASIVA DE CARACTERÍSTICAS
# =============================================================================

print("\n🔬 Iniciando extracción masiva de características...")

# Inicializar extractor
extractor = AdvancedFeatureExtractor()

# Extraer características de todos los archivos
X, y = extractor.batch_extract(all_audio_files, all_emotions, "Todos los datasets")

print(f"\n📊 Características extraídas:")
print(f"   - Forma de X: {X.shape}")
print(f"   - Forma de y: {y.shape}")
print(f"   - Emociones únicas: {np.unique(y)}")

# =============================================================================
# PARTE 6: PREPROCESAMIENTO AVANZADO
# =============================================================================

print("\n⚙️ Aplicando preprocesamiento avanzado...")

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
emotion_classes = label_encoder.classes_
n_classes = len(emotion_classes)

print(f"🏷️ Clases codificadas: {emotion_classes}")
print(f"📝 Número de clases: {n_classes}")

# Convertir a categorical
y_categorical = to_categorical(y_encoded, num_classes=n_classes)

# División estratificada de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y_train, axis=1)
)

print(f"🔢 División de datos:")
print(f"   - Entrenamiento: {X_train.shape[0]} muestras")
print(f"   - Validación: {X_val.shape[0]} muestras") 
print(f"   - Prueba: {X_test.shape[0]} muestras")

# Estandarización robusta
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Reformatear para CNN 1D
X_train_cnn = np.expand_dims(X_train_scaled, axis=2)
X_val_cnn = np.expand_dims(X_val_scaled, axis=2)
X_test_cnn = np.expand_dims(X_test_scaled, axis=2)

print(f"🎯 Datos preparados para CNN 1D:")
print(f"   - Forma final: {X_train_cnn.shape}")

# =============================================================================
# PARTE 7: ARQUITECTURA AVANZADA DE CNN 1D
# =============================================================================

print("\n🏗️ Construyendo arquitectura avanzada de CNN 1D...")

def create_advanced_cnn_model(input_shape, num_classes, 
                            conv1_filters=256, conv2_filters=128,
                            kernel_size=5, dropout_rate=0.3,
                            use_batch_norm=True):
    """
    Crea un modelo CNN 1D avanzado con arquitectura optimizada
    
    Arquitectura detallada:
    - Input Layer: (180, 1)
    - Conv1D Block 1: Conv1D(256, 5) + BatchNorm + ReLU + MaxPool + Dropout
    - Conv1D Block 2: Conv1D(128, 5) + BatchNorm + ReLU + MaxPool + Dropout  
    - Global Average Pooling (alternativa a Flatten)
    - Dense Layer con regularización
    - Output Layer con Softmax
    """
    
    model = Sequential([
        # Capa de entrada
        Input(shape=input_shape, name='input_features'),
        
        # Primer bloque convolucional
        Conv1D(filters=conv1_filters, 
               kernel_size=kernel_size, 
               padding='same',
               activation='relu',
               name='conv1d_1'),
        
        # Batch Normalization (opcional pero recomendado)
        BatchNormalization(name='batch_norm_1') if use_batch_norm else tf.keras.layers.Lambda(lambda x: x),
        
        # MaxPooling y Dropout
        MaxPooling1D(pool_size=5, name='maxpool_1'),
        Dropout(dropout_rate, name='dropout_1'),
        
        # Segundo bloque convolucional  
        Conv1D(filters=conv2_filters,
               kernel_size=kernel_size,
               padding='same', 
               activation='relu',
               name='conv1d_2'),
               
        BatchNormalization(name='batch_norm_2') if use_batch_norm else tf.keras.layers.Lambda(lambda x: x),
        
        MaxPooling1D(pool_size=5, name='maxpool_2'),
        Dropout(dropout_rate, name='dropout_2'),
        
        # Global Average Pooling (más robusto que Flatten)
        GlobalAveragePooling1D(name='global_avg_pool'),
        
        # Capa densa intermedia (opcional)
        Dense(64, activation='relu', name='dense_intermediate'),
        Dropout(dropout_rate * 0.5, name='dropout_final'),
        
        # Capa de salida
        Dense(num_classes, activation='softmax', name='output_emotions')
    ])
    
    return model

# Crear modelo
model = create_advanced_cnn_model(
    input_shape=(180, 1),
    num_classes=n_classes,
    conv1_filters=256,
    conv2_filters=128,
    kernel_size=5,
    dropout_rate=0.2,
    use_batch_norm=True
)

print("🔍 Resumen del modelo:")
model.summary()

# Calcular parámetros por capa
total_params = model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

print(f"\n📊 Análisis de parámetros:")
print(f"   - Parámetros totales: {total_params:,}")
print(f"   - Parámetros entrenables: {trainable_params:,}")
print(f"   - Memoria estimada: ~{total_params * 4 / 1024 / 1024:.2f} MB")

# =============================================================================
# PARTE 8: CONFIGURACIÓN AVANZADA DEL OPTIMIZADOR ADAM
# =============================================================================

print("\n⚡ Configurando optimizador ADAM avanzado...")

# Configuración detallada de ADAM
adam_optimizer = Adam(
    learning_rate=0.001,      # α - Learning rate inicial
    beta_1=0.9,              # β₁ - Momento exponencial de primer orden
    beta_2=0.999,            # β₂ - Momento exponencial de segundo orden
    epsilon=1e-7,            # ε - Pequeño valor para estabilidad numérica
    amsgrad=False,           # Variante AMSGrad (opcional)
    name='adam_optimizer'
)

print(f"🎯 Parámetros de ADAM configurados:")
print(f"   - Learning rate (α): {adam_optimizer.learning_rate.numpy()}")
print(f"   - Beta 1 (β₁): {adam_optimizer.beta_1.numpy()}")  
print(f"   - Beta 2 (β₂): {adam_optimizer.beta_2.numpy()}")
print(f"   - Epsilon (ε): {adam_optimizer.epsilon.numpy()}")

# Compilar modelo con configuración avanzada
model.compile(
    optimizer=adam_optimizer,
    loss='categorical_crossentropy',  # Función de pérdida para clasificación multiclase
    metrics=['accuracy', 'top_2_accuracy'],  # Métricas adicionales
    run_eagerly=False  # Optimización para GPU
)

print("✅ Modelo compilado exitosamente")

# =============================================================================
# PARTE 9: CALLBACKS AVANZADOS PARA ENTRENAMIENTO
# =============================================================================

print("\n📋 Configurando callbacks avanzados...")

# Crear directorio para logs y checkpoints
log_dir = "/content/training_logs"
checkpoint_dir = "/content/model_checkpoints"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# 1. Early Stopping con paciencia adaptativa
early_stopping = EarlyStopping(
    monitor='val_loss',           # Métrica a monitorear
    patience=15,                  # Épocas sin mejora antes de parar
    restore_best_weights=True,    # Restaurar mejores pesos
    verbose=1,
    mode='min'
)

# 2. Reducción adaptativa del learning rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,                   # Factor de reducción (lr * factor)
    patience=7,                   # Épocas sin mejora antes de reducir
    min_lr=1e-6,                 # Learning rate mínimo
    verbose=1,
    mode='min'
)

# 3. Guardado del mejor modelo
model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
    mode='max'
)

# 4. TensorBoard para visualización
tensorboard = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,             # Frecuencia de histogramas
    write_graph=True,             # Guardar grafo del modelo
    write_images=True,            # Guardar imágenes de pesos
    update_freq='epoch',          # Actualizar cada época
    profile_batch=0               # No perfilar por defecto
)

callbacks_list = [early_stopping, reduce_lr, model_checkpoint, tensorboard]

print(f"🔧 Callbacks configurados:")
print(f"   - Early Stopping: paciencia {early_stopping.patience}")
print(f"   - ReduceLR: factor {reduce_lr.factor}, paciencia {reduce_lr.patience}")
print(f"   - ModelCheckpoint: guardando en {checkpoint_dir}")
print(f"   - TensorBoard: logs en {log_dir}")

# =============================================================================
# PARTE 10: ENTRENAMIENTO CON MONITOREO DETALLADO
# =============================================================================

print("\n🚀 Iniciando entrenamiento avanzado...")
print("=" * 60)

# Configuración de entrenamiento
EPOCHS = 100
BATCH_SIZE = 64
VALIDATION_FREQ = 1

print(f"⚙️ Configuración de entrenamiento:")
print(f"   - Épocas máximas: {EPOCHS}")
print(f"   - Tamaño de batch: {BATCH_SIZE}")
print(f"   - Frecuencia de validación: cada {VALIDATION_FREQ} época(s)")
print(f"   - Muestras de entrenamiento: {len(X_train_cnn)}")
print(f"   - Batches por época: {len(X_train_cnn) // BATCH_SIZE}")

# Entrenar modelo
print(f"\n🎯 Comenzando entrenamiento...")
start_time = tf.timestamp()

try:
    history = model.fit(
        X_train_cnn, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val_cnn, y_val),
        callbacks=callbacks_list,
        validation_freq=VALIDATION_FREQ,
        verbose=1,
        shuffle=True,
        workers=4,               # Paralelización
        use_multiprocessing=True  # Multiprocesamiento
    )
    
    end_time = tf.timestamp()
    training_time = end_time - start_time
    
    print(f"\n✅ Entrenamiento completado!")
    print(f"⏱️ Tiempo total: {training_time.numpy():.2f} segundos")
    print(f"📊 Épocas entrenadas: {len(history.history['loss'])}")
    
except Exception as e:
    print(f"❌ Error durante entrenamiento: {e}")
    raise

# =============================================================================
# PARTE 11: ANÁLISIS DETALLADO DE RESULTADOS
# =============================================================================

print("\n📈 Analizando resultados del entrenamiento...")

# Cargar mejor modelo guardado
try:
    best_model = tf.keras.models.load_model(os.path.join(checkpoint_dir, 'best_model.h5'))
    print("✅ Mejor modelo cargado exitosamente")
except:
    best_model = model
    print("⚠️ Usando modelo final (no se encontró checkpoint)")

# Evaluación en conjunto de prueba
print("\n🧪 Evaluando en conjunto de prueba...")
test_loss, test_accuracy, test_top2_accuracy = best_model.evaluate(
    X_test_cnn, y_test, 
    batch_size=BATCH_SIZE, 
    verbose=1
)

print(f"📊 Métricas finales en conjunto de prueba:")
print(f"   - Pérdida: {test_loss:.4f}")
print(f"   - Precisión: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   - Top-2 Precisión: {test_top2_accuracy:.4f} ({test_top2_accuracy*100:.2f}%)")

# Predicciones para análisis detallado
y_pred_proba = best_model.predict(X_test_cnn, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

# Reporte de clasificación detallado
print(f"\n📋 Reporte de clasificación detallado:")
classification_rep = classification_report(
    y_true, y_pred, 
    target_names=emotion_classes,
    digits=4,
    output_dict=True
)

print(classification_report(y_true, y_pred, target_names=emotion_classes, digits=4))

# Matriz de confusión
print(f"\n🔍 Generando matriz de confusión...")
cm = confusion_matrix(y_true, y_pred)

# Visualizar matriz de confusión
plt.figure(figsize=(12, 10))
sns.heatmap(cm, 
           annot=True, 
           fmt='d', 
           cmap='Blues',
           xticklabels=emotion_classes,
           yticklabels=emotion_classes,
           square=True,
           cbar_kws={'label': 'Número de muestras'})
plt.title('Matriz de Confusión - CNN 1D para Reconocimiento de Emociones', 
          fontsize=16, fontweight='bold')
plt.xlabel('Predicción', fontsize=14)
plt.ylabel('Valor Real', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/content/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# PARTE 12: ANÁLISIS DE CURVAS DE ENTRENAMIENTO
# =============================================================================

print("\n📉 Analizando curvas de entrenamiento...")

# Extraer historial de entrenamiento
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs_range = range(1, len(train_loss) + 1)

# Crear visualización de curvas de entrenamiento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico de pérdida
ax1.plot(epochs_range, train_loss, 'b-', label='Entrenamiento', linewidth=2)
ax1.plot(epochs_range, val_loss, 'r-', label='Validación', linewidth=2)
ax1.set_title('Curvas de Pérdida', fontsize=14, fontweight='bold')
ax1.set_xlabel('Época', fontsize=12)
ax1.set_ylabel('Pérdida (Categorical Crossentropy)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico de precisión
ax2.plot(epochs_range, train_acc, 'b-', label='Entrenamiento', linewidth=2)
ax2.plot(epochs_range, val_acc, 'r-', label='Validación', linewidth=2)
ax2.set_title('Curvas de Precisión', fontsize=14, fontweight='bold')
ax2.set_xlabel('Época', fontsize=12)
ax2.set_ylabel('Precisión', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Análisis de convergencia
best_epoch = np.argmin(val_loss) + 1
best_val_loss = min(val_loss)
best_val_acc = val_acc[np.argmin(val_loss)]

print(f"🎯 Análisis de convergencia:")
print(f"   - Mejor época: {best_epoch}")
print(f"   - Mejor pérdida de validación: {best_val_loss:.4f}")
print(f"   - Precisión en mejor época: {best_val_acc:.4f}")

# Detección de overfitting
final_train_loss = train_loss[-1]
final_val_loss = val_loss[-1]
overfitting_ratio = final_val_loss / final_train_loss

print(f"\n🔍 Análisis de overfitting:")
print(f"   - Pérdida final entrenamiento: {final_train_loss:.4f}")
print(f"   - Pérdida final validación: {final_val_loss:.4f}")
print(f"   - Ratio overfitting: {overfitting_ratio:.2f}")

if overfitting_ratio > 1.2:
    print("   ⚠️ Posible overfitting detectado")
elif overfitting_ratio < 1.1:
    print("   ✅ Modelo bien generalizado")
else:
    print("   ✅ Overfitting controlado")

# =============================================================================
# PARTE 13: ANÁLISIS POR EMOCIÓN
# =============================================================================

print("\n🎭 Análisis detallado por emoción...")

# Calcular métricas por clase
class_metrics = {}
for i, emotion in enumerate(emotion_classes):
    class_indices = (y_true == i)
    if np.sum(class_indices) > 0:
        class_predictions = y_pred[class_indices]
        class_accuracy = accuracy_score(y_true[class_indices], class_predictions)
        
        # Estadísticas adicionales
        true_positives = np.sum((y_true == i) & (y_pred == i))
        false_positives = np.sum((y_true != i) & (y_pred == i))
        false_negatives = np.sum((y_true == i) & (y_pred != i))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[emotion] = {
            'accuracy': class_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'support': np.sum(class_indices)
        }

# Mostrar métricas por emoción
print("\n📊 Métricas detalladas por emoción:")
print("-" * 80)
print(f"{'Emoción':<12} {'Precisión':<10} {'Recall':<10} {'F1-Score':<10} {'Soporte':<10}")
print("-" * 80)

for emotion, metrics in class_metrics.items():
    print(f"{emotion:<12} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
          f"{metrics['f1_score']:<10.3f} {metrics['support']:<10}")

# Identificar emociones más difíciles de clasificar
emotions_by_f1 = sorted(class_metrics.items(), key=lambda x: x[1]['f1_score'])
worst_emotion = emotions_by_f1[0]
best_emotion = emotions_by_f1[-1]

print(f"\n🎯 Análisis de dificultad:")
print(f"   - Emoción más difícil: {worst_emotion[0]} (F1: {worst_emotion[1]['f1_score']:.3f})")
print(f"   - Emoción más fácil: {best_emotion[0]} (F1: {best_emotion[1]['f1_score']:.3f})")

# =============================================================================
# PARTE 14: ANÁLISIS DE ACTIVACIONES Y FILTROS
# =============================================================================

print("\n🔬 Analizando activaciones y filtros aprendidos...")

# Crear modelo para extraer activaciones intermedias
layer_outputs = [layer.output for layer in best_model.layers if 'conv1d' in layer.name]
activation_model = Model(inputs=best_model.input, outputs=layer_outputs)

# Obtener activaciones de una muestra de ejemplo
sample_idx = 0
sample_input = X_test_cnn[sample_idx:sample_idx+1]
activations = activation_model.predict(sample_input)

print(f"🧠 Activaciones extraídas:")
for i, activation in enumerate(activations):
    print(f"   - Capa Conv1D {i+1}: {activation.shape}")

# Visualizar filtros de la primera capa convolucional
conv1_layer = best_model.get_layer('conv1d_1')
conv1_weights = conv1_layer.get_weights()[0]  # Shape: (kernel_size, input_channels, filters)

print(f"🔍 Análisis de filtros Conv1D_1:")
print(f"   - Forma de pesos: {conv1_weights.shape}")
print(f"   - Número de filtros: {conv1_weights.shape[2]}")
print(f"   - Tamaño de kernel: {conv1_weights.shape[0]}")

# Visualizar algunos filtros
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Filtros Aprendidos - Primera Capa Convolucional', fontsize=16, fontweight='bold')

for i in range(8):
    row = i // 4
    col = i % 4
    filter_weights = conv1_weights[:, 0, i]  # Tomar el primer canal de entrada
    
    axes[row, col].plot(filter_weights, linewidth=2)
    axes[row, col].set_title(f'Filtro {i+1}', fontsize=12)
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].set_xlabel('Posición en kernel')
    axes[row, col].set_ylabel('Peso')

plt.tight_layout()
plt.savefig('/content/learned_filters.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# PARTE 15: GUARDADO Y EXPORTACIÓN DE RESULTADOS
# =============================================================================

print("\n💾 Guardando resultados y modelos...")

# Crear directorio de resultados
results_dir = "/content/results"
os.makedirs(results_dir, exist_ok=True)

# Guardar modelo final
model_path = os.path.join(results_dir, 'cnn1d_emotion_model.h5')
best_model.save(model_path)
print(f"✅ Modelo guardado en: {model_path}")

# Guardar scaler
import joblib
scaler_path = os.path.join(results_dir, 'feature_scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler guardado en: {scaler_path}")

# Guardar label encoder
encoder_path = os.path.join(results_dir, 'label_encoder.pkl')
joblib.dump(label_encoder, encoder_path)
print(f"✅ Label encoder guardado en: {encoder_path}")

# Crear reporte completo de resultados
results_summary = {
    'model_architecture': {
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'input_shape': (180, 1),
        'output_classes': n_classes,
        'emotion_classes': emotion_classes.tolist()
    },
    'training_config': {
        'epochs_trained': len(history.history['loss']),
        'batch_size': BATCH_SIZE,
        'optimizer': 'Adam',
        'learning_rate': float(adam_optimizer.learning_rate.numpy()),
        'loss_function': 'categorical_crossentropy'
    },
    'performance_metrics': {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'test_top2_accuracy': float(test_top2_accuracy),
        'best_validation_accuracy': float(best_val_acc),
        'best_epoch': int(best_epoch)
    },
    'class_performance': class_metrics,
    'dataset_info': {
        'total_samples': len(X),
        'training_samples': len(X_train_cnn),
        'validation_samples': len(X_val_cnn),
        'test_samples': len(X_test_cnn),
        'emotion_distribution': emotion_distribution.to_dict()
    }
}

# Guardar reporte en JSON
report_path = os.path.join(results_dir, 'training_report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)
print(f"✅ Reporte guardado en: {report_path}")

# =============================================================================
# PARTE 16: FUNCIÓN DE PREDICCIÓN EN TIEMPO REAL
# =============================================================================

print("\n🔮 Creando función de predicción en tiempo real...")

def predict_emotion(audio_file_path, model, scaler, label_encoder, extractor):
    """
    Predice la emoción de un archivo de audio usando el modelo entrenado
    
    Args:
        audio_file_path (str): Ruta al archivo de audio
        model: Modelo CNN 1D entrenado
        scaler: StandardScaler entrenado
        label_encoder: LabelEncoder entrenado
        extractor: AdvancedFeatureExtractor
    
    Returns:
        dict: Diccionario con predicción y probabilidades
    """
    try:
        # Extraer características
        features = extractor.extract_features(audio_file_path)
        if features is None:
            return {"error": "No se pudieron extraer características"}
        
        # Preprocesar
        features_scaled = scaler.transform(features.reshape(1, -1))
        features_cnn = np.expand_dims(features_scaled, axis=2)
        
        # Predecir
        predictions = model.predict(features_cnn, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Decodificar etiqueta
        predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
        
        # Preparar resultado
        result = {
            "predicted_emotion": predicted_emotion,
            "confidence": confidence,
            "all_probabilities": {
                emotion: float(prob) 
                for emotion, prob in zip(label_encoder.classes_, predictions[0])
            } if isinstance(predictions[0], (list, np.ndarray)) else {}
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Error en predicción: {str(e)}"}

# Ejemplo de uso de la función de predicción
print("🧪 Probando función de predicción...")
if len(all_audio_files) > 0:
    test_file = all_audio_files[0]
    prediction_result = predict_emotion(test_file, best_model, scaler, label_encoder, extractor)
    
    print(f"📁 Archivo de prueba: {os.path.basename(test_file)}")
    if "error" not in prediction_result:
        print(f"🎭 Emoción predicha: {prediction_result['predicted_emotion']}")
        print(f"🎯 Confianza: {prediction_result['confidence']:.4f}")
        print(f"📊 Top 3 probabilidades:")
        sorted_probs = sorted(prediction_result['all_probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        for emotion, prob in sorted_probs:
            print(f"   - {emotion}: {prob:.4f}")
    else:
        print(f"❌ {prediction_result['error']}")

# =============================================================================
# PARTE 17: RESUMEN FINAL Y RECOMENDACIONES
# =============================================================================

print("\n" + "="*80)
print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print("="*80)

print(f"\n📊 RESUMEN DE RESULTADOS:")
print(f"   🎯 Precisión en prueba: {test_accuracy*100:.2f}%")
print(f"   📈 Top-2 Precisión: {test_top2_accuracy*100:.2f}%")
print(f"   ⏱️ Tiempo de entrenamiento: {training_time.numpy():.2f} segundos")
print(f"   🔢 Épocas entrenadas: {len(history.history['loss'])}")
print(f"   💾 Parámetros del modelo: {total_params:,}")

print(f"\n🎭 ANÁLISIS POR EMOCIÓN:")
print(f"   ✅ Mejor clasificada: {best_emotion[0]} (F1: {best_emotion[1]['f1_score']:.3f})")
print(f"   ⚠️ Más difícil: {worst_emotion[0]} (F1: {worst_emotion[1]['f1_score']:.3f})")

print(f"\n💡 RECOMENDACIONES PARA MEJORA:")
if test_accuracy < 0.75:
    print("   - Considerar aumentar datos con técnicas de data augmentation")
    print("   - Probar arquitecturas más profundas o con attention mechanisms")
    print("   - Explorar ensemble de múltiples modelos")
elif overfitting_ratio > 1.2:
    print("   - Aumentar regularización (dropout, weight decay)")
    print("   - Aplicar más data augmentation")
    print("   - Reducir complejidad del modelo")
else:
    print("   - El modelo muestra buen rendimiento y generalización")
    print("   - Considerar deployment para aplicación en producción")

print(f"\n📁 ARCHIVOS GENERADOS:")
print(f"   - Modelo entrenado: {model_path}")
print(f"   - Scaler: {scaler_path}")
print(f"   - Label encoder: {encoder_path}")
print(f"   - Reporte completo: {report_path}")
print(f"   - Matriz de confusión: /content/confusion_matrix.png")
print(f"   - Curvas de entrenamiento: /content/training_curves.png")
print(f"   - Filtros aprendidos: /content/learned_filters.png")

print(f"\n🚀 PRÓXIMOS PASOS:")
print("   1. Evaluar el modelo con datos externos")
print("   2. Implementar en aplicación web o móvil")
print("   3. Optimizar para inferencia en tiempo real")
print("   4. Considerar transfer learning para otros idiomas")

print("\n" + "="*80)
print("🎊 ¡PROYECTO DE CNN 1D COMPLETADO CON ÉXITO!")
print("="*80)