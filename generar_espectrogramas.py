import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import random
import kagglehub

# Configuración de carpetas para guardar espectrogramas
os.makedirs('graficas/espectrogramas', exist_ok=True)

def verificar_o_descargar(nombre, kaggle_id):
    if not os.path.exists(nombre):
        print(f"Descargando {nombre} desde Kaggle...")
        path = kagglehub.dataset_download(kaggle_id)
    else:
        print(f"El dataset {nombre} ya existe, no se descarga de nuevo.")
        path = nombre
    return path

def obtener_archivos_audio(base_path, num_archivos=5):
    """Obtiene una muestra aleatoria de archivos de audio de un dataset"""
    archivos = glob(os.path.join(base_path, "**", "*.wav"), recursive=True)
    if not archivos:
        # Intentar con otros formatos
        archivos.extend(glob(os.path.join(base_path, "**", "*.mp3"), recursive=True))
        archivos.extend(glob(os.path.join(base_path, "**", "*.flac"), recursive=True))
    
    if len(archivos) == 0:
        print(f"No se encontraron archivos de audio en {base_path}")
        return []
    
    # Seleccionar archivos aleatorios
    num_archivos = min(num_archivos, len(archivos))
    return random.sample(archivos, num_archivos)

def generar_espectrograma(audio_path, output_path, titulo):
    """Genera y guarda un espectrograma de un archivo de audio"""
    try:
        # Cargar audio
        y, sr = librosa.load(audio_path, sr=22050, duration=30)  # Máximo 30 segundos
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Análisis Espectral: {titulo}', fontsize=16, fontweight='bold')
        
        # 1. Forma de onda
        axes[0, 0].plot(np.linspace(0, len(y)/sr, len(y)), y)
        axes[0, 0].set_title('Forma de Onda')
        axes[0, 0].set_xlabel('Tiempo (s)')
        axes[0, 0].set_ylabel('Amplitud')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Espectrograma (STFT)
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img1 = librosa.display.specshow(S_db, x_axis='time', y_axis='hz', 
                                       sr=sr, ax=axes[0, 1], cmap='viridis')
        axes[0, 1].set_title('Espectrograma Lineal')
        axes[0, 1].set_xlabel('Tiempo (s)')
        axes[0, 1].set_ylabel('Frecuencia (Hz)')
        plt.colorbar(img1, ax=axes[0, 1], format='%+2.0f dB')
        
        # 3. Espectrograma Mel
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db_mel = librosa.power_to_db(S, ref=np.max)
        img2 = librosa.display.specshow(S_db_mel, x_axis='time', y_axis='mel',
                                       sr=sr, ax=axes[1, 0], cmap='plasma')
        axes[1, 0].set_title('Espectrograma Mel')
        axes[1, 0].set_xlabel('Tiempo (s)')
        axes[1, 0].set_ylabel('Frecuencia Mel')
        plt.colorbar(img2, ax=axes[1, 0], format='%+2.0f dB')
        
        # 4. Espectrograma Cromático
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        img3 = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma',
                                       ax=axes[1, 1], cmap='coolwarm')
        axes[1, 1].set_title('Cromograma')
        axes[1, 1].set_xlabel('Tiempo (s)')
        axes[1, 1].set_ylabel('Clases de Pitch')
        plt.colorbar(img3, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Espectrograma guardado: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error procesando {audio_path}: {str(e)}")
        return False

def main():
    # Descargar o verificar datasets
    print("Verificando datasets...")
    mesd_path = verificar_o_descargar("mexican-emotional-speech-database-mesd", 
                                     "saurabhshahane/mexican-emotional-speech-database-mesd")
    ravdess_path = verificar_o_descargar("ravdess-emotional-speech-audio", 
                                        "uwrfkaggler/ravdess-emotional-speech-audio")
    ser_path = verificar_o_descargar("speech-emotion-recognition-en", 
                                    "dmitrybabko/speech-emotion-recognition-en")
    
    datasets = [
        ("MESD", mesd_path),
        ("RAVDESS", ravdess_path),
        ("SER", ser_path)
    ]
    
    # Generar espectrogramas para cada dataset
    total_generados = 0
    
    for nombre_dataset, path in datasets:
        print(f"\nProcesando dataset {nombre_dataset}...")
        archivos = obtener_archivos_audio(path, num_archivos=3)  # 3 archivos por dataset
        
        for i, archivo in enumerate(archivos):
            nombre_archivo = os.path.basename(archivo)
            titulo = f"{nombre_dataset} - {nombre_archivo}"
            output_path = f"graficas/espectrogramas/{nombre_dataset}_{i+1}_{nombre_archivo.split('.')[0]}.png"
            
            if generar_espectrograma(archivo, output_path, titulo):
                total_generados += 1
    
    print(f"\n¡Proceso completado! Se generaron {total_generados} espectrogramas en la carpeta 'graficas/espectrogramas/'")
    
    # Generar un resumen de los archivos procesados
    archivos_generados = glob("graficas/espectrogramas/*.png")
    if archivos_generados:
        print("\nArchivos generados:")
        for archivo in archivos_generados:
            print(f"  - {os.path.basename(archivo)}")

if __name__ == "__main__":
    # Establecer semilla para reproducibilidad
    random.seed(42)
    main()