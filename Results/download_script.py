# Script de descarga automática - Reconocimiento de Emociones CNN+LDA
from google.colab import files
import os
import glob

print("🚀 Iniciando descarga automática...")

# Descargar ZIP completo
print("📦 Descargando archivo ZIP completo...")
files.download('/content/emotion_recognition_results.zip')

# Descargar archivos importantes individualmente
print("📄 Descargando archivos importantes...")
important_files = [
    '/content/Results/Model/emotion_cnn_lda_model.keras',  # Modelo formato nativo (.keras)
    '/content/Results/Model/emotion_cnn_lda_model.h5',  # Modelo formato legacy (.h5)
    '/content/Results/Model/model_results.json',
    '/content/Results/LDA/lda_analysis_results.json',
    '/content/Results/complete_report.txt'
]

for file_path in important_files:
    if os.path.exists(file_path):
        print(f"⬇️ Descargando: {os.path.basename(file_path)}")
        files.download(file_path)
    else:
        print(f"⚠️ No encontrado: {file_path}")

print("✅ Descarga completada!")
print("📁 Revisa tu carpeta de descargas")
