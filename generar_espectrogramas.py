# -*- coding: utf-8 -*-
"""
Script para generar visualizaciones 3D interactivas de PCA y LDA
usando Plotly para analizar emociones en archivos de audio.
VERSIÓN ADAPTADA PARA EJECUTARSE LOCALMENTE (ej. en VS Code).

Este script realiza las siguientes tareas principales:
1. Descarga datasets de audio emocional desde Kaggle
2. Extrae características de audio (pitch y MFCCs)
3. Aplica técnicas de reducción de dimensionalidad (PCA y LDA)
4. Genera visualizaciones 3D interactivas con Plotly
"""

# =============================================================================
# PASO 0: REQUISITOS PREVIOS (Asegúrate de hacer esto en tu terminal)
# =============================================================================
# 1. INSTALAR LIBRERÍAS:
#    Abre una terminal o símbolo del sistema y ejecuta el siguiente comando:
#    pip install kaggle plotly pandas numpy scikit-learn librosa
#
# 2. CONFIGURAR API DE KAGGLE:
#    - Descarga tu archivo 'kaggle.json' desde tu cuenta de Kaggle.
#    - Coloca el archivo en la carpeta correcta:
#      - Windows: C:\Users\<Tu-Usuario>\.kaggle\kaggle.json
#      - Mac/Linux: ~/.kaggle/kaggle.json
# =============================================================================


# =============================================================================
# PASO 1: IMPORTACIÓN DE LIBRERÍAS Y CARGA DE DATOS
# =============================================================================
print("--- Paso 1: Carga y Procesamiento de Datos ---")

# Importamos las librerías necesarias para el script:
import os           # Para operaciones del sistema operativo (rutas, directorios)
import librosa      # Para procesamiento de audio y extracción de características
import numpy as np  # Para operaciones numéricas y manejo de arrays
import pandas as pd # Para manipulación de datos tabulares (DataFrames)
from sklearn.decomposition import PCA  # Para análisis de componentes principales
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # Para análisis discriminante lineal
from sklearn.preprocessing import StandardScaler  # Para normalización de datos
from glob import glob  # Para búsqueda de archivos con patrones
import kaggle      # Librería oficial de Python para interactuar con la API de Kaggle
import plotly.express as px  # type: ignore # Para crear visualizaciones interactivas

def verificar_o_descargar_api(nombre_dir, kaggle_id):
    """
    Esta función gestiona la descarga automática de datasets desde Kaggle.
    
    Parámetros:
    - nombre_dir: Nombre del directorio donde se guardará el dataset
    - kaggle_id: Identificador único del dataset en Kaggle (formato: usuario/nombre-dataset)
    
    Funcionamiento:
    1. Verifica si el dataset ya está descargado localmente
    2. Si no existe, autentica con la API de Kaggle usando kaggle.json
    3. Descarga y descomprime automáticamente los archivos
    4. Maneja errores de autenticación o descarga
    
    Retorna:
    - La ruta del directorio si todo fue exitoso
    - None si hubo algún error
    """
    # Verificamos si la carpeta del dataset ya existe en el sistema
    if not os.path.exists(nombre_dir):
        print(f"Descargando y descomprimiendo '{nombre_dir}'...")
        try:
            # Autentica con Kaggle usando las credenciales del archivo kaggle.json
            kaggle.api.authenticate()
            
            # Descarga el dataset completo y lo descomprime automáticamente
            # path=nombre_dir: especifica dónde guardar los archivos
            # unzip=True: descomprime automáticamente los archivos ZIP
            kaggle.api.dataset_download_files(kaggle_id, path=nombre_dir, unzip=True)
            print(f"Dataset '{nombre_dir}' listo.")
        except Exception as e:
            # Captura cualquier error durante la descarga (credenciales, conexión, etc.)
            print(f"ERROR al descargar '{nombre_dir}': {e}")
            print("Asegúrate de que tu archivo 'kaggle.json' está configurado correctamente.")
            return None
    else:
        # Si la carpeta ya existe, evitamos descargar nuevamente
        print(f"El dataset '{nombre_dir}' ya existe.")
    return nombre_dir

def extraer_caracteristicas(audio_path):
    """
    Extrae características numéricas de un archivo de audio para análisis de emociones.
    
    Características extraídas:
    1. Pitch (tono fundamental): Indica la frecuencia fundamental de la voz
    2. MFCCs (Mel-Frequency Cepstral Coefficients): Capturan la forma espectral del audio
    
    Parámetros:
    - audio_path: Ruta completa al archivo de audio
    
    Proceso:
    1. Carga el audio con librosa a 22050 Hz (frecuencia estándar)
    2. Calcula el pitch usando piptrack (seguimiento de tonos)
    3. Extrae 13 coeficientes MFCC y calcula su promedio temporal
    4. Retorna un diccionario con todas las características
    
    Retorna:
    - Diccionario con características extraídas
    - None si hay algún error en el procesamiento
    """
    try:
        # Carga el archivo de audio:
        # sr=22050: frecuencia de muestreo estándar para análisis de voz
        # y: señal de audio como array numpy
        # sr: frecuencia de muestreo real
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Verificamos que el audio tenga suficiente duración para análisis
        # 2048 muestras ≈ 93ms a 22050 Hz
        if len(y) < 2048: 
            return None
        
        # EXTRACCIÓN DE PITCH (tono fundamental):
        # piptrack detecta múltiples tonos y sus magnitudes a lo largo del tiempo
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Calculamos el pitch promedio considerando solo los tonos más fuertes:
        # 1. Filtramos solo magnitudes superiores a la mediana (tonos más prominentes)
        # 2. Calculamos el promedio de los pitches correspondientes
        if np.any(magnitudes > 0):
            pitch = np.mean(pitches[magnitudes > np.median(magnitudes)])
        else:
            pitch = 0  # Si no hay tonos detectables, asignamos 0
        
        # EXTRACCIÓN DE MFCCs:
        # Los MFCCs capturan características espectrales importantes para reconocimiento de emociones
        # n_mfcc=13: número estándar de coeficientes MFCC
        # .mean(axis=1): promediamos a lo largo del tiempo para obtener un valor por coeficiente
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        
        # Construimos el diccionario de características:
        # - pitch: valor escalar del tono fundamental
        # - mfcc_1 a mfcc_13: los 13 coeficientes MFCC promediados
        return {"pitch": pitch, **{f"mfcc_{i+1}": mfcc[i] for i in range(13)}}
    
    except Exception:
        # Si hay cualquier error en el procesamiento del audio, retornamos None
        # Esto puede ocurrir por archivos corruptos, formatos no soportados, etc.
        return None

def cargar_dataset(dataset_name, base_path, emociones_map, ext="wav"):
    """
    Procesa un dataset completo de audio emocional y extrae características.
    
    Esta función orquesta todo el proceso de carga para un dataset específico:
    1. Busca todos los archivos de audio en el directorio
    2. Identifica la emoción de cada archivo según su nombre
    3. Extrae características de audio de cada archivo válido
    4. Construye un DataFrame con todas las muestras procesadas
    
    Parámetros:
    - dataset_name: Nombre del dataset ("RAVDESS" o "TESS")
    - base_path: Ruta base donde están los archivos del dataset
    - emociones_map: Diccionario que mapea códigos/palabras a nombres de emociones
    - ext: Extensión de archivos de audio a buscar (por defecto "wav")
    
    Retorna:
    - DataFrame de pandas con características extraídas y etiquetas de emoción
    - DataFrame vacío si no se encuentran archivos o hay errores
    """
    # Si no se pudo descargar/encontrar el dataset, retornamos DataFrame vacío
    if base_path is None: 
        return pd.DataFrame()
    
    # BÚSQUEDA RECURSIVA DE ARCHIVOS:
    # "**" indica búsqueda recursiva en subdirectorios
    # f"*.{ext}" busca archivos con la extensión especificada
    search_path = os.path.join(base_path, "**", f"*.{ext}")
    archivos = glob(search_path, recursive=True)
    
    # Verificamos que se encontraron archivos
    if not archivos: 
        print(f"ADVERTENCIA: No se encontraron archivos en {search_path}")
        return pd.DataFrame()

    # Lista para almacenar los datos procesados de cada archivo
    data = []
    print(f"Procesando {len(archivos)} archivos de '{dataset_name}'...")
    
    # PROCESAMIENTO DE CADA ARCHIVO:
    for archivo in archivos:
        # Extraemos solo el nombre del archivo (sin ruta) y lo convertimos a minúsculas
        nombre_base = os.path.basename(archivo).lower()
        emocion_encontrada = None

        # IDENTIFICACIÓN DE EMOCIÓN SEGÚN EL DATASET:
        
        if dataset_name == "RAVDESS":
            """
            RAVDESS usa un formato de nomenclatura específico:
            Ejemplo: "03-01-05-01-01-01-12.wav"
            Donde el tercer número (posición 2) indica la emoción:
            01=neutralidad, 02=calma, 03=alegría, 04=tristeza, 05=enojo, 06=miedo, 07=disgusto, 08=sorpresa
            """
            try:
                # Separamos por puntos para quitar la extensión, luego por guiones
                partes = nombre_base.split('.')[0].split('-')
                # El código de emoción está en la posición 2 (tercer elemento)
                emotion_code = int(partes[2])
                # Buscamos la emoción correspondiente en nuestro mapa
                emocion_encontrada = emociones_map.get(emotion_code)
            except (IndexError, ValueError):
                # Si el formato no coincide o hay error de conversión, omitimos el archivo
                continue
                
        elif dataset_name == "TESS":
            """
            TESS incluye el nombre de la emoción directamente en el nombre del archivo:
            Ejemplo: "YAF_back_angry_01.wav" contiene "_angry_" que indica la emoción
            """
            # Buscamos cada palabra clave de emoción en el nombre del archivo
            for emo_label, emo_keyword in emociones_map.items():
                # Verificamos si la palabra clave aparece rodeada de guiones bajos
                if f"_{emo_keyword.lower()}_" in nombre_base:
                    emocion_encontrada = emo_label
                    break  # Una vez encontrada, no seguimos buscando
        
        # PROCESAMIENTO DEL ARCHIVO SI SE IDENTIFICÓ LA EMOCIÓN:
        if emocion_encontrada:
            # Extraemos las características de audio del archivo
            features = extraer_caracteristicas(archivo)
            
            # Si la extracción fue exitosa, agregamos los datos
            if features:
                # Añadimos la etiqueta de emoción a las características
                features["emocion"] = emocion_encontrada
                # Agregamos el registro completo a nuestra lista de datos
                data.append(features)

    print(f"Se procesaron {len(data)} archivos válidos de '{dataset_name}'.")
    # Convertimos la lista de diccionarios a un DataFrame de pandas
    return pd.DataFrame(data)

# =============================================================================
# EJECUCIÓN DE LA CARGA DE DATOS
# =============================================================================

# DESCARGA DE DATASETS:
# Intentamos descargar ambos datasets de Kaggle usando sus IDs únicos
ravdess_path = verificar_o_descargar_api("RAVDESS_DATA", "uwrfkaggler/ravdess-emotional-speech-audio")
tess_path = verificar_o_descargar_api("TESS_DATA", "ejlok1/toronto-emotional-speech-set-tess")

# MAPAS DE TRADUCCIÓN DE EMOCIONES:

# Para RAVDESS: mapea códigos numéricos a nombres de emociones en español
# Los códigos 1 y 2 ambos representan neutralidad (neutral y calm)
ravdess_emociones_map = {
    1: 'neutralidad',  # neutral
    2: 'neutralidad',  # calm (lo agrupamos con neutralidad)
    3: 'alegría',      # happy
    4: 'tristeza',     # sad
    5: 'enojo',        # angry
    6: 'miedo',        # fearful
    7: 'disgusto',     # disgust
    8: 'sorpresa'      # surprised
}

# Para TESS: mapea nombres de emociones en español a palabras clave en inglés del dataset
tess_emociones_map = {
    "alegría": "happy",
    "tristeza": "sad", 
    "enojo": "angry",
    "miedo": "fear",
    "sorpresa": "ps",        # "surprised" aparece como "ps" en TESS
    "disgusto": "disgust",
    "neutralidad": "neutral"
}

# CARGA Y COMBINACIÓN DE DATASETS:
print("\n--- Cargando datasets de audio emocional ---")

# Procesamos cada dataset usando las funciones definidas
df_ravdess = cargar_dataset("RAVDESS", ravdess_path, ravdess_emociones_map)
df_tess = cargar_dataset("TESS", tess_path, tess_emociones_map)

# Combinamos ambos datasets en un solo DataFrame:
# ignore_index=True: reinicia el índice en el DataFrame combinado
# dropna(): elimina filas con valores faltantes
df = pd.concat([df_ravdess, df_tess], ignore_index=True).dropna()


# =============================================================================
# PASO 2: PREPARACIÓN DE DATOS PARA MODELADO 3D
# =============================================================================

# VERIFICACIÓN DE DATOS:
if df.empty:
    print("\nERROR CRÍTICO: El DataFrame está vacío. No se puede continuar.")
    print("Verifica que los datasets se hayan descargado correctamente.")
else:
    print(f"\nDataset combinado creado con {len(df)} muestras.")
    print("Distribución de emociones:\n", df['emocion'].value_counts())
    
    print("\n--- Paso 2: Preparando Datos para Modelado 3D ---")
    
    # SEPARACIÓN DE CARACTERÍSTICAS Y ETIQUETAS:
    # X: matriz de características (pitch + 13 MFCCs = 14 características)
    # y: vector de etiquetas (emociones)
    features_cols = [col for col in df.columns if col != 'emocion']
    X = df[features_cols]  # Todas las columnas excepto 'emocion'
    y = df['emocion']      # Solo la columna de emociones
    
    print(f"Características utilizadas: {features_cols}")
    print(f"Forma de la matriz X: {X.shape}")
    print(f"Emociones únicas en y: {y.unique()}")

    # NORMALIZACIÓN DE DATOS:
    # StandardScaler normaliza cada característica para tener media 0 y desviación estándar 1
    # Esto es crucial para PCA y LDA ya que son sensibles a la escala de las variables
    print("\nNormalizando características...")
    scaler = StandardScaler()
    
    # fit_transform: calcula la media y desviación estándar, luego normaliza
    X_scaled = scaler.fit_transform(X)
    print("Normalización completada.")

    # CREACIÓN DE DIRECTORIO PARA GRÁFICAS:
    output_dir = "graficas/plots_3d"
    os.makedirs(output_dir, exist_ok=True)  # exist_ok=True evita error si ya existe
    print(f"Las gráficas se guardarán en: {output_dir}")

    # =============================================================================
    # PASO 3: GENERACIÓN DE GRÁFICA 3D CON PCA
    # =============================================================================
    print("\n--- Paso 3: Generando Gráfica 3D de PCA ---")
    
    """
    PCA (Análisis de Componentes Principales):
    - Técnica de reducción de dimensionalidad no supervisada
    - Encuentra direcciones de máxima varianza en los datos
    - Proyecta los datos a un espacio de menor dimensión
    - Preserva la mayor cantidad posible de información (varianza)
    - No considera las etiquetas de clase
    """
    
    # Creamos una instancia de PCA para reducir de 14 dimensiones a 3
    pca = PCA(n_components=3)
    print("Aplicando PCA para reducir a 3 dimensiones...")
    
    # fit_transform: calcula los componentes principales y transforma los datos
    X_pca_3d = pca.fit_transform(X_scaled)
    
    # Información sobre la varianza explicada por cada componente
    explained_variance = pca.explained_variance_ratio_
    total_variance = np.sum(explained_variance)
    print(f"Varianza explicada por cada componente: {explained_variance}")
    print(f"Varianza total explicada por los 3 componentes: {total_variance:.3f} ({total_variance*100:.1f}%)")
    
    # PREPARACIÓN DE DATOS PARA PLOTLY:
    # Creamos un DataFrame con los componentes principales y las etiquetas
    df_pca_3d = pd.DataFrame(X_pca_3d, columns=['PC1', 'PC2', 'PC3'])
    df_pca_3d['emocion'] = y.values  # Agregamos las etiquetas de emoción
    
    print("Estructura del DataFrame para PCA:")
    print(df_pca_3d.head())

    # CREACIÓN DE LA VISUALIZACIÓN 3D:
    print("Creando visualización 3D interactiva con Plotly...")
    fig_pca = px.scatter_3d(
        df_pca_3d, 
        x='PC1', y='PC2', z='PC3',  # Ejes x, y, z
        color='emocion',             # Colorear puntos según la emoción
        title='Visualización 3D de PCA por Emoción',
        labels={
            'PC1': 'Componente Principal 1', 
            'PC2': 'Componente Principal 2', 
            'PC3': 'Componente Principal 3'
        }
    )
    
    # PERSONALIZACIÓN DE LA VISUALIZACIÓN:
    # Ajustamos el tamaño y transparencia de los puntos para mejor visualización
    fig_pca.update_traces(marker=dict(size=3, opacity=0.8))
    
    # Añadimos información adicional al título
    fig_pca.update_layout(
        title=f'Visualización 3D de PCA por Emoción<br><sub>Varianza explicada: {total_variance*100:.1f}%</sub>'
    )
    
    # GUARDADO DE LA GRÁFICA:
    # Guardamos como HTML para mantener la interactividad
    pca_filepath = os.path.join(output_dir, "pca_3d_plot.html")
    fig_pca.write_html(pca_filepath)
    print(f"Gráfica 3D de PCA guardada en: {pca_filepath}")

    # =============================================================================
    # PASO 4: GENERACIÓN DE GRÁFICA 3D CON LDA
    # =============================================================================
    print("\n--- Paso 4: Generando Gráfica 3D de LDA ---")
    
    """
    LDA (Análisis Discriminante Lineal):
    - Técnica de reducción de dimensionalidad supervisada
    - Busca direcciones que maximizan la separación entre clases
    - Considera las etiquetas de clase durante el entrenamiento
    - Número máximo de componentes = min(n_clases-1, n_características)
    - Mejor para clasificación que PCA
    """
    
    try:
        # CÁLCULO DEL NÚMERO DE COMPONENTES LDA:
        # LDA puede generar máximo (número_de_clases - 1) componentes
        n_clases = len(df['emocion'].unique())
        n_caracteristicas = X.shape[1]
        
        # El número de componentes está limitado por:
        # 1. Número de clases - 1
        # 2. Número de características originales  
        # 3. Número deseado para visualización (3)
        n_components_lda = min(n_clases - 1, n_caracteristicas, 3)
        
        print(f"Número de clases de emoción: {n_clases}")
        print(f"Número de características: {n_caracteristicas}")
        print(f"Componentes LDA calculados: {n_components_lda}")
        
        # VERIFICACIÓN DE VIABILIDAD:
        if n_components_lda < 3:
            print(f"ADVERTENCIA: LDA solo puede generar {n_components_lda} componentes.")
            print("Se necesita un mínimo de 3 para un gráfico 3D. Omitiendo gráfico LDA 3D.")
        else:
            # APLICACIÓN DE LDA:
            print("Aplicando LDA para maximizar separación entre emociones...")
            lda = LDA(n_components=n_components_lda)
            
            # fit_transform: entrena LDA con las etiquetas y transforma los datos
            X_lda_3d = lda.fit_transform(X_scaled, y)
            
            # PREPARACIÓN DE DATOS PARA PLOTLY:
            # Creamos nombres para los discriminantes lineales
            lda_cols = [f'LD{i+1}' for i in range(n_components_lda)]
            df_lda_3d = pd.DataFrame(X_lda_3d, columns=lda_cols)
            df_lda_3d['emocion'] = y.values
            
            print("Estructura del DataFrame para LDA:")
            print(df_lda_3d.head())

            # CREACIÓN DE LA VISUALIZACIÓN 3D:
            print("Creando visualización 3D de LDA...")
            fig_lda = px.scatter_3d(
                df_lda_3d, 
                x='LD1', y='LD2', z='LD3',  # Discriminantes lineales como ejes
                color='emocion',             # Colorear según emoción
                title='Visualización 3D de LDA por Emoción',
                labels={
                    'LD1': 'Discriminante Lineal 1', 
                    'LD2': 'Discriminante Lineal 2', 
                    'LD3': 'Discriminante Lineal 3'
                }
            )
            
            # PERSONALIZACIÓN DE LA VISUALIZACIÓN:
            fig_lda.update_traces(marker=dict(size=3, opacity=0.8))
            
            # Añadimos subtítulo informativo
            fig_lda.update_layout(
                title='Visualización 3D de LDA por Emoción<br><sub>Optimizado para separación de clases</sub>'
            )
            
            # GUARDADO DE LA GRÁFICA:
            lda_filepath = os.path.join(output_dir, "lda_3d_plot.html")
            fig_lda.write_html(lda_filepath)
            print(f"Gráfica 3D de LDA guardada en: {lda_filepath}")

    except Exception as e:
        print(f"ERROR al generar la gráfica 3D de LDA: {e}")
        print("Esto puede ocurrir si hay muy pocas muestras por clase o problemas de singularidad.")

    # =============================================================================
    # FINALIZACIÓN Y RESUMEN
    # =============================================================================
    print("\n" + "="*50)
    print("¡PROCESO COMPLETADO EXITOSAMENTE!")
    print("="*50)
    print(f"✓ Datos procesados: {len(df)} muestras de audio")
    print(f"✓ Características extraídas: {len(features_cols)} por muestra")
    print(f"✓ Emociones analizadas: {list(y.unique())}")
    print(f"✓ Gráficas guardadas en: {output_dir}")
    print("\nArchivos generados:")
    print(f"  - PCA 3D: {os.path.join(output_dir, 'pca_3d_plot.html')}")
    if n_components_lda >= 3:
        print(f"  - LDA 3D: {os.path.join(output_dir, 'lda_3d_plot.html')}")
    print("\nAbra los archivos HTML en su navegador para visualizar las gráficas interactivas.")
