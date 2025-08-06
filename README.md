# 🚀 Reconocimiento de Emociones en Voz con CNN 1D

Este README documenta el proceso de entrenamiento de una Red Neuronal Convolucional 1D (CNN) para el reconocimiento de emociones a partir de audio. El script realiza la preparación de datos, extracción de características, construcción del modelo, entrenamiento y evaluación.

## 📋 Índice

1.  [Configuración del Entorno](https://www.google.com/search?q=%231-configuraci%C3%B3n-del-entorno)
2.  [Preparación de Datos](https://www.google.com/search?q=%232-preparaci%C3%B3n-de-datos)
3.  [Extracción de Características](https://www.google.com/search?q=%233-extracci%C3%B3n-de-caracter%C3%ADsticas)
4.  [Preprocesamiento y División](https://www.google.com/search?q=%234-preprocesamiento-y-divisi%C3%B3n)
5.  [Arquitectura del Modelo](https://www.google.com/search?q=%235-arquitectura-del-modelo)
6.  [Entrenamiento](https://www.google.com/search?q=%236-entrenamiento)
7.  [Resultados y Evaluación](https://www.google.com/search?q=%237-resultados-y-evaluaci%C3%B3n)
8.  [Modelo Guardado](https://www.google.com/search?q=%238-modelo-guardado)

-----

## 1\. Configuración del Entorno

### Dependencias

Se instalaron las dependencias necesarias para el proyecto.

  - ✅ `resampy` instalado correctamente.

> ⚠️ **Nota:** Se encontró un error durante la instalación de un paquete vía `pip`. Esto no afectó la ejecución del script principal, pero se debe tener en cuenta para futuras ejecuciones.
>
> ```sh
> error: subprocess-exited-with-error
> × python setup.py egg_info did not run successfully.
> ...
> error: metadata-generation-failed
> ```

### Especificaciones del Sistema

  - **Versión de TensorFlow:** `2.18.0`
  - **Hardware:** ⚠️ No se detectó GPU, el entrenamiento se ejecutó en CPU.

-----

## 2\. Preparación de Datos

### Fuentes de Datos

  - ✅ Acceso a Kaggle configurado.
  - ✅ Dataset **RAVDESS** cargado.
  - ✅ Dataset **TESS** cargado.
  - ⬇️ Dataset **MESD** descargado exitosamente.

### Carga y Distribución de Archivos

  - 📊 **RAVDESS:** 2496 archivos encontrados.

  - 📊 **TESS:** 4800 archivos encontrados.

  - ⚠️ **MESD:** No se encontró o no estuvo disponible durante la carga.

  - 📈 **Total de archivos cargados:** 7296

  - 🎭 **Distribución final de emociones:**

    ```
    fearful      1184
    happy        1184
    angry        1184
    sad          1184
    disgust      1184
    neutral       992
    surprised     384
    Name: count, dtype: int64
    ```

-----

## 3\. Extracción de Características

Se extrajeron características de audio avanzadas de los 7296 archivos.

### 🔧 Configuración del Extractor

  - **Frecuencia de muestreo:** 22050 Hz
  - **MFCCs:** 40
  - **Características Chroma:** 12
  - **Mel-spectrogram bins:** 128

### 📊 Resultados de la Extracción

  - **Forma de $X$ (características):** `(7296, 180)`
  - **Forma de $y$ (etiquetas):** `(7296,)`
  - **Emociones únicas:** `['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']`

-----

## 4\. Preprocesamiento y División

Los datos fueron preprocesados y divididos para el entrenamiento del modelo.

  - **🏷️ Clases codificadas:** 7 clases en total.
  - **🔢 División de datos:**
      - **Entrenamiento:** 4668 muestras
      - **Validación:** 1168 muestras
      - **Prueba:** 1460 muestras
  - **🎯 Forma de entrada para la CNN 1D:** `(num_muestras, 180, 1)`

-----

## 5\. Arquitectura del Modelo

Se construyó una arquitectura de CNN 1D avanzada con capas de convolución, normalización por lotes, max-pooling y dropout para evitar el sobreajuste.

### 🔍 Resumen del Modelo

```text
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv1d_1 (Conv1D)               │ (None, 180, 128)       │           768 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_norm_1 (BatchNormalizatio │ (None, 180, 128)       │           512 │
│ n)                              │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ maxpool_1 (MaxPooling1D)        │ (None, 90, 128)        │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 90, 128)        │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv1d_2 (Conv1D)               │ (None, 90, 64)         │        41,024 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_norm_2 (BatchNormalizatio │ (None, 90, 64)         │           256 │
│ n)                              │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ maxpool_2 (MaxPooling1D)        │ (None, 45, 64)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 45, 64)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_avg_pool                 │ (None, 64)             │             0 │
│ (GlobalAveragePooling1D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_intermediate (Dense)      │ (None, 64)             │         4,160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_final (Dropout)         │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_emotions (Dense)         │ (None, 7)              │           455 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

  - **Parámetros totales:** 47,175
  - **Parámetros entrenables:** 46,791
  - **Parámetros no entrenables:** 384

-----

## 6\. Entrenamiento

El modelo fue compilado con el optimizador **ADAM** y la función de pérdida `categorical_crossentropy`. Se utilizaron callbacks como `ModelCheckpoint`, `ReduceLROnPlateau` y `EarlyStopping`.

El entrenamiento se ejecutó durante **50 épocas**, mostrando una convergencia estable y una mejora consistente en la precisión de validación. La tasa de aprendizaje se ajustó dinámicamente durante el entrenamiento.

  - **Precisión final de entrenamiento (Epoch 50):** \~89.04%
  - **Precisión final de validación (Epoch 48):** \~89.38%

-----

## 7\. Resultados y Evaluación

🎉 **¡Entrenamiento completado exitosamente\!**

El modelo fue evaluado en el conjunto de prueba, que no fue visto durante el entrenamiento.

  - 📊 **Precisión final en prueba: 89.73%**

### Reporte de Clasificación

```text
              precision    recall  f1-score   support

       angry       0.89      0.93      0.91       237
     disgust       0.89      0.92      0.90       237
     fearful       0.94      0.89      0.91       237
       happy       0.92      0.85      0.88       237
     neutral       0.93      0.96      0.94       198
         sad       0.92      0.86      0.89       237
   surprised       0.67      0.83      0.74        77

    accuracy                           0.90      1460
   macro avg       0.88      0.89      0.88      1460
weighted avg       0.90      0.90      0.90      1460
```

-----

## 8\. Modelo Guardado

  - 💾 El modelo final fue guardado en el archivo: `emotion_cnn_model.h5`

> **Advertencia de Keras:** Se recomienda usar el nuevo formato `.keras` en lugar del formato legacy `.h5` para guardar modelos. Por ejemplo: `model.save('my_model.keras')`.