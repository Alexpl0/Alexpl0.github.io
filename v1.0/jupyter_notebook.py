{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Análisis Fundamental de Modelos de Regresión y Optimización en Inteligencia Artificial\n",
    "\n",
    "Este notebook implementa y visualiza los conceptos teóricos presentados en el documento \"Análisis Fundamental de Modelos de Regresión y Optimización en Inteligencia Artificial\"."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Importamos las bibliotecas necesarias\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.preprocessing import PolynomialFeatures\n",
    "from sklearn.pipeline import make_pipeline\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Configuramos el estilo de las visualizaciones\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "sns.set_context(\"notebook\", font_scale=1.5)\n",
    "\n",
    "# Para reproducibilidad\n",
    "np.random.seed(42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Introducción a la Regresión en el Aprendizaje Automático\n",
    "\n",
    "### Definición y Propósito de la Regresión en el Contexto de la IA\n",
    "\n",
    "La regresión, en el ámbito del aprendizaje automático (ML) y la inteligencia artificial (IA), constituye una técnica estadística esencial cuyo objetivo primordial es predecir el valor de una variable dependiente, comúnmente denominada \"etiqueta\" o \"variable de respuesta\", a partir de una o más variables independientes, conocidas como \"características\" o \"variables predictoras\".\n",
    "\n",
    "Este proceso implica la identificación y modelado de la relación matemática subyacente entre estas variables, lo que permite pronosticar valores desconocidos basándose en datos ya disponibles y relacionados.\n",
    "\n",
    "La capacidad de transformar datos brutos en inteligencia empresarial y conocimientos accionables subraya la importancia de la regresión en diversas industrias."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### La Importancia de Modelar Relaciones entre Variables\n",
    "\n",
    "La modelización de relaciones entre variables es crucial debido a la simplicidad y la interpretabilidad que ofrecen los modelos de regresión, proporcionando una fórmula matemática clara para entender las conexiones en los datos. Estos modelos son ampliamente utilizados para el análisis preliminar de datos y la previsión de tendencias futuras en campos tan diversos como la biología, las ciencias del comportamiento, las ciencias ambientales y las ciencias sociales.\n",
    "\n",
    "Un aspecto distintivo del aprendizaje automático es el cambio de un enfoque estadístico tradicional a uno algorítmico. En ML, los algoritmos informáticos analizan grandes conjuntos de datos y, a partir de ellos, calculan la ecuación de regresión lineal. Esta metodología difiere de la estadística clásica, donde las ecuaciones a menudo se derivan de supuestos teóricos y luego se aplican. En ML, el algoritmo aprende los parámetros y la ecuación directamente de los datos, a menudo a través de procesos iterativos, sin una programación explícita de cada coeficiente."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Tipos de \"Rectas\" en el Análisis de Datos y Ajuste de Modelos\n",
    "\n",
    "### 2.1. Regresión Lineal: El Modelo Fundamental\n",
    "\n",
    "#### Concepto y Ecuación\n",
    "\n",
    "La regresión lineal asume una relación directa y lineal entre las variables, buscando ajustar una línea recta que mejor represente esta relación en un diagrama de dispersión. En el contexto del aprendizaje automático, la ecuación de un modelo de regresión lineal simple se expresa comúnmente como: \n",
    "\n",
    "y' = b + w₁x₁\n",
    "\n",
    "En esta formulación, y' representa la etiqueta predicha (la salida deseada), y x₁ es la característica (la entrada)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Interpretación de los Parámetros: Peso (w) y Sesgo (b)\n",
    "\n",
    "El término w₁ (peso) es el concepto equivalente a la pendiente (m) en la ecuación algebraica y = mx + b. Este parámetro indica la magnitud y dirección en que cambia y' por cada unidad de cambio en x₁.\n",
    "\n",
    "Por otro lado, b (sesgo) es el concepto equivalente a la intersección con el eje y (b o a) en la ecuación algebraica, representando el valor de y' cuando x₁ es cero.\n",
    "\n",
    "Durante el proceso de entrenamiento del modelo, los valores óptimos de w y b se calculan para minimizar la función de pérdida, lo que resulta en el \"mejor modelo\" posible."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Implementación de Regresión Lineal Simple\n",
    "\n",
    "Vamos a implementar una regresión lineal simple tanto manualmente como usando scikit-learn. Primero, generaremos algunos datos sintéticos con una relación lineal y algo de ruido."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Generamos datos sintéticos con una relación lineal y ruido\n",
    "X = np.linspace(0, 10, 100).reshape(-1, 1)  # Valores de x entre 0 y 10\n",
    "y_true = 2 * X.ravel() + 3  # Relación lineal verdadera: y = 2x + 3\n",
    "y = y_true + np.random.normal(0, 1, size=X.shape[0])  # Añadimos ruido gaussiano\n",
    "\n",
    "# Visualizamos los datos\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.scatter(X, y, alpha=0.7, label='Datos con ruido')\n",
    "plt.plot(X, y_true, 'r-', linewidth=2, label='Relación verdadera (y = 2x + 3)')\n",
    "plt.xlabel('x')\n",
    "plt.ylabel('y')\n",
    "plt.title('Datos Sintéticos para Regresión Lineal')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Implementación Manual de la Regresión Lineal (OLS)\n",
    "\n",
    "Ahora, implementaremos el método de Mínimos Cuadrados Ordinarios (OLS) manualmente para encontrar los parámetros w (pendiente) y b (intersección)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Implementación manual del método OLS para regresión lineal\n",
    "def manual_ols(x, y):\n",
    "    \"\"\"Calcula los parámetros w (pendiente) y b (intercepto) usando OLS.\"\"\"\n",
    "    # Aseguramos que x e y sean arrays unidimensionales\n",
    "    x = x.ravel()\n",
    "    y = y.ravel()\n",
    "    \n",
    "    # Calculamos las medias\n",
    "    x_mean = np.mean(x)\n",
    "    y_mean = np.mean(y)\n",
    "    \n",
    "    # Calculamos el numerador y denominador para la pendiente\n",
    "    numerator = np.sum((x - x_mean) * (y - y_mean))\n",
    "    denominator = np.sum((x - x_mean) ** 2)\n",
    "    \n",
    "    # Calculamos la pendiente (w)\n",
    "    w = numerator / denominator\n",
    "    \n",
    "    # Calculamos el intercepto (b)\n",
    "    b = y_mean - w * x_mean\n",
    "    \n",
    "    return w, b\n",
    "\n",
    "# Aplicamos nuestro método manual de OLS\n",
    "w_manual, b_manual = manual_ols(X, y)\n",
    "print(f'Parámetros calculados manualmente:\\nPendiente (w): {w_manual:.4f}\\nIntercepto (b): {b_manual:.4f}')\n",
    "\n",
    "# Calculamos las predicciones con nuestros parámetros\n",
    "y_pred_manual = w_manual * X.ravel() + b_manual\n",
    "\n",
    "# Visualizamos los resultados\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.scatter(X, y, alpha=0.7, label='Datos con ruido')\n",
    "plt.plot(X, y_true, 'r-', linewidth=2, label='Relación verdadera (y = 2x + 3)')\n",
    "plt.plot(X, y_pred_manual, 'b--', linewidth=2, label=f'Regresión manual (y = {w_manual:.4f}x + {b_manual:.4f})')\n",
    "plt.xlabel('x')\n",
    "plt.ylabel('y')\n",
    "plt.title('Regresión Lineal con OLS Manual')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Usando scikit-learn para Regresión Lineal\n",
    "\n",
    "Ahora, resolveremos el mismo problema utilizando la biblioteca scikit-learn, que implementa regresión lineal de forma optimizada."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Creamos y entrenamos un modelo de regresión lineal con scikit-learn\n",
    "model_sklearn = LinearRegression()\n",
    "model_sklearn.fit(X, y)\n",
    "\n",
    "# Obtenemos los parámetros aprendidos\n",
    "w_sklearn = model_sklearn.coef_[0]\n",
    "b_sklearn = model_sklearn.intercept_\n",
    "print(f'Parámetros calculados con scikit-learn:\\nPendiente (w): {w_sklearn:.4f}\\nIntercepto (b): {b_sklearn:.4f}')\n",
    "\n",
    "# Realizamos predicciones\n",
    "y_pred_sklearn = model_sklearn.predict(X)\n",
    "\n",
    "# Visualizamos los resultados\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.scatter(X, y, alpha=0.7, label='Datos con ruido')\n",
    "plt.plot(X, y_true, 'r-', linewidth=2, label='Relación verdadera (y = 2x + 3)')\n",
    "plt.plot(X, y_pred_sklearn, 'g--', linewidth=2, label=f'Scikit-learn (y = {w_sklearn:.4f}x + {b_sklearn:.4f})')\n",
    "plt.xlabel('x')\n",
    "plt.ylabel('y')\n",
    "plt.title('Regresión Lineal con scikit-learn')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Comparando los resultados\n",
    "\n",
    "Vamos a comparar ambos métodos (manual y scikit-learn) para verificar que obtenemos resultados similares."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Calculamos el error cuadrático medio (MSE) para ambos métodos\n",
    "mse_manual = mean_squared_error(y, y_pred_manual)\n",
    "mse_sklearn = mean_squared_error(y, y_pred_sklearn)\n",
    "\n",
    "print(f'Error Cuadrático Medio (MSE):\\nMétodo Manual: {mse_manual:.4f}\\nScikit-learn: {mse_sklearn:.4f}')\n",
    "\n",
    "# Visualizamos ambos modelos juntos\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.scatter(X, y, alpha=0.7, label='Datos con ruido')\n",
    "plt.plot(X, y_true, 'r-', linewidth=2, label='Relación verdadera (y = 2x + 3)')\n",
    "plt.plot(X, y_pred_manual, 'b--', linewidth=2, label=f'Manual (y = {w_manual:.4f}x + {b_manual:.4f})')\n",
    "plt.plot(X, y_pred_sklearn, 'g--', linewidth=2, label=f'Scikit-learn (y = {w_sklearn:.4f}x + {b_sklearn:.4f})')\n",
    "plt.xlabel('x')\n",
    "plt.ylabel('y')\n",
    "plt.title('Comparación de Métodos de Regresión Lineal')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.2. Regresión Polinomial: Capturando Relaciones No Lineales\n",
    "\n",
    "#### Extensión de la Regresión Lineal para Datos Curvos\n",
    "\n",
    "La regresión polinomial es una extensión crucial de la regresión lineal, diseñada específicamente para modelar relaciones no lineales en los datos, es decir, cuando la relación entre las variables no puede ser representada por una línea recta.\n",
    "\n",
    "A diferencia de la regresión lineal, que se limita a ajustar líneas rectas, la regresión polinomial permite que la curva se \"doble\", capturando patrones curvos al ajustar ecuaciones polinomiales de grados superiores.\n",
    "\n",
    "#### Ecuación Polinomial y el Concepto de Grado\n",
    "\n",
    "La ecuación matemática de un modelo de regresión polinomial extiende la ecuación lineal añadiendo términos polinomiales de mayor grado. La forma general es: \n",
    "\n",
    "y = b₀ + b₁x + b₂x² +... + bₙxⁿ + ε\n",
    "\n",
    "En esta ecuación, n representa el grado polinomial, que determina la complejidad de la curva que el modelo puede ajustar. A medida que el grado n aumenta, el modelo gana flexibilidad para capturar patrones de datos más intrincados."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Implementación de Regresión Polinomial\n",
    "\n",
    "Vamos a generar datos con una relación no lineal y aplicar regresión polinomial con diferentes grados para ver cómo se ajustan a los datos."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Generamos datos con una relación no lineal (cuadrática) y ruido\n",
    "X_nonlinear = np.linspace(0, 10, 100).reshape(-1, 1)\n",
    "y_true_nonlinear = 1 + 2 * X_nonlinear.ravel() - 0.3 * X_nonlinear.ravel()**2  # Relación cuadrática\n",
    "y_nonlinear = y_true_nonlinear + np.random.normal(0, 1.5, size=X_nonlinear.shape[0])  # Añadimos ruido\n",
    "\n",
    "# Visualizamos los datos no lineales\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.scatter(X_nonlinear, y_nonlinear, alpha=0.7, label='Datos con ruido')\n",
    "plt.plot(X_nonlinear, y_true_nonlinear, 'r-', linewidth=2, label='Relación verdadera (cuadrática)')\n",
    "plt.xlabel('x')\n",
    "plt.ylabel('y')\n",
    "plt.title('Datos No Lineales para Regresión Polinomial')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Aplicamos regresión polinomial con diferentes grados\n",
    "degrees = [1, 2, 5, 15]  # Probaremos diferentes grados polinomiales\n",
    "plt.figure(figsize=(14, 10))\n",
    "\n",
    "X_test = np.linspace(0, 10, 100).reshape(-1, 1)  # Puntos para evaluar el modelo\n",
    "\n",
    "for i, degree in enumerate(degrees):\n",
    "    # Creamos un pipeline de transformación polinomial + regresión lineal\n",
    "    polynomial_model = make_pipeline(\n",
    "        PolynomialFeatures(degree=degree, include_bias=False),\n",
    "        LinearRegression()\n",
    "    )\n",
    "    \n",
    "    # Entrenamos el modelo\n",
    "    polynomial_model.fit(X_nonlinear, y_nonlinear)\n",
    "    \n",
    "    # Realizamos predicciones\n",
    "    y_pred_poly = polynomial_model.predict(X_test)\n",
    "    \n",
    "    # Calculamos el MSE\n",
    "    mse_poly = mean_squared_error(y_nonlinear, polynomial_model.predict(X_nonlinear))\n",
    "    \n",
    "    # Creamos un subplot para este grado\n",
    "    plt.subplot(2, 2, i+1)\n",
    "    plt.scatter(X_nonlinear, y_nonlinear, alpha=0.7, label='Datos con ruido')\n",
    "    plt.plot(X_nonlinear, y_true_nonlinear, 'r-', linewidth=2, label='Relación verdadera')\n",
    "    plt.plot(X_test, y_pred_poly, 'g--', linewidth=2, label=f'Polinomio grado {degree}')\n",
    "    plt.xlabel('x')\n",
    "    plt.ylabel('y')\n",
    "    plt.title(f'Grado {degree}, MSE: {mse_poly:.4f}')\n",
    "    plt.legend()\n",
    "    plt.grid(True)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.3. Overfitting (Sobreajuste): El Riesgo de la Complejidad del Modelo\n",
    "\n",
    "#### Definición y Analogía del \"Memorizar\" el Conjunto de Entrenamiento\n",
    "\n",
    "El sobreajuste ocurre cuando un modelo se crea para que coincida (o \"memorice\") el conjunto de entrenamiento tan precisamente que falla en hacer predicciones precisas sobre datos nuevos, no vistos.\n",
    "\n",
    "Un modelo sobreajustado es análogo a una invención que funciona perfectamente en un entorno de laboratorio, pero resulta inútil en aplicaciones del mundo real porque no logra adaptarse a condiciones no vistas.\n",
    "\n",
    "Este fenómeno se manifiesta cuando un modelo se vuelve excesivamente sensible a los detalles específicos y al ruido en los datos de entrenamiento, aprendiendo patrones que no son significativos para la generalización a nuevos datos.\n",
    "\n",
    "#### Causas Comunes del Sobreajuste\n",
    "\n",
    "Varias causas contribuyen al sobreajuste:\n",
    "\n",
    "1. **Conjunto de entrenamiento insuficiente**: Si el conjunto de datos de entrenamiento es demasiado pequeño, puede que no represente adecuadamente todos los escenarios que el modelo encontrará en el mundo real, limitando su capacidad de generalización.\n",
    "\n",
    "2. **Datos inexactos, erróneos o irrelevantes**: Los errores o el ruido en los datos de entrenamiento pueden llevar al modelo a aprender patrones que no reflejan las relaciones subyacentes verdaderas.\n",
    "\n",
    "3. **Pesos grandes**: En los modelos de ML, los pesos numéricos que representan la importancia de las características pueden volverse desproporcionadamente grandes. Esto hace que el modelo sea hipersensible a ciertas características, incluido el ruido.\n",
    "\n",
    "4. **Sobreentrenamiento**: Entrenar un modelo durante demasiado tiempo en los mismos datos puede hacer que memorice puntos de datos específicos en lugar de aprender las relaciones subyacentes.\n",
    "\n",
    "5. **Arquitectura del modelo demasiado sofisticada**: Las arquitecturas de modelo excesivamente complejas tienen una mayor capacidad para capturar detalles finos, pero también aumentan la probabilidad de aprender ruido o detalles irrelevantes."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Demostración del Sobreajuste con Regresión Polinomial\n",
    "\n",
    "Vamos a demostrar el sobreajuste dividiendo nuestros datos en conjuntos de entrenamiento y prueba, y ajustando modelos polinomiales de diferentes grados."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Dividimos los datos en conjuntos de entrenamiento y prueba\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X_nonlinear, y_nonlinear, test_size=0.3, random_state=42\n",
    ")\n",
    "\n",
    "# Ordenamos los datos para facilitar la visualización\n",
    "sort_idx_train = np.argsort(X_train.ravel())\n",
    "X_train_sorted = X_train[sort_idx_train]\n",
    "y_train_sorted = y_train[sort_idx_train]\n",
    "\n",
    "sort_idx_test = np.argsort(X_test.ravel())\n",
    "X_test_sorted = X_test[sort_idx_test]\n",
    "y_test_sorted = y_test[sort_idx_test]\n",
    "\n",
    "# Definimos un rango de grados polinomiales para probar\n",
    "degrees = [1, 2, 5, 15]\n",
    "\n",
    "# Creamos una figura para comparar los diferentes grados\n",
    "plt.figure(figsize=(14, 10))\n",
    "\n",
    "# Para guardar resultados para una tabla comparativa\n",
    "results = []\n",
    "\n",
    "for i, degree in enumerate(degrees):\n",
    "    # Creamos y entrenamos el modelo polinomial\n",
    "    polynomial_model = make_pipeline(\n",
    "        PolynomialFeatures(degree=degree, include_bias=False),\n",
    "        LinearRegression()\n",
    "    )\n",
    "    polynomial_model.fit(X_train, y_train)\n",
    "    \n",
    "    # Realizamos predicciones en datos de entrenamiento y prueba\n",
    "    y_train_pred = polynomial_model.predict(X_train)\n",
    "    y_test_pred = polynomial_model.predict(X_test)\n",
    "    \n",
    "    # Calculamos MSE para entrenamiento y prueba\n",
    "    train_mse = mean_squared_error(y_train, y_train_pred)\n",
    "    test_mse = mean_squared_error(y_test, y_test_pred)\n",
    "    \n",
    "    # Guardamos resultados\n",
    "    results.append([degree, train_mse, test_mse])\n",
    "    \n",
    "    # Visualizamos los resultados\n",
    "    plt.subplot(2, 2, i+1)\n",
    "    \n",
    "    # Generamos una curva suave para las predicciones\n",
    "    X_smooth = np.linspace(0, 10, 100).reshape(-1, 1)\n",
    "    y_smooth_pred = polynomial_model.predict(X_smooth)\n",
    "    \n",
    "    # Graficamos datos y predicciones\n",
    "    plt.scatter(X_train, y_train, color='blue', alpha=0.7, label='Train')\n",
    "    plt.scatter(X_test, y_test, color='red', alpha=0.7, label='Test')\n",
    "    plt.plot(X_smooth, y_smooth_pred, 'g-', label=f'Modelo (grado {degree})')\n",
    "    plt.plot(X_nonlinear, y_true_nonlinear, 'k--', alpha=0.5, label='Verdadero')\n",
    "    \n",
    "    plt.title(f'Grado {degree}\\nTrain MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')\n",
    "    plt.xlabel('x')\n",
    "    plt.ylabel('y')\n",
    "    plt.legend()\n",
    "    plt.grid(True)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Creamos una tabla con los resultados\n",
    "results_df = pd.DataFrame(results, columns=['Grado', 'MSE Entrenamiento', 'MSE Prueba'])\n",
    "results_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Visualización de la Curva de Generalización\n",
    "\n",
    "Vamos a observar cómo evolucionan los errores de entrenamiento y prueba a medida que aumentamos el grado del polinomio."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Probamos un rango más amplio de grados polinomiales\n",
    "degrees_range = range(1, 21)\n",
    "train_errors = []\n",
    "test_errors = []\n",
    "\n",
    "for degree in degrees_range:\n",
    "    polynomial_model = make_pipeline(\n",
    "        PolynomialFeatures(