#!/usr/bin/env python
# coding: utf-8

# # Modelos de Regresión

# ## 1 Importación de librerías y paquetes

# Librerías de análisis y tratamiento de datos
import pandas as pd
# Librerías para visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns
# Modulos para partición de datos
from sklearn.model_selection import train_test_split
# Librerías para Modelos de ML
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy import stats 
# Metricas de analisis
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


# ## 2 Lectura y visualización de datos

# Importación del dataset
df = pd.read_csv(r".\datasets\happiness\2019.csv")
# Eliminación de columnas innecesarias
df = df.drop(['Overall rank','Country or region'],axis=1)
df.head()


# Visualización de información clave de cada campo
df.info()


# ## 3 Exploración de los datos (EDA)

# Graficación de matriz de confusión para análisis del dataset
plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), cmap = "rocket_r", annot=True, fmt=".2f")
plt.show()


# Análisis de la distribución de los datos
fig = plt.figure(figsize=(30, 8))

for i, column in enumerate(df):
    sub = fig.add_subplot(2, 4, i + 1)
    sub.set_xlabel(column)
    df[column].plot(kind='kde', color='powderblue')
    sub.set_xlabel(column)


# Normalización de los datos para poder extraer conclusiones más certeras
scaler = StandardScaler()

df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])


# Análisis de la distribución de los datos aplicando normalización en los datos
fig = plt.figure(figsize=(30, 8))

for i, column in enumerate(df):
    sub = fig.add_subplot(2, 4, i + 1)
    sub.set_xlabel(column)
    df[column].plot(kind='kde', color='powderblue')
    sub.set_xlabel(column)


# Análisis de la distribución de los datos en base al Score
target_attribute = 'Score' 
other_columns = [col for col in df.columns if col != target_attribute]

fig = plt.figure(figsize=(25, 10))

for i, column in enumerate(other_columns):
    df[column] = stats.zscore(df[column])
    sub = fig.add_subplot(2, 3, i + 1)
    sub.scatter(df[target_attribute], df[column], color='powderblue')
    sub.set_xlabel(target_attribute)
    sub.set_ylabel(column)
    sub.set_title(f'{target_attribute} vs {column}')


# ## 4 Entrenamiento y Comparativa de los modelos

# Extracción de las variables independientes (X) y la variable dependiente (y)
X = df.drop(['Score', 'Generosity', 'Perceptions of corruption'], axis=1)
y = df["Score"]

# División del dataset en training y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Listado de modelos a entrenar
models = [LinearRegression(), Lasso(alpha=0.001), Ridge(alpha = 0.1), make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())]
models_list = ["Linear Regression", "Lasso Regression", "Ridge Regression", "Polynomial Regression"]
mserrors = []
r2s = []
preds = []
# Entrenamiento de modelos y obteción de valores de predicción
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    preds.append(y_pred)


# ## 5 Evaluación de los resultados de los modelos

# Definición de métricas de evaluación
metrics = []

for name, y_pred in zip(models_list, preds):
    mse_score = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics.append({'Model': name, 'MSE': mse_score, 'R2': r2})

metrics_df = pd.DataFrame(metrics)

print(metrics_df)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

# Graficación de los resultados
sns.barplot(x='Model', y='MSE', data=metrics_df, hue='Model', palette='RdYlBu', ax=ax[0])
ax[0].set_title("Comparison of MSE between Models")
ax[0].tick_params(axis='x', labelsize=8)

sns.barplot(x='Model', y='R2', data=metrics_df, hue='Model', palette='RdYlBu', ax=ax[1])
ax[1].set_title("Comparison of R² between Models")
ax[1].tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.show()


# Graficacion del rendimiento de los modelos
plt.figure(figsize=(6, 4))
colors = ['powderblue', 'sandybrown', '#fdff52', '#9dc100']
i = 0
for name, model in zip(models_list, models):
    if name != 'Polynomial Regression':
        plt.plot(model.coef_, color=colors[i], label=name)
        i += 1
plt.title("Model Coefficients")
plt.xlabel("Features")
plt.ylabel("Coefficient")
plt.legend()
plt.show()


# Graficación de los resultados de predicción vs valores reales
plt.figure(figsize=(6, 4))
i = 0
for name, y_pred in zip(models_list, preds):
    sns.scatterplot(x=y_test, y=y_pred, label=name, color=colors[i])
    i += 1
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
plt.title("Predictions vs. Actual Valuess")
plt.xlabel("Actual Values")
plt.ylabel("Predictions")
plt.legend()
plt.show()


# Graficación de los resultados obtenidos del modelo de regresión polinómica
plt.figure(figsize=(6, 4))
i = 0
for name, y_pred in zip(models_list, preds):
    errors = y_test - y_pred
    sns.scatterplot(x=y_pred, y=errors, label=name, color=colors[i])
    i += 1
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Prediction Errors")
plt.xlabel("Predictions")
plt.ylabel("Errors (Actual Value - Prediction)")
plt.legend()
plt.show()


# #### ¡Muchas gracias por leer nuestro artículo!
