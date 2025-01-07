#!/usr/bin/env python
# coding: utf-8

# Modelos de Clasifiacion

## 1 Importación de librerías y paquetes
# Librerías de análisis y tratamiento de datos
import pandas as pd
# Librerías para visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns
# Librerías para Modelos de ML
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Metricas de analisis
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
# Modulos para partición de datos
from sklearn.model_selection import train_test_split
# Otras Librerías y Módulos
import warnings
warnings.filterwarnings("ignore") 


## 2 Lectura y visualización de datos

# Importación del dataset
df = pd.read_csv(r".\datasets\mushroom\mushroom_cleaned.csv")
df.head()

# Eliminación de duplicados
df = df.drop_duplicates()

# Visualización de información clave de cada campo
df.info()


# Graficación de matriz de correlación de los campos del dataset
plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), cmap = "rocket_r", annot=True, fmt=".2f")
plt.show()


## 3 Exploración de los datos (EDA)

#Visualización de la distribución de los datos
fig = plt.figure(figsize=(30, 8))

for i, column in enumerate(df):
    sub = fig.add_subplot(2, 5, i + 1)
    sub.set_xlabel(column)
    df[column].plot(kind='hist', color='powderblue')
    sub.set_xlabel(column)


# Análisis de atributos discrtos
category = ['cap-shape', 'gill-attachment', 'gill-color', 'stem-color', 'season']
fig = plt.figure(figsize=(30, 4))


for i, column in enumerate(category):
    sub = fig.add_subplot(1, 5, i + 1)
    chart = sns.countplot(data=df, x=column, hue='class', palette='RdYlBu')


# Análisis de atributos continuos
continuous = ['cap-diameter', 'stem-height', 'stem-width']
fig = plt.figure(figsize=(25, 5))

for i, column in enumerate(continuous):
    sub = fig.add_subplot(1, 5, i + 1)
    sns.boxplot(data=df, x='class', y=column, hue='class', palette='RdYlBu')


## 4 Comparativa Modelos de Clasificación

# Extracción de las variables independientes (X) y la variable dependiente (y)
X = df.drop(['class'], axis=1)
y = df["class"]

# División del dataset en training y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


models = [LogisticRegression(), KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto'), SVC(), GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier()]
models_list = ["Logistic Regression", "KNN", "SVM", "Naive Bayes", "Decision Tree", "Random Forest"]
accuracies = []
precisions = []
recalls = []
aucs = []
confusion_matrixs = []
rocs = []

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    rocs.append((fpr, tpr))
    aucs.append(round(auc(fpr, tpr), 2))
    confusion_matrixs.append(confusion_matrix(y_test, y_pred)) 


## 5 Evaluación de Modelos

# Lista de modelos a evaluar
models_list = ["Logistic Regression", "KNN", "SVM", "Naive Bayes", "Decision Tree", "Random Forest"]

# Aplicación de la matriz de confusión para analalizar los resultados obtenidos
fig = plt.figure(figsize=(15, 9))

for i, model_name in enumerate(models_list):
    print(model_name)
    print(f"accuracy: {accuracies[i]}\t precision: {precisions[i]}\t recall: {recalls[i]}\t auc: {aucs[i]}")
    sub = fig.add_subplot(2, 3, i + 1).set_title(model_name)
    sns.heatmap(confusion_matrixs[i], cmap = "rocket_r", annot=True, fmt=".2f")


#Aplicación de la curva de ROC para analalizar los resultados 
fig = plt.figure(figsize=(15, 9))

for i, model_name in enumerate(models_list):
    sub = fig.add_subplot(2, 3, i + 1)
    sub.plot(rocs[i][0], rocs[i][1], color='powderblue', lw=2, label='ROC curve (area = %0.2f)' % aucs[i])
    sub.plot([0, 1], [0, 1], color='sandybrown', lw=2, linestyle='--', label='Random Guess')
    sub.set_xlim([0.0, 1.0])
    sub.set_ylim([0.0, 1.05])
    sub.set_xlabel('False Positive Rate')
    sub.set_ylabel('True Positive Rate')
    sub.set_title(model_name)
    sub.legend(loc='lower right')




