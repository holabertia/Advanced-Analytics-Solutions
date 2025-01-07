#!/usr/bin/env python
# coding: utf-8

# # Modelos de Clustering - Aprendizaje No Supervisado
# 
# ## 1 Objetivos y Perspectiva general
# 
# En este notebook vamos a trabajar en un caso práctico de clasificaciones de tipos de pingüinos ([penguins](https://www.kaggle.com/datasets/youssefaboelwafa/clustering-penguins-species)) generando clusters.

# 
# <section style="width: 100%;"> 
# 	<div style="width: 50%; float: left; border-radius: 10px;"> 
#         El objetivo es diferenciar las especies de pinguinos que encontramos dentro del dataset. Para ello utilizaremos clustering, uno de los tipos de aorendizaje no supervisado.
#         Para ello contamos con los siguentes datos de 321 ejemplares de pinguinos, con la siguiente información:
#         <ol>
#             <li><em>culmen</em> o <em>bill length</em> (longitud del pico) en milímetros.</li>
#             <li><em>culmen</em> o <em>bill depth</em> (ancho del pico) en milímetros.</li>
#             <li><em>flipper length</em> (longitud de la aleta) en milímetros.</li>
#             <li><em>body mass</em> (masa corporal) en gramos.</li>
#             <li><em>sex</em> (sexo), macho o hembra.</li>
#         </ol>
# 	</div> <div style="width: 50%; float: left;">
# 		<img src="https://user-images.githubusercontent.com/54525819/139198017-769e8f61-2e58-48a9-947d-fd22947a6548.png" />
# 	 </div>
# </section>
# 

# ¡Comencemos!

# ## 2 Importación de librerías y paquetes

# In[32]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Modelos
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
# Metricas
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree

import warnings
warnings.filterwarnings("ignore")


# ## 3 Lectura y visualización de datos

# In[33]:


df = pd.read_csv(r".\datasets\penguins\penguins.csv")
df.head()


# Podemos ver que el dataset tiene multiples valores en `NULL`. Como son pocos, lo que haremos será borrar directamente las filas que los contengan.

# In[34]:


df.info()


# In[35]:


df.isnull().sum()


# In[36]:


df.dropna(inplace=True)

df.isnull().sum()


# Estos datos nos indican que *flipper_lenght_mm* puede estar teniendo _outlayers_, es decir, valores, que por algún motivo, estan fuera de rango (mala medición, error en la transcripción, valor no resentativo, etc.). Así pues, procedemos a eliminarlos para que no interfieran en la clasificación posterior.

# In[37]:


df.describe()


# In[38]:


q_low = df["flipper_length_mm"].quantile(0.01)
q_hi  = df["flipper_length_mm"].quantile(0.99)
df = df[(df['flipper_length_mm'] < q_hi) & (df['flipper_length_mm'] > q_low)]


# También hemos encontrado un valor erroneo en el sexo de los pinguinos, ya sea porque hay un error o no se pudo determinar, en el sexo del animal no deberia aparecer el atributo `'.'`.

# In[39]:


df['sex'].unique()


# In[40]:


df = df[~(df['sex'] == '.')]


# In[41]:


df.reset_index(drop=True, inplace=True)


# ## 4 Exploración de los datos (EDA)

# Una vez limipios los datos, podemos ver la distribución de los datos:

# In[42]:


fig = plt.figure(figsize=(25, 6))

for i, column in enumerate(df.columns[:-1]):
    sub = fig.add_subplot(1, 5, i + 1)
    sns.boxplot(data=df, y=column, color='powderblue')


# In[43]:


sns.countplot(x=df['sex'], hue=df['sex'], palette='RdYlBu')
plt.show()


# Las categorias con strings dan problemas al momento de trabajar con ellas en un modelo. Por ello simplemente cambiaremos el valor del sexo a un número que siga representando su valor. 

# In[44]:


df['sex'] = pd.factorize(df['sex'])[0]


# In[45]:


df.head()


# Una vez más, estandarizaremos los datos para asegurarnos de que aplicaremos la uniformidad necesaria para garantizar la efectividad de sus análisis.

# In[46]:


Scaler = StandardScaler()
scaled = Scaler.fit_transform(df)

df = pd.DataFrame(scaled, columns=df.columns)
df.head()


# Aplicaremos PCA para reducir el número de características para ser más fáciles de manejar. En este caso, estamos elegiendo el número óptimo de componentes principales para preservar la mayor varianza posible.

# In[47]:


#composición del PCA
pca = PCA(n_components=None)
pca_data = pca.fit(df)

pca_data.explained_variance_ratio_


# In[48]:


pca = PCA(n_components=2)
pca_data = pca.fit_transform(df)


# ## 5 Comparación de Modelos

# A continuación, procedemos a aplicar una serie de algoritmos de clústering con el objetivo de verificar cuál nos proporciona un resultado final óptimo para la clasificación de los grupos de pingüinos.mucho)

# ## K-Means
# El primer modelo que validaremos será el K-Means, el cuál consiste en buscar los centros de los cúmulos de datos y asociarlos con un centroide. El número de grupos que generamos lo decidimos nosotros. Una técnica para hacerlo es la de _elbow method_ o método del codo, donde probando diferentes números de centros, vemos la variación de los resultados. Una vez obtenidos, se genera un gráfico con una curva descendiente. El punto de más que quedaría situado en el codo es el número de grupos ideales para nuestro _dataset_.

# In[72]:


wcss = []   # within cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)


# In[74]:


plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color="powderblue")
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means Clustering')
plt.show()


# In[77]:


kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(pca_data)

plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('K-means Clustering')
plt.show()


# Por último, en la aplicación del Kmeans calcularemos el "Silhouette Score", una métrica que evalúa la calidad de un modelo de clustering. Esta métrica mide qué tan bien están separados los clusters y qué tan cohesionados están sus puntos internos. Es decir, indica qué tan adecuadamente cada punto de un cluster está agrupado con los puntos de su mismo cluster y qué tan lejos está de los puntos de otros clusters.

# In[52]:


silhouette_score(pca_data, kmeans.labels_)

print(f"The silhouette score for K-means algorithm is: {silhouette_score(pca_data, kmeans.labels_):.2f}")


# Hemos obtenido un resultado de 0.65 para el Silhouette Score, un valor realmente bueno, pues indica que los clusters están bien formados en la mayoría de los casos.

# # K-Medoids
# 
# Otro modelo muy popular en clústering es el K-Medoids. A diferencia de K-Means, K-Medoids usa como centro puntos conocidos de los datos, buscando cuál cumple con la menor distancia entre ellos. Esto los hace más resistentes al ruido o los valores anómalos, pero son más costosos computacionalmente.

# In[79]:


wcss = []   # within cluster sum of squares
for i in range(1, 11):
    kmedoids = KMedoids(n_clusters=i, random_state=42)
    kmedoids.fit(pca_data)
    wcss.append(kmedoids.inertia_)


# In[80]:


plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='powderblue')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-medoids Clustering')
plt.show()


# In[54]:


# kmedoids
kmedoids = KMedoids(n_clusters=4, random_state=42)
kmedoids.fit(pca_data)

# plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmedoids.labels_, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('K-medoids Clustering')
plt.show()


# Procedemos a calcular el "Silhouette Score" para evaluar la calidad del modelo. 

# In[55]:


# compute the silhouette score
silhouette_score(pca_data, kmedoids.labels_)
print(f"The silhouette score for K-medoids algorithm is: {silhouette_score(pca_data, kmedoids.labels_):.2f}")


# Al igual que en el modelo de K-Means, hemos obtenido un resultado de 0.65 en el Silhouette Score aplicando el modelo de K-Medoids, lo cual indica que los clusters están bien formados en la mayoría de los casos.

# ## Clustering Jerárquico
# 
# Clustering jerárquico es una técnica de aprendizaje no supervisado que organiza un conjunto de datos en grupos (clusters) siguiendo una estructura jerárquica, similar a un árbol. Su objetivo es agrupar elementos similares entre sí en diferentes niveles, de modo que los clusters más pequeños se integran en clusters más grandes a medida que avanzamos en la jerarquía.
# 
# Hay dos tipos de enfoques para crear una jerarquía de clusters:
# - Aglomerativo (de abajo a arriba).
# - Divisiva (de arriba a abajo).
# 
# El método **aglomerativo** comienza considerando cada punto de datos como un cluster independiente. Luego, el algoritmo va fusionando iterativamente los clusters más similares en un solo grupo, hasta que todos los datos forman un único cluster.
# 
# Por otro lado, el enfoque **divisivo** hace lo contrario: parte de un único cluster que contiene todos los datos y los va dividiendo en subgrupos más pequeños.
# 
# Hay varias formas de combinar los clusters calculando las distancias entre:
# 1. Centroides de cada cluster (average linkage o promedio de cada grupo).
# 2. Puntos más cercanos a cada cluster (single linkage o enlace único).
# 3. Puntos más lejanos de cada cluster (complete linkage o vínculo completo)
#    
# En el presente artículo nos centraremos en analizar el método aglomerativo en modelos de clustering jerárquico. 

# Empezamos nuestro análisis explorando el **single linkage**, el cual se basa en calcular la distancia mínima entre cualquier par de puntos, donde cada punto pertenece a uno de los dos clusters que se están comparando.

# In[81]:


# single linkage
single_linkage = linkage(pca_data, method='single', metric='euclidean')

dendrogram(single_linkage)
plt.title('Single Linkage')
plt.gca().set_xticklabels([])
plt.show()


# A continuación, analizaremos el método del **complete linkage**, el cual se basa en calcular la distancia máxima entre cualquier par de puntos, donde cada punto pertenece a uno de los dos clusters que se están comparando.

# In[82]:


# complete linkage

complete_linkage = linkage(pca_data, method='complete', metric='euclidean')

dendrogram(complete_linkage)
plt.gca().set_xticklabels([])
plt.title('Complete Linkage')
plt.show()


# Por último, analizaremos el **average linkage**, el cual  se basa en calcular la distancia promedio entre todos los pares de puntos, donde cada par contiene un punto de cada uno de los dos clusters que se están comparando.

# In[83]:


# average linkage

average_linkage = linkage(pca_data, method='average', metric='euclidean')

dendrogram(average_linkage)
plt.title('Average Linkage')
plt.gca().set_xticklabels([])
plt.show()


# In[61]:


cluster_labels = pd.Series(cut_tree(average_linkage, n_clusters=4).reshape(-1,))

# plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Hierarchical Clustering')
plt.show()


# Al igual que en los anteriores modelos, procedemos a calcular el "Silhouette Score" para evaluar la calidad del modelo

# In[62]:


# compute the silhouette score
silhouette_score(pca_data, cluster_labels)
print(f"The silhouette score for Hierarchical algorithm is: {silhouette_score(pca_data, cluster_labels):.2f}")


# Hemos obtenido un resultado de 0.65 para el Silhouette Score aplicando Clustering jerárquico, el cual indica que los clusters están bien formados en la mayoría de los casos.
# 
# Por lo tanto, aplicando los tres modelos hemos llegado a la conclusión de que los clusters / agrupaciones se componen de forma correcta. Sin embargo, sería necesario seguir haciendo pruebas y entrenando los modelos con más datos para llegar a obtener mejores resultados.
# 
# #### ¡Muchas gracias por leer nuestro artículo!
