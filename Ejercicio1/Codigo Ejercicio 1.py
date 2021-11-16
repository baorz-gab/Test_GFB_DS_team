# Tratamiento de datos
import pandas as pd
import numpy as np
from datetime import datetime
# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns

import warnings        
# ignorar 
warnings.filterwarnings("ignore") # if there is a warning after some codes, this will avoid us to see them.
plt.style.use('ggplot')

#Cargango datos
url_agosto = "https://www.ecobici.cdmx.gob.mx/sites/default/files/data/usages/2011-08.csv" #url de los datos para el mes de agosto
url_septiembre = "https://www.ecobici.cdmx.gob.mx/sites/default/files/data/usages/2011-09.csv" #url de los datos para el mes de septiembre
url_octubre = "https://www.ecobici.cdmx.gob.mx/sites/default/files/data/usages/2011-10.csv" #url de los datos para el mes de octubre

#leemos el csv y lo guardamos en una variable
agosto = pd.read_csv(url_agosto)
septiembre = pd.read_csv(url_septiembre)
octubre = pd.read_csv(url_octubre)

#convertimos en un solo dataframe
datos = pd.concat([agosto, septiembre, octubre], ignore_index=True)
#Información-> nombre de col, tipo de dato, si hay datos faltantes
datos.info()

datos.head() #visualización de como es el dataframe

datos['Hora_Retiro'] =datos.Hora_Retiro.astype('str') #transformamos primero los datos de esta columna en cadena 
datos['Hora_Retiro'] =datos.Hora_Retiro.str.slice(stop=2) #cortamos la cadena hasta el indice 2

datos['Hora_Arribo'] =datos.Hora_Arribo.astype('str') #transformamos primero los datos de esta columna en cadena
datos['Hora_Arribo'] =datos.Hora_Arribo.str.slice(stop=2) #cortamos la cadena hasta el indice 2

#Convertimos los datos de la columna en enteros
datos['Hora_Retiro'] =datos.Hora_Retiro.astype('int')
datos['Hora_Arribo'] =datos.Hora_Arribo.astype('int')

#Tambien pasamos los datos de las columnas de fecha a datetime
datos['Fecha_Retiro'] = pd.to_datetime(datos.Fecha_Retiro)
datos['Fecha_Arribo'] = pd.to_datetime(datos.Fecha_Arribo)

datos

boxplot = datos.boxplot(column=['Ciclo_Estacion_Retiro','Ciclo_Estacion_Arribo']) #diagrama de cajas para saber si existen estaciones registradas fuera del rango

#Delimitamos para tomar los datos que tienen como cicloestacion el numero 90 o menor
datos = datos.loc[datos['Ciclo_Estacion_Retiro'] <= 90]
datos = datos.loc[datos['Ciclo_Estacion_Arribo'] <= 90]

#varificamos que no existan mas datos fuera del limite
boxplot = datos.boxplot(column=['Ciclo_Estacion_Retiro','Ciclo_Estacion_Arribo'])

#Pregunta 1

#Creamos nuevas columnas para los dia de la semana donde 0 = lunes, 1 = martes, ..., 6 = domingo
datos['Dia_Semana_Arribo'] = datos.Fecha_Arribo.dt.dayofweek #En que día de la semana cayo la fecha
datos['Dia_Semana_Retiro'] = datos.Fecha_Retiro.dt.dayofweek #En que día de la semana cayo la fecha

retiro = pd.crosstab(index=datos['Hora_Retiro'], columns=datos['Dia_Semana_Retiro']) # tabla con las frecuencias de uso: ej. para el lunes cuantos retiros hubo a las 7
arribo = pd.crosstab(index=datos['Hora_Arribo'], columns=datos['Dia_Semana_Arribo']) # tabla con las frecuencias de uso: ej. para el lunes cuantos retiros hubo a las 7
fig, ax = plt.subplots(1,2, figsize=(15,8)) #definimos tamaño de gráfica y el eje
sns.heatmap(retiro, annot=True, fmt="d", linewidths=.5, ax=ax[0]) #gráfica de calor
sns.heatmap(arribo, annot=True, fmt="d", linewidths=.5, ax=ax[1]) #gráfica de calor
fig.show() #imprimimos

#vamos a calcular un histograma para ver que cicloestacion es la mas frecuentada

intervalos = range(datos.Ciclo_Estacion_Arribo.min(), datos.Ciclo_Estacion_Arribo.max() + 2) #determinamos el rango para la gráfica del eje y
sns.displot(datos.Ciclo_Estacion_Arribo, color='#F2AB6D', bins=intervalos, kde=True) #creamos el gráfico en Seaborn

#configuramos en Matplotlib
plt.ylabel('Frecuencia')
plt.xlabel('Ciclo Estaciones Arribo')
plt.title('Histograma')

plt.show()

#Hacemos lo mismo para Ciclo_Estaciones_Retiro 
intervalos = range(datos.Ciclo_Estacion_Retiro.min(), datos.Ciclo_Estacion_Retiro.max() + 2) #determinamos el rango para la gráfica del eje y
sns.displot(datos.Ciclo_Estacion_Retiro, color="y", bins=intervalos, kde=True) #creamos el gráfico en Seaborn
#configuramos en Matplotlib
plt.ylabel('Frecuencia')
plt.xlabel('Ciclo Estaciones Retiro')
plt.title('Histograma')
plt.show()

#Quitamos las fechas, genero y bici ya que esos datos no nos interesan por el momento
df = datos.iloc[:, 3:11] 
df = df.drop(['Fecha_Retiro', 'Fecha_Arribo'], axis = 1) 
#Agrupamos por dia de la semana
df.groupby('Dia_Semana_Arribo').apply(pd.DataFrame.mode)

df.groupby('Dia_Semana_Retiro').apply(pd.DataFrame.mode)

#Pregunta 2

df_fechas = datos.iloc[:, 3:5] #obtenemos las columnas Fecha_Retiro y Ciclo_Estacion_Retiro
#Agrupamos con respecto a fechas con frecuencia, en el agrupamiento se reporta la moda para la columna Ciclo_Estacion_Retiro
df_fechas = df_fechas.groupby(pd.Grouper(key = 'Fecha_Retiro', freq = 'D')).apply(pd.DataFrame.mode)

df_fechas2 = datos.iloc[:, 6:8]  #obtenemos las columnas Fecha_Retiro y Ciclo_Estacion_Arribo
#Agrupamos con respecto a fechas con frecuencia, en el agrupamiento se reporta la moda para la columna Ciclo_Estacion_Arribo
df_fechas2 = df_fechas2.groupby(pd.Grouper(key = 'Fecha_Arribo', freq = 'D')).apply(pd.DataFrame.mode)

#Graficamos los datos obtenidos por el agrupamiento
plt.plot(df_fechas.Fecha_Retiro, df_fechas.Ciclo_Estacion_Retiro)
plt.show()

#Graficamos los datos obtenidos por el agrupamiento
plt.plot(df_fechas.Fecha_Retiro, df_fechas.Ciclo_Estacion_Retiro)
plt.show()

#Pregunta 3

estaciones = pd.crosstab(index=datos['Ciclo_Estacion_Retiro'], columns=datos['Ciclo_Estacion_Arribo']) # tabla con frecuencias ej: Cuantas veces se sale de la estacion 1 y se llega a la estacion 2
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(estaciones, annot=True, fmt="d", linewidths=.5)
fig.show()

#Pregunta 4

#Graficamos el histograma de cada columna
plt.rcParams['figure.figsize'] = (16, 9)
df.hist()
plt.show()
#Escalamos nuestros datos
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler() #metodo de transformación
X = ms.fit_transform(df)
from sklearn.cluster import KMeans
cs = [] 
#funcion mara metodo del codo
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()

from sklearn.cluster import KMeans
#Funcion para hacer nuestro agrupamiento mediante KMeans con numero de clusters = 3
kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(X)

# comprobamos cuántas de las muestras se etiquetaron correctamente
labels = kmeans.labels_
#Unimos nuestro resultado a los datos 
clusters=pd.concat([df, pd.DataFrame({'cluster':labels})], axis=1)
clusters.head()

for c in clusters:
    grid= sns.FacetGrid(clusters, col='cluster')
    grid.map(plt.hist, c)

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
#Definimos cuantos PCA queremos obtener y en este caso son 2 ya que queremos tener un resultado 2D
pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(X)  
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 
X_principal.head(2)

x, y = X_principal['P1'], X_principal['P2'] #definimos variables para cada columna del PCA
#Grafica del PCA con respecto a los cluster
colors = {0: 'red',
          1: 'blue',
          2: 'green'}

names = {0: 'Personas que usan el servicio en la mañana y tarde entre semana', 
         1: 'Personas que usan el servicio los fines de semana en la tarde', 
         2: 'Peronsas que usan el servicio en la mañana y tarde entre semana ',}
  
df = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(20, 13)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,
            color=colors[name],label=names[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.legend()
ax.set_title("Clusters")
plt.show()
