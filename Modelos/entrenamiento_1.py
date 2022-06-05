# from os import name
import time
import tensorflow as tf
import tensorflow_addons as tfa
import keras
import numpy as np
import cv2
from sklearn.model_selection import KFold
import os
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer, Input, Conv2D, MaxPool2D, Reshape, Dense, Flatten


def cargarDatos(rutaOrigen, numeroCategorias, ancho, alto):
    imagenesCargadas = []
    valorEsperado = []
    for categoria in range(1, numeroCategorias):
        dir=rutaOrigen + str(categoria)
        files = os.listdir(dir)
        for file_name in files:
            ruta=dir+ "/"+file_name
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (ancho, alto), interpolation = cv2.INTER_AREA)
            imagen = imagen.flatten()
            imagen = imagen / 255
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias)
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados

#################Implementación del modelo ####################
#Definir dimension imagen
width = 128
height = 128
pixeles = width * height
num_channels = 1        #Si imagen blanco/negro = 1     rgb = 3
img_shape = (width, height, num_channels)
#cantidad elementos clasificar
num_class = 10
dirc="E:/david/U PC/2022-1/Inteligentes/Parte 2/MachineLEarning/Parcial/Crops/cartas_dataset/"
#CargaImagen
imagenes,probabilidades = cargarDatos(dirc+"train/", num_class, width, height)
imagenesPrueba,probabilidadesPrueba = cargarDatos(dirc+"test/", num_class, width, height)



model = Sequential()
#Capa de entrada
model.add(InputLayer(input_shape= (pixeles,)))
#Rearmar la imagen
model.add(Reshape(img_shape))

#Convolucional Layer
model.add(Conv2D(kernel_size= 3, strides= 2, filters= 64, padding= "same", activation= "relu", name= "capa_1" ))
model.add(MaxPool2D(pool_size=2, strides= 2))

#Convolucional Layer
model.add(Conv2D(kernel_size= 3, strides= 2, filters= 64, padding= "same", activation= "relu", name= "capa_2" ))
model.add(MaxPool2D(pool_size=2, strides= 2))

#Aplanamiento
model.add(Flatten())
model.add(Dense(100, activation="relu"))

#Capa de salida
model.add(Dense(num_class, activation="softmax"))

######COMPILACIÓN
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy',
tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tfa.metrics.F1Score(num_classes=2, average="micro"),
tfa.metrics.MultiLabelConfusionMatrix(num_classes=2)])


#CROSS-VALIDATION
numero_fold=1
accuracy_fold=[]
precision_fold=[]
recall_fold=[]
f1score_fold=[]
loss_fold=[]

kFold= KFold(n_splits=5,shuffle=True)

#JUNTAMOS LOS DATOS PARA QUE LA VALIDACIÓN CRUZADA LOS ORDENE
X = np.concatenate((imagenes, imagenesPrueba), axis=0)
y = np.concatenate((probabilidades, probabilidadesPrueba), axis=0)


# Tiempo de inicio de ejecución.
inicio = time.time()

for train, test in kFold.split( X, y):
    print("##################Training fold ",numero_fold,"###################################")
    model.fit(X[train], y[train],
            epochs=20,         #Epocas--> Cantidad de veces que debe repetir el entrenamiento
            batch_size=191      #Batch --> Cantidad de datos que puede cargar en memoria para realizar el entrenamiento en una fase
            )
    metricas=model.evaluate(X[test],y[test])
    f1score_fold.append(metricas[4])
    recall_fold.append(metricas[3])
    precision_fold.append(metricas[2])
    accuracy_fold.append(metricas[1])
    loss_fold.append(metricas[0])
    numero_fold+=1


# Tiempo de fin de ejecución.
fin = time.time()
# Tiempo de ejecución.
tiempo_total = fin - inicio
print(tiempo_total,"tiempo total")

for i in range(0,len(loss_fold)):
  print("Fold ",(i+1),"- Loss(Error)=",loss_fold[i]," - Accuracy=",accuracy_fold[i]," - Precision=",precision_fold[i]," - Recall=",recall_fold[i]," - F1 Score=",f1score_fold[i])
print("-------Average scores-------")
print("Loss",np.mean(loss_fold))
print("Accuracy",np.mean(accuracy_fold))
print("Precision",np.mean(precision_fold))
print("Recall",np.mean(recall_fold))
print("F1 Score",np.mean(f1score_fold))

#Guardar el modelo
ruta = "models/modelo_1v2.h5"
model.save(ruta)

# Resumen - Estructura de la red
model.summary()