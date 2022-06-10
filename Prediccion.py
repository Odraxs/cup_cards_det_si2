import os
from tensorflow.python.keras.models import load_model
import tensorflow_addons as tfa
import numpy as np
import cv2

class Prediccion():
    def __init__(self,ruta,ancho,alto):
        self.ruta = ruta
        self.modelo=load_model(self.ruta,custom_objects={"Addons>F1Score":tfa.metrics.F1Score(num_classes=2, average="micro")})
        self.alto=alto
        self.ancho=ancho

    def predecir(self,imagen):
        imagen = cv2.resize(imagen, (self.ancho, self.alto))
        imagen = imagen.flatten()
        imagen = imagen / 255
        imagenesCargadas=[]
        imagenesCargadas.append(imagen)
        imagenesCargadasNPA=np.array(imagenesCargadas)
        predicciones=self.modelo.predict(x=imagenesCargadasNPA)
        print("Predicciones=",predicciones)
        clasesMayores=np.argmax(predicciones,axis=1)
        return clasesMayores[0]

    def cargarDatos(self, rutaOrigen, numeroCategorias, ancho, alto):
        imagenesCargadas = []
        valorEsperado = []
        for categoria in range(1, numeroCategorias):
            dir = rutaOrigen + str(categoria)
            files = os.listdir(dir)
            for file_name in files:
                ruta = dir + "/" + file_name
                imagen = cv2.imread(ruta)
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                imagen = cv2.resize(imagen, (ancho, alto), interpolation=cv2.INTER_AREA)
                imagen = imagen.flatten()
                imagen = imagen / 255
                imagenesCargadas.append(imagen)
                probabilidades = np.zeros(numeroCategorias)
                probabilidades[categoria] = 1
                valorEsperado.append(probabilidades)
        imagenesEntrenamiento = np.array(imagenesCargadas)
        valoresEsperados = np.array(valorEsperado)
        return imagenesEntrenamiento, valoresEsperados