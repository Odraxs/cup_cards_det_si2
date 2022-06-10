from recorte import Recorta
import numpy as np
import cv2
import os
import glob
from Parcial.Prediccion import Prediccion


def calcularAreas(figura):
    areas = []
    for figuraActual in figura:
        areas.append(cv2.contourArea(figuraActual))
    return areas


####    CARGA LAS IMAGENES QUE SE ESTÁN EN LA CARPETA "imagenesPrueba"
def cargarDatos(numeroCategorias):
    imagenesCargadas = []
    for categoria in range(1, numeroCategorias):
        ruta = "imagenesPrueba/" + str(categoria) + "_1.jpg"
        imagen = cv2.imread(ruta)
        imagenesCargadas.append(imagen)
    return imagenesCargadas


###     ELIMINA LAS IMAGENES QUE SE HAN TOMADO
def eliminarImagenes():
    py_files = glob.glob('imagenesPrueba/*.jpg')

    for py_file in py_files:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{e.strerror}")


def detectarPoligono(imagen):
    global num, suma, acumulado
    global flag
    # PREPROCESADO DE LA IMÁGEN
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(imagenGris, 59, 86)
    kernel = np.ones((2, 2), np.uint8)
    bordes = cv2.dilate(bordes, kernel)

    ######### DETECCIÓN DE IMÁGEN ################
    # RETR_EXTERNAL solo para contornos padres
    figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = calcularAreas(figuras)

    i = 0
    sendFiguras = []
    for figuraActual in figuras:
        if areas[i] > 1000:  # Elimina los contornos insignificantes
            cv2.drawContours(imagen, [figuraActual], 0, (251, 247, 0), 2)
            cv2.putText(imagen, "Carta", np.array(figuraActual[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (251, 247, 0), 2,
                        cv2.LINE_AA)
            sendFiguras.append(figuraActual)
        i = i + 1

    ############# SUMAR LOS VALORES
    if flag:
        suma = 0
        # Cuenta la cantidad de archivos en la carpeta
        _, _, files = next(os.walk("imagenesPrueba/"))
        file_count = len(files) + 1
        # Carga la cantidad de archivos en la carpeta
        imagenes = cargarDatos(file_count)

        # Predice el valor de cada imagen
        for imgActual in imagenes:
            imgActual = cv2.cvtColor(imgActual, cv2.COLOR_BGR2GRAY)
            suma = suma + modeloCNN.predecir(imgActual)  # Retorna el valor de la carta
        acumulado = acumulado + suma

        # Elimina las imágenes para volver a sumarlas
        eliminarImagenes()
        num = 1
        flag = False

    msg1 = "Suma = " + str(suma)
    msg2 = "Acumulado = " + str(acumulado)
    cv2.putText(imagen, msg1, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (243, 239, 93), 2, cv2.LINE_AA)
    cv2.putText(imagen, msg2, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (243, 239, 93), 2, cv2.LINE_AA)

    return [bordes, sendFiguras]


# Numero de fotos
num = 1  # Número que lleva el conteo de las fotos tomadas
suma = 0
acumulado = 0  # Lleva el acumulado de la suma de las cartas
flag = False  # Bandera que se habilita para realizar las operaciones
video = cv2.VideoCapture(1)  # Abrir camara
dir_root="E:/david/U PC/2022-1/Inteligentes/Parte 2/MachineLEarning/Parcial/Modelos/models/"
modeloCNN = Prediccion(dir_root+"modelo_1v2.h5", 128, 128) #Cargar el modelo

while True:
    _, imagen = video.read()
    imgBorder, shapes = detectarPoligono(imagen)
    small_img = Recorta()

    cv2.imshow("Imagen", imagen)
    # Cerrar la ventana
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    ## Captura las imagenes para procesarlas ##
    if k == ord('p'):
        num = small_img.recortar('imagenesPrueba/', imgBorder, shapes, 1, num)
        num = num + 1
        flag = True


