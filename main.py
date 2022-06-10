import cv2
import numpy as np
import base64
import os
import requests
import json
from Parcial.Cut import Cut



def nothing(x):
    pass


def constructorVentana(nameWindow):
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min", nameWindow, 59, 255, nothing)
    cv2.createTrackbar("max", nameWindow, 86, 255, nothing)
    cv2.createTrackbar("kernel", nameWindow, 2, 100, nothing)
    cv2.createTrackbar("areaMin", nameWindow, 350, 2000, nothing)
    cv2.createTrackbar("areaMax", nameWindow, 1300, 5000, nothing)


"""calcular area de la figura"""


def calcularAreas(figuras):
    areas = []
    for figuraActual in figuras:
        areas.append(cv2.contourArea(figuraActual))
    return areas

def dar_texto_resaltar_img(imagen, mensaje, figura_actual):
    cv2.putText(imagen, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawContours(imagen, [figura_actual], 0, (0, 0, 255), 2)

"""detectar forma dentro del rectangulo """

def detectarFormas(imagen, idImg, recorte, nameWindow):
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    min = cv2.getTrackbarPos("min", nameWindow)
    max = cv2.getTrackbarPos("max", nameWindow)
    bordes = cv2.Canny(imagen_gris, min, max)
    tamaño_kernel = cv2.getTrackbarPos("kernel", nameWindow)
    kernel = np.ones((tamaño_kernel, tamaño_kernel), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    cv2.imshow("Bordes", bordes)

    # Deteccion de la figura
    figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = calcularAreas(figuras)
    area_min = cv2.getTrackbarPos("areaMin", nameWindow)
    i = 0
    for figura_actual in figuras:
        if areas[i] >= area_min:
            vertices = cv2.approxPolyDP(figura_actual, 0.05 * cv2.arcLength(figura_actual, True), True)
            if len(vertices) == 4:
                #mensaje = "Cuadrado"
                #dar_texto_resaltar_img(imagen, mensaje, figura_actual)
                #idImg, imagen = recorte.crop(carta, cnts, idImg, bordes)
                idImg, imagen = recorte.crop2(imagen, figuras, idImg, bordes,imagen_gris)
                i += 1
    return imagen, idImg

def detectarFormas2(imagen, idImg, recorte, nameWindow):
    carta = imagen
    imagengris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    min = cv2.getTrackbarPos("min", nameWindow)
    max = cv2.getTrackbarPos("max", nameWindow)
    tamañoKernel = cv2.getTrackbarPos("kernel", nameWindow)
    areamin = cv2.getTrackbarPos("areaMin", nameWindow)
    areamax = cv2.getTrackbarPos("areaMax", nameWindow)
    bordes = cv2.Canny(imagengris, min, max)
    kernel = np.ones((tamañoKernel, tamañoKernel), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    # cv2.imshow('Imagen', bordes)
    #contornosR, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts, _ = cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('Imagen2', bordes)
    areas = calcularAreas(cnts)
    i = 0
    for figuraActual in cnts:
        area = cv2.contourArea(figuraActual)
        if area >= areamin and area <= areamax:
            # cv2.drawContours(carta, [figuraActual], 0, (0, 0, 255), 2)
            # cv2.imshow('bordes', carta)
            epsilon = 0.01 * cv2.arcLength(figuraActual, True)
            approx = cv2.approxPolyDP(figuraActual, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            # coordenadas de los vertices
            vertices = cv2.approxPolyDP(figuraActual, epsilon, True)
            vertices = cv2.approxPolyDP(figuraActual, 0.05 * cv2.arcLength(figuraActual, True), True)
            if len(vertices) == 4:
                aspect_ratio = float(w) / h
                if aspect_ratio == 1:
                    #cv2.putText(imagen, 'Cuadrado', (x, y - 5), 1, 1.5, (0, 255, 0), 2)
                    print("cuadro")
                else:
                    # print("recorte")
                    # cv2.putText(imagen, 'rect', (x, y - 5), 1, 1.5, (0, 255, 0), 2)
                    idImg, imagen = recorte.crop2(carta, cnts, idImg, bordes)
                    #Para tomar gris y bordes al tiempo
                    #idImg, imagen = recorte.crop2(carta, cnts, idImg, imagengris)
        i += 1
    return carta, idImg


def cnvrtBase64(rutaImagen):
    img = cv2.imread(rutaImagen)
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)
    encoded_string = jpg_as_text.decode('utf-8')
    return encoded_string


"""Función para devolver los nombres de las imagenes obtenidas"""


def listar_cartas(ruta):
    dir = ruta
    imagenes = []
    with os.scandir(dir) as ficheros:
        for fichero in ficheros:
            imagenes.append(fichero.name)
    return imagenes


def gui(nameWindow):
    camara = cv2.VideoCapture(1)
    recorte = Cut
    k = 0
    # Identificación del cliente
    cantModelos = input('Ingrese la cantidad de modelos que desea usar: ')
    modelos = []
    if cantModelos == "1" or cantModelos == "2" or cantModelos == "3":
        for i in range(int(cantModelos)):
            modelos.append(input('Ingrese el número de modelo "Para el modelo A 1, modelo B 2, modelo C 3": '))
            if modelos[i] != "1" and modelos[i] != "2" and modelos[i] != "3":
                print('No ingresó un modelo válido')
                exit()
    else:
        print('No ingresó una cantidad de modelos válido')
        exit()
    constructorVentana(nameWindow)
    idImg = 351
    while True:
        k = cv2.waitKey(1)
        _, imagen = camara.read()
        ##TODO:
        imagen, idImg = detectarFormas(imagen, idImg, recorte,nameWindow)
        cv2.imshow('Imagen', imagen)
        if k == ord('e'):
            break
    return modelos


"""Inicio el envío de las imagenes al servidor 'Crops/carta_n.jpg'"""


def service(idcliente,modelos):
    imgBase64 = []
    cartas = listar_cartas('Crops')
    # print(cartas)
    idsImg = 0
    for imgcarta in cartas:
        imageBase64 = cnvrtBase64('Crops/' + imgcarta)
        imgBase64.append({"id": idsImg, "content": imageBase64})
        idsImg += 1
    data = {'id_client': idcliente, 'images': imgBase64, 'models': modelos}
    headers = {'Content-Type': 'application/json'}
    resp = requests.post('http://localhost:5000/predict', data=json.dumps(data), headers=headers)
    respuesta = json.loads(resp.text)
    print(respuesta)

def main():
    nameWindow = "Calculadora Canny"
    #idcliente = input('Ingrese un identificador: ')
    modelos=gui(nameWindow)
    ##service(idcliente,modelos)


if __name__ == "__main__":
    main()
