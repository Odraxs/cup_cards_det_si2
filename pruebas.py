import os
import time

import cv2
from Parcial.Prediccion import Prediccion

class suma():
    acumulado = 0

def predecir():
    clases = [ 1,  2,  3,  4,  5,  6,  7,  8,  9]
    ancho = 128
    alto = 128
    dirc = "E:/david/U PC/2022-1/Inteligentes/Parte 2/MachineLEarning/Parcial/Crops/cartas_dataset/"
    #dirc = "E:/david/U PC/2022-1/Inteligentes/Parte 2/MachineLEarning/Parcial/"
    dirm = "E:/david/U PC/2022-1/Inteligentes/Parte 2/MachineLEarning/Parcial/Modelos/models/"
    miModeloCNN = Prediccion(dirm + "modelo_2.h5", ancho, alto)
    files = os.listdir(dirc + "nuevas/")
    for file in files:
        imagen = cv2.imread(dirc + "nuevas/" + file)
        imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        # imagen=cv2.imread(dirc+direccion)
        inicio = time.time()
        claseResultado = miModeloCNN.predecir(imagenGris)
        fin = time.time()
        print(fin-inicio,"tiempo respuesta")
        #print("La imagen cargada es ", clases[claseResultado - 1])
        suma.acumulado += clases[claseResultado - 1]
        msg1="prediccion "+str(clases[claseResultado - 1])
        dar_texto_resaltar_img(imagen, str(suma.acumulado) + " copas")
        cv2.putText(imagen, msg1, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("resultado"+file, imagen)


    while True:
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()

def dar_texto_resaltar_img(imagen, mensaje):
    cv2.putText(imagen, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #cv2.drawContours(imagen, [figura_actual], 0, (0, 0, 255), 2)

#predecir()

