import cv2
from Parcial.Prediccion import Prediccion

class suma():
    acumulado = 0

def predecir(direccion,color,contorno):
    clases = [ 1,  2,  3,  4,  5,  6,  7,  8,  9]
    ancho = 128
    alto = 128
    #dirc = "E:/david/U PC/2022-1/Inteligentes/Parte 2/MachineLEarning/Parcial/Crops/cartas_dataset/"
    dirc = "E:/david/U PC/2022-1/Inteligentes/Parte 2/MachineLEarning/Parcial/"
    dirm = "E:/david/U PC/2022-1/Inteligentes/Parte 2/MachineLEarning/Parcial/Modelos/models/"
    miModeloCNN = Prediccion(dirm + "modelo_1v2.h5", ancho, alto)
    #imagen = cv2.imread(dirc + "test/9/9_160.jpg")
    imagen=cv2.imread(dirc+direccion)
    claseResultado = miModeloCNN.predecir(imagen)
    print("La imagen cargada es ", clases[claseResultado-1])
    suma.acumulado+= clases[claseResultado-1]
    dar_texto_resaltar_img(color, str(suma.acumulado) + " copas", contorno)
    cv2.imshow("resultado",color)
    # while True:
    #     cv2.imshow("imagen", imagen)
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         break
    #cv2.destroyAllWindows()

def dar_texto_resaltar_img(imagen, mensaje, figura_actual):
    cv2.putText(imagen, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #cv2.drawContours(imagen, [figura_actual], 0, (0, 0, 255), 2)

#predecir("a")

