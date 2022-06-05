import cv2

def calcular_areas(figuras):
    areas = []
    for figuraActual in figuras:
        areas.append(cv2.contourArea(figuraActual))
    return areas


dirc = "E:/david/U PC/2022-1/Inteligentes/Parte 2/MachineLEarning/Parcial/Crops/"
imagen = cv2.imread(dirc + "dos cartas.jpg")

# Convertimos a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

cv2.imshow("gris",gris)
# Aplicar suavizado Gaussiano
gauss = cv2.GaussianBlur(gris, (5, 5), 0)
cv2.imshow("gauss",gauss)
# Detectamos los bordes con Canny
canny = cv2.Canny(gauss, 59, 90)
cv2.imshow("canny",canny)
figuras, jerarquia = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

areas = calcular_areas(figuras)
i = 0
for figura_actual in figuras:
    if areas[i] >= 300:
        i = i + 1
        vertices = cv2.approxPolyDP(figura_actual, 0.05 * cv2.arcLength(figura_actual, True), True)
        if len(vertices) == 4:
            mensaje = "Cuadrado"
            # dar_texto_resaltar_img(imagen, mensaje, figura_actual)

            cv2.drawContours(imagen, [figura_actual], 0, (0, 0, 255), 2)
            cv2.imshow("contornos", imagen)
            # if cv2.waitKey(1) & 0xFF == ord('c'):
            #     cv2.imshow("carta", imagen)

cv2.waitKey(0)
