from tensorflow.python.keras.models import load_model
import tensorflow_addons as tfa
from sklearn.metrics import confusion_matrix
import numpy as np
from Parcial.Prediccion import Prediccion
## Matrices de confusión
width = 128
height = 128
num_class = 10
dirc="E:/david/U PC/2022-1/Inteligentes/Parte 2/MachineLEarning/Parcial/Crops/cartas_dataset/"
dir_root="E:/david/U PC/2022-1/Inteligentes/Parte 2/MachineLEarning/Parcial/Modelos/models/"
###  CAMBIAR RUTAS PARA MOSTRAR LOS OTROS MODELOS    ###
miModeloCNN = Prediccion(dir_root+"modelo_3.h5", width, height)
imagenesPrueba,probabilidadesPrueba = miModeloCNN.cargarDatos( dirc+"test/", num_class, width, height)

model= load_model(dir_root+"/modelo_3.h5",custom_objects={"Addons>F1Score":tfa.metrics.F1Score(num_classes=2, average="micro")})
YPred= model.predict(imagenesPrueba)
yPred= np.argmax(YPred, axis=1)
MatrixConf= confusion_matrix( np.argmax(probabilidadesPrueba,axis=1),yPred)
print(MatrixConf)
