import numpy as np
import os
from sklearn.model_selection import train_test_split
import shutil

dir ="E:/david/U PC/2022-1/Inteligentes/Parte 2/MachineLEarning/Parcial/Crops/9/"

# The list of items
files = os.listdir(dir)

xTrain, xTest = train_test_split(files ,test_size=0.20,random_state=42)

# Loop to print each filename separately
for filename in xTrain:
    shutil.move(dir+filename,dir+"train/"+filename)
    #print(filename)

for filename in xTest:
    shutil.move(dir + filename, dir + "test/" + filename)
    # print(filename)

