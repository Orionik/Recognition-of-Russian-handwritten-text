# -*- coding:utf-8 -*-
 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import imghdr
import numpy as np
import pathlib
from tensorflow import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM, BatchNormalization
from keras import backend as K
from keras.constraints import maxnorm
import tensorflow as tf
from scipy import io as spio
import idx2numpy 
from matplotlib import pyplot as plt
from typing import *
import time


image = cv2.imread('D.png', 0)
cv2.imshow("gray",image )

ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

contour, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

letters_con = []
for i in range(len(hierarchy[0])):
    if hierarchy[0][i][-1] == 0:
        letters_con.append(contour[i])

letters = []
for i in range(len(letters_con)):
    x_min=letters_con[i][0][0][0]
    x_max=letters_con[i][0][0][0]
    y_min=letters_con[i][0][0][1]
    y_max=letters_con[i][0][0][1]
    for j in range(len(letters_con[i])):
        if x_min > letters_con[i][j][0][0]:
            x_min = letters_con[i][j][0][0]
        if x_max < letters_con[i][j][0][0]:
            x_max = letters_con[i][j][0][0]
        if y_min > letters_con[i][j][0][1]:
            y_min = letters_con[i][j][0][1]
        if y_max < letters_con[i][j][0][1]:
            y_max = letters_con[i][j][0][1]
    letters.append([y_min,y_max,x_min,x_max])
        
letters.sort(key=lambda y:y[2])

txt =""
model = keras.models.load_model('CNN.h5')
alp = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
for i in range(len(letters)):
    crop_img = image[letters[i][0]-2:letters[i][1]+2, letters[i][2]-2:letters[i][3]+2]
    crop_img = cv2.resize(crop_img,(28,28))
    im = np.round(1-crop_img/255)
    x = np.expand_dims(im, axis=0)
    res = model.predict(x)
    txt += alp[np.argmax(res)-1]

print(txt)

