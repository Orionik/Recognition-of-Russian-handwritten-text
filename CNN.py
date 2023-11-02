import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization # batch normalisation


y_train =  np.asarray(list(map(int,np.load("alp_y.npy"))))


x_train = np.load("all_x.npy")



y_test = np.asarray(list(map(int,np.load("test_y.npy"))))
x_test = np.load("test_x.npy")



y_train_cat = keras.utils.to_categorical(y_train,34)
y_test_cat = keras.utils.to_categorical(y_test,34)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

print( x_train.shape )

model = keras.Sequential ([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),        # фильтры, размер маски, выходная карта признаков имее тот же размер (28,28)
    MaxPooling2D((2, 2), strides=2),                                                      #размер окна, шаг сканирования
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(34,  activation='softmax')
])

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
print(model.summary())

his=model.fit(x_train, y_train_cat, batch_size=1024, epochs=1000, validation_split=0.33)

model.evaluate(x_test, y_test_cat)
model.save('T.h5')
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.show()