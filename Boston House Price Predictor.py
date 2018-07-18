# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:25:01 2018
@author: HARSH
"""

"""
This program is to predict House prices using dataset already available with Keras.
"""

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
(X_train, Y_train), (X_test, Y_test) = boston_housing.load_data()

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2)

model = models.Sequential()
model.add(layers.Dense(30, activation = 'relu', input_shape = (13,)))
model.add(layers.Dense(1, activation = 'relu'))

model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')


history = model.fit(x = X_train, y = Y_train, epochs = 400, validation_data = (X_val, Y_val))
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'b', label = 'training loss')
plt.plot(epochs, val_loss, 'r', label = 'validation loss')
plt.title('training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()

acc = history.history['mean_absolute_error']
val_acc = history.history['val_mean_absolute_error']
plt.plot(epochs, acc, 'b', label = 'training mean_absolute_error')
plt.plot(epochs, val_acc, 'r', label = 'validation mean_absolute_error')
plt.title('training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
