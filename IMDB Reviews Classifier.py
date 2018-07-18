# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 00:06:32 2018

@author: HARSH
"""

import keras
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from keras.datasets import imdb
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words = 10000)


def vectorize(X, dim = 10000):
    z = np.zeros((len(X), dim))
    for i,j in enumerate(X):
        z[i,j] = 1
    return z


def IMDB(input_shape):
    X_input = Input(input_shape)
    X = Dense(16, activation = 'relu', name = 'l1')(X_input)
    X = Dense(16, activation = 'relu', name = 'l2')(X)
    X = Dense(1, activation = 'sigmoid', name = 'l3')(X)
    model = Model(inputs = X_input, outputs = X)
    return model

X_train = vectorize(X_train)
X_train = X_train.astype(np.uint8)
X_test = vectorize(X_test)
X_test = X_test.astype(np.uint8)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.4)

model = IMDB(X_train.shape[1:])
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(x = X_train, y = Y_train, batch_size = 2, epochs = 20, validation_data = (X_val, Y_val))

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Clear figure 
plt.clf()

acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label = 'training accuracy')
plt.plot(epochs, val_acc, 'b', label = 'validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

eva = model.evaluate(x = X_test, y = Y_test, batch_size = 20)
pred = model.predict(x = X_test, verbose = 1)     

"""word_index = imdb.get_word_index()
reverse_word_index = dict(
[(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
[reverse_word_index.get(i - 3, '?') for i in train_data[0]])"""
