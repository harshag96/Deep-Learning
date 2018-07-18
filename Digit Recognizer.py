# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 18:35:13 2018

@author: HARSH
"""
import keras
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Input, Dropout
import numpy as np
import h5py
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
np.random.seed(1)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train  = X_train/255
X_test = X_test/255
#One hot encoding
enc = OneHotEncoder()
Y_train = enc.fit_transform(Y_train.reshape((-1,1)))
Y_test = enc.transform(Y_test.reshape((-1,1)))

def digit_recognizer(input_shape):
    X_input = Input(input_shape)
    
    X = Flatten()(X_input)
    X = Dense(512, activation = 'relu', name = 'l1')(X)
    X = Dropout(0.25)(X)
    #X = Activation('relu')(X)
    
    X = Dense(84, activation = 'relu', name = 'l2')(X)
    X = Dropout(0.25)(X)
    #X = Activation('reu')(X)
    
    X = Dense(10, activation = 'softmax', name = 'l3')(X)
    
    model = Model(inputs = X_input, outputs = X, name = 'digit_recognizer')

    return model

model = digit_recognizer(X_train.shape[1:])
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x = X_train, y = Y_train, epochs = 5, batch_size = 128, validation_split = 0.2)
eva = model.evaluate(X_test, Y_test)
pred = model.predict(x = X_test, verbose = 1)
