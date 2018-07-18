# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:59:47 2018
@author: HARSH

Predicts 1 for DOG and 0 for CAT
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Dropout, Flatten, Activation, BatchNormalization
from keras.models import Model
#from sklearn.model_selection import train_test_split

path1 = r'cat dog dataset\\PetImages\\cat\\'
path2 = r'cat dog dataset\\PetImages\\dog\\'
samples = 1000
dim=64

X = np.zeros((2*samples, dim, dim, 3))
Y = np.zeros((2*samples, 1))
j = 0
for i in range(samples):
    filename = path1+repr(i+1)+'.jpg'
    img = cv2.imread(filename)
    if(img is not None):
        X[j] = cv2.resize(img, (dim,dim))
        j+=1
    filename = path2+repr(i+1)+'.jpg'
    img = cv2.imread(filename)
    if(img is not None):
        X[j] = cv2.resize(img, (dim,dim))
        Y[j] = 1
        j+=1

#X = np.array(X)
#Y = np.array(Y)
X = X/255
#separating 80 and 20 percentage
brk = int((2*samples)*0.8)
X_test = X[brk:,:,:,:]
Y_test = Y[brk:,:]
X = X[:brk,:,:,:]
Y = Y[:brk,:]
#plt.imshow(X)
        
def modell(input_shape):
    X_input = Input(input_shape)
    X = Conv2D(32, (3,3))(X_input)
    #X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)
    X = Dropout(0.1)(X)
    
    X = Conv2D(32, (3,3))(X)
    #X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)
    X = Dropout(0.15)(X)
    
    X = Flatten()(X)
    X = Dense(256, activation = 'relu')(X)
    X = Dropout(0.1)(X)
    X = Dense(128, activation = 'relu')(X)
    #X = Dropout(0.05)(X)
    X_out = Dense(1, activation = 'sigmoid')(X)
    model = Model(inputs = X_input, outputs =  X_out, name = 'classifier')
    return model

model = modell((dim, dim, 3))
model.compile('rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(X, Y, epochs = 10, batch_size = 32, validation_split = 0.1)
h = history.history
epochs = range(1,len(h['loss'])+1)
plt.plot(epochs, h['acc'], 'b', label = 'training accuracy')
plt.plot(epochs, h['val_acc'], 'r', label = 'validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('accuracy')
plt.legend()
plt.show()
