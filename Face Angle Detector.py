# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 17:43:47 2018

@author: HARSH
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import keras
from PIL import Image
from scipy import ndimage
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
#import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras import optimizers
#from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.applications.vgg16 import VGG16

#%matplotlib inline

def load_dataset():
    dataset = h5py.File('fei_data.h5', "r")
    X = np.array(dataset["X"][:])
    Y = np.array(dataset["Y"][:])
    
    Y[Y == 1] = -90
    Y[Y == 2] = -60
    Y[Y == 3] = -45
    Y[Y == 4] = -30
    Y[Y == 5] = -15
    Y[Y == 6] = 15
    Y[Y == 7] = 30
    Y[Y == 8] = 45
    Y[Y == 9] = 60
    Y[Y == 10] = 90
    
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)
    
    return X_train, X_test, Y_train, Y_test

def HappyModel(input_shape):

    X_input = Input(input_shape)
    X = Dropout(rate = 0.1)(X_input)
    
    #X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(6, (5,5), strides = (1,1), name = 'conv0')(X_input)
    #X = BatchNormalization(axis = 3, name = 'bn0')(X)
    #X = Activation('relu')(X)    
    X = MaxPooling2D(pool_size = (2,2), strides = 2, name = 'max_pool0')(X)
    X = Dropout(rate = 0.1)(X)
    
    X = Conv2D(16, (5,5), strides = (1,1), name = 'conv1')(X)
    #X = BatchNormalization(axis = 3, name = 'bn1')(X)
    #X = Activation('relu')(X)
    X = MaxPooling2D(pool_size = (2,2), strides = 2, name = 'max_pool1')(X)
    X = Dropout(rate = 0.1)(X)
    
    X = Flatten()(X)
    
    X = Dense(200, activation = None, name = 'fc0')(X)
    X = Dropout(rate = 0.1)(X)
    
    X = Dense(84, activation = None, name = 'fc1')(X)
    X = Dropout(rate = 0.1)(X)
    
    X = Dense(1, activation = None, name = 'fc2')(X)
    
    HappyModel = Model(inputs = X_input, outputs = X, name = 'happy_house')
    
    return HappyModel

X_train, X_test, Y_train, Y_test = load_dataset()
# Normalize image vectors
X_train = X_train/255.
X_test = X_test/255.


happyModel = HappyModel(X_train.shape[1:])

sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
happyModel.compile('Adam', 'mean_squared_error')

#happyModel.summary()

happyModel.fit(x = X_train, y = Y_train, epochs = 30, batch_size = 50)

preds = happyModel.evaluate(x = X_train, y = Y_train, batch_size=32, verbose=1)
print ("Loss = " + str(preds))
#print ("Train Accuracy = " + str(preds[1]))

preds = happyModel.evaluate(x = X_test, y = Y_test, batch_size=32, verbose=1)
print ("Loss = " + str(preds))
#print ("Test Accuracy = " + str(preds[1]))

predictions = happyModel.predict(X_test, verbose = 1)

"""
from keras.models import load_model
#happyModel.save('KerasModel.h5')
#del happyModel
happyModel = keras.models.load_model('KerasModel.h5')
"""

"""
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
"""
