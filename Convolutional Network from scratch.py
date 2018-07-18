# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 00:30:12 2018
@author: HARSH

This is a code for 'Feed Forward Neural Network' using Numpy.
this network is used to classify images in two categories (CAT, NON CAT) without using any convolution layers.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def architecture(L,non):
    w = []
    b = []
    for i in range(L):
        w.append(np.random.rand(non[i+1],non[i]))
        b.append(np.random.rand(non[i+1]).reshape((non[i+1],1)))
    return w,b


def sigmoid(z):
    return 1/(1+np.exp(-z))


def ReLU(z):
    z[z<=0] = 0
    return z


def relu_derivative(z):
    z[z>0] = 1
    z[z<=0] = 0
    return z


def forward_propagate(X,w,b,L):
    a = []
    z = []
    a.append(X)
    z.append(X)
    for i in range(L):
        #print(a[-1].shape,w[i].shape,b[i].shape)
        z.append(np.dot(a[-1] , w[i].T) + b[i].T)
        if(i != L-1):
            a.append(ReLU(z[-1]))
        else:
            a.append(sigmoid(z[-1]))
    return a,z


def back_propagation(a,z,w,b,L,y):
    m = y.shape[0]
    da = []
    dw = []
    dz = []
    db = []
    #print(a)
    #cost = (-1/m)*np.sum(y*np.log(a[-1]) + (1-y)*np.log(1-a[-1]))
    cost = (1/m)*(np.sum(np.square(y-a[-1])))
    
    dz.append(a[-1] - y)
    dw.append(np.dot(dz[-1].T,a[L-1]))
    db.append(np.sum(dz[-1],axis = 0).T.reshape((dz[-1].shape[1],1)))
    
    for i in range(L-1):
        da.append(np.dot(dz[-1],w[L-1-i]))
        dz.append(da[-1]*relu_derivative(z[L-1-i]))
        dw.append(np.dot(dz[-1].T , a[L-2-i]))
        db.append(np.sum(dz[-1],axis = 0).T.reshape((dz[-1].shape[1],1)))
    
    grads = {"dw" : dw,
             "db" : db}
    return cost,grads


def update(w,b,grad,L,l_rate,m,lambd):
    
    dw = grad["dw"]
    db = grad["db"]
    
    for i in range(L):
        w[i] = w[i] - l_rate*((dw[L-1-i] + (lambd*w[i]))/m)
        b[i] = b[i] - l_rate*((db[L-1-i] + (lambd*b[i]))/m)
    
    return w,b


def predict(X,w,b,L):
    a,z = forward_propagate(X,w,b,L)
    pred = a[-1]
    pred[pred>0.49] = 1
    pred[pred != 1] = 0
    return pred


def gradient_desc(X,y,w,b,l_rate,n_iter,L,lambd):
    m = y.shape[0]
    costs = []
    grads = []
    for i in range(n_iter):
        a,z = forward_propagate(X,w,b,L)
        cost,grad = back_propagation(a,z,w,b,L,y)
        w,b = update(w,b,grad,L,l_rate,m,lambd)
        costs.append(cost)
        grads.append(grad["dw"][-1][-1])
        if(i%100 == 0):
            print(i,"th iteration cost: ",cost)
    return w,b,costs,grads,a

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(m_train,-1)
test_set_x_flatten = test_set_x_orig.reshape(m_test,-1)
X_train = train_set_x_flatten/255.
X_test = test_set_x_flatten/255.
y_train = train_set_y.reshape((m_train,1))
y_test = test_set_y.reshape((m_test,1))
 
#Optimized hyperparameters for single layer
#l_rate = [0.01]
#lambd = [0.3]
#L = 1
#n_iter = 5000
#non = [X_train.shape[1],1]

#l_rate = [0.001,0.003,0.01,0.03,0.1]
#lambd = [0.01,0.03,0.1,0.3,1,3,10,30]

l_rate = [0.03]
lambd = [1]
n_iter = 3000
L = 2
non = [X_train.shape[1],10,1]

for i in l_rate:
    for j in lambd:
        w,b = architecture(L,non)
        w,b,costs,grads,a = gradient_desc(X_train,y_train,w,b,i,n_iter,L,j)
        y_pred = predict(X_test,w,b,L)
        acc = 100 - np.mean(np.abs(y_pred - y_test))*100
        y_pred = predict(X_train,w,b,L)
        acc1 = 100 - np.mean(np.abs(y_pred - y_train))*100
        print("l_rate = ",i," lambd = ",j," accuracy = ",acc, " train acc = ",acc1)
plt.plot(costs)
plt.show()
#Images are shown using below command
#plt.imshow(test_set_x_orig[0])
