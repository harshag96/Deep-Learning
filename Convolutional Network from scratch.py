# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 00:30:12 2018
@author: HARSH

This is a code for 'Feed Forward Neural Network' using Numpy.
this network is used to classify images in two categories (CAT, NON CAT) without using any convolution layers."""

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


def sigmoid(z):
    return 1/(1+np.exp(-z))


def ReLU(z):
    z[z<=0] = 0
    return z


def relu_derivative(z):
    z[z>0] = 1
    z[z<=0] = 0
    return z


def softmax(z):
    z = np.exp(z)
    t = np.sum(z)
    return z/t


def pad (X,p):
    new_X = np.zeros((X.shape[0]+(2*p),X.shape[1]+(2*p),X.shape[2]))
    new_X[p:X.shape[0]+p,p:X.shape[1]+p,:] = X
    return new_X


def architecture(layers,f,s,p,nof,shap):
    w = []
    b = []
    for i,layer in enumerate(layers):
        if(layer == 'conv'):
            w.append(np.random.randn(nof[i],f[i],f[i],shap[2]))
            b.append(np.nan)
            shap[0] = int(np.floor((shap[0]+ 2*p[i] - f[i])/s[i]) + 1)
            shap[1] = int(np.floor((shap[1]+ 2*p[i] - f[i])/s[i]) + 1)
            shap[2] = nof[i]
        elif(layer == 'fullyc'):
            shap[0] = shap[0]*shap[1]*shap[2]
            shap[1] = 1
            shap[2] = 1
            w.append(np.random.randn(f[i],int(shap[0])))
            b.append(np.random.randn(f[i],1))
            shap[0] = f[i]
        else:
            shap[0] = int(np.floor((shap[0]+ 2*p[i] - f[i])/s[i]) + 1)
            shap[1] = int(np.floor((shap[1]+ 2*p[i] - f[i])/s[i]) + 1)
            b.append(np.nan)
            w.append(np.nan)
    params = {"w" : w,
              "b" : b}
    return params
            


def conv(X,F,p,s):
    if(p!=0):
        X = pad(X,p)
    
    itr = int(np.floor((X.shape[0]-F.shape[0])/s)+1)
    jtr = int(np.floor((X.shape[1]-F.shape[1])/s)+1)
    res = np.random.rand(itr,jtr)
    for i in range(itr):
        for j in range(jtr):
            spt1 = i*s
            spt2 = j*s
            res[i,j] = np.sum(X[spt1:spt1+F.shape[0],spt2:spt2+F.shape[1],:]*F)
    
    return res


def maxpool(X,f,p,s):
    if(p!=0):
        X = pad(X,p)
    
    itr = int(np.floor((X.shape[0]-f/s)+1))
    jtr = int(np.floor((X.shape[1]-f)/s)+1)
    res = np.random.rand(itr,jtr,X.shape[2])
    for i in range(itr):
        for j in range(jtr):
            spt1 = i*s
            spt2 = j*s
            res[i,j,:] = np.max(np.max(X[spt1:spt1+f,spt2:spt2+f,:],axis=0),axis=0)
    
    return res


def fullyc(X,w,b,act):
    z = np.dot(w,X) + b
    if(act == "sigmoid"):
        z = sigmoid(z)
    elif(act == "softmax"):
        z = softmax(z)
    else:
        z = ReLU(z)
    return z


def propagate(X,layers,f,s,p,params):
    w = params["w"]
    b = params["b"]
    a = [X]
    shap = list(X.shape)
    
    for i,layer in enumerate(layers):
        if(layer == 'conv'):
            shap[0] = int(np.floor((shap[0]+ 2*p[i] - f[i])/s[i]) + 1)
            shap[1] = int(np.floor((shap[1]+ 2*p[i] - f[i])/s[i]) + 1)
            shap[2] = nof[i]
            res = np.random.rand(shap[0],shap[1],shap[2])
            for j in range(nof[i]):
                res[:,:,j] = conv(X,w[i][j],p[i],s[i])
            a.append(res)
        
        elif(layer == 'fullyc'):
            X = X.reshape((-1,1))
            a.append(fullyc(X,w[i],b[i],'ReLU'))
            shap = list(a[-1].shape)
        else:
            shap[0] = int(np.floor((shap[0]+ 2*p[i] - f[i])/s[i]) + 1)
            shap[1] = int(np.floor((shap[1]+ 2*p[i] - f[i])/s[i]) + 1)
            a.append(maxpool(X,f[i],p[i],s[i]))
        X = a[-1]
    return a
    
    
def grad_descent(X_train,y_train,params,layers,f,s,p,nof,l_rate,lambd,n_iter):
    costs = []
    for i in range(n_iter):
        for j in range(10,11):#X_train.shape[0]
            X = X_train[j]
            X = X/np.linalg.norm(X)
            a = propagate(X,layers,f,s,p,params)
    return a,costs        
            
    
    
    
    
#Main code

# Loading the data (cat/non-cat)
X_train, y_train, X_test, y_test, classes = load_dataset()

#Shape of input image
shap = list(X_train[0].shape)
#Defining architecture
layers = ['conv','maxpool','fullyc','fullyc']
#Filter size
f = [3,3,100,1]
#Stride
s = [1,1,1,1]
#Padding
p = [0,0,0,0]
#Number of filters
nof = [6,0,0,0]
#Initializing parameters
params = architecture(layers,f,s,p,nof,shap)

#Hyperparameters
n_iter = 1
l_rate = 0.01
lambd = 0.0
a,cost = grad_descent(X_train,y_train,params,layers,f,s,p,nof,l_rate,lambd,n_iter)
