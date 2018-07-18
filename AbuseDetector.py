# -*- coding: utf-8 -*-
"""
Fight online abuse

Can you confidently and accurately tell via a particular is abusive?
Dataset: Toxic Comments on Kaggle (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation
from keras.preprocessing.text import Tokenizer

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

trainX = train.comment_text
Y_train = train.iloc[:,2:]

testX = test.comment_text

tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(trainX)
X_train = tokenizer.texts_to_matrix(trainX)
X_test = tokenizer.texts_to_matrix(testX)
wi = tokenizer.word_index

#defining model
def AbuseDetector(input_shape):
    X_input = Input(input_shape)
    
    X = Dense(512, activation = 'relu')(X_input)
    X = Dense(6, activation = 'sigmoid')(X)
    model = Model(X_input, X)
    return model

model = AbuseDetector(X_train.shape[1:])
model.compile('rmsprop', loss = 'binary_crossentropy', metrics= ['accuracy'])
history = model.fit(x = X_train, y = Y_train, epochs = 1, batch_size = 64, validation_split = 0.2)
Y_test = model.predict(x = X_test)
co = ['id','toxic', 'severe_toxic', 'obscene', 'threat', 'insult','identity_hate']
df = pd.DataFrame(test.id)
to_write = pd.concat([df,pd.DataFrame(Y_test)],axis = 1)
to_write.columns = co
to_write.to_csv('choot.csv',sep = ',',index = False)
