#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 21:16:38 2017

@author: asingh61
"""

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import pandas as pd
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D ,AveragePooling2D
from keras import regularizers
from keras.utils import np_utils
from keras import optimizers
from keras import initializers
from keras import callbacks
from keras.layers.normalization import BatchNormalization
      
        
print("Running........")
train_x = np.loadtxt("mini_master_data_final.csv", delimiter=",") # load from text 
print("Training Data loaded!!!")
train_y = np.loadtxt("mini_master_label_final.csv", delimiter=",") 



le = preprocessing.LabelEncoder()
train_y = le.fit_transform(train_y)
np.savetxt("train_y_new.csv", train_y,delimiter=",")

tr_x, valid_x, tr_y, valid_y = train_test_split(train_x, train_y, test_size=0.2)
print(tr_x.shape)
train_x  = train_x.reshape(train_x.shape[0],28, 28,1,order ='C')
tr_x = tr_x.reshape(tr_x.shape[0],28, 28,1,order ='C')

valid_x = valid_x.reshape(valid_x.shape[0],28, 28,1,order = 'C')
print("Loading Segmented Test....")
test_x2 = np.loadtxt("segmented_test_centered.csv",delimiter=",")
test_y = np.loadtxt("train_y.csv", delimiter =",")
test_x = test_x.reshape(test_x.shape[0],28, 28,1,order = 'C')
test_x2 = test_x2.reshape(test_x2.shape[0],28,28,1,order = 'C')

tr_x = tr_x.astype('float32')
valid_x = valid_x.astype('float32')
test_x = test_x.astype('float32')
test_x2 = test_x2.astype('float32')
tr_x = (tr_x - np.mean(tr_x))/np.std(tr_x)
test_x = (test_x - np.mean(test_x))/np.std(test_x)
valid_x = (valid_x - np.mean(valid_x))/np.std(valid_x)
print(np.min(tr_x),np.max(tr_x))
print(np.mean(tr_x),np.std(tr_x))
print(np.min(test_x),np.max(test_x))
print(np.mean(test_x),np.std(test_x))
#tr_x /= 255
#valid_x/=255
#test_x /= 255
Y_train = np_utils.to_categorical(tr_y, 12)
Y_test = np_utils.to_categorical(valid_y, 12)
  
batch_size = 128
num_classes = 12
epochs = 3

# input image dimensions
img_rows, img_cols = 28, 28


model = Sequential()
model.add(Convolution2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

adam = optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0007) 
model.compile(loss='categorical_crossentropy',
              optimizer= adam,
              metrics=['accuracy'])

print("The training starts....GL")
model.fit(tr_x, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(valid_x, Y_test))

b_count2 = 0
pred_list2 = []
num_list2 = []
pred_prob2 = np.array([])
for i in range(0,len(test_x2),3):
    
    numbers = np.asarray([0, 0 ,0])
    count = 0;
    pred2 = model.predict(test_x2[i:i+3])
    #print(pred1.size)
    if i==0:
        pred_prob2 = pred2;
    else:
        pred_prob2= np.append(pred_prob2,pred2,axis=0);
    pred2 = np.argmax(pred2,axis=1)
    for j in range(0,len(pred1)):
        if pred2[j]<10:
            numbers[count] = pred2[j]
            count = count+1
    if 10 in pred2:
        p = numbers[0]+numbers[1]
    elif 11 in pred2:
        p = numbers[0]*numbers[1]
    else:
        b_count2 = b_count2+1
    pred_list2.append(p)
    num_list2.append(pred2)
    
dfpred = pd.DataFrame(pred_list2)
dfpred.index.name = 'Id'
dfpred.to_csv('predictions.csv',header = ['Label'] ) 

