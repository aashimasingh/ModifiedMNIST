#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 20:27:20 2017

@author: asingh61
"""
import numpy as np
import pandas as pd
from sklearn import linear_model

#load the test data from CSV file
segdata = np.loadtxt("segmented_test_centered.csv", delimiter=",")
seglabel = np.loadtxt("train_y.csv", delimiter=",")

#load the train data from CSV file
train_x = np.loadtxt("master_data_final.csv", delimiter=",")
train_y = np.loadtxt("master_label_final.csv", delimiter=",")
h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e1)

pred_list = []
num_list =[]
count =0

#training the model
logreg.fit(train_x, train_y)

for i in range(0,len(segdata),3):
    
    numbers = np.asarray([0, 0 ,0])
    count = 0;
    #predict for each 3 segments
    pred1 = logreg.predict(segdata[i:i+3])
    if i==0:
        pred_prob = pred1;
    else:
        pred_prob= np.append(pred_prob,pred1,axis=0);
    for j in range(0,len(pred1)):
        if pred1[j]<10:
            numbers[count] = pred1[j]
            count = count+1
    if 10 in pred1:
        p = numbers[0]+numbers[1]
    elif 11 in pred1:
        p = numbers[0]*numbers[1]
    else:
        count = count+1
    pred_list.append(int(p))
    num_list.append(pred1)

pred_list = np.array(pred_list)
pred_list = pd.DataFrame(pred_list)
pred_list.index.name = 'Id'

#save the predicted data in csv file
pred_list.to_csv("logistic_regression.csv", sep=',', header=['Label'])