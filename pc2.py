# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:08:40 2019

@author: Ashan M Silva
"""

import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
import random
import pickle
from sklearn.model_selection import train_test_split



datadir = "F:/MACHINE LEARNING/Machine learning/kvasir-dataset-v2" #path to images folder
categories = ['dyed-lifted-polyps','dyed-resection-margins','esophagitis','normal-cecum','normal-pylorus','normal-z-line','polyps','ulcerative-colitis']

def create_training_data_set(img_size):
    training_data_set =[] #create list of training dataset
    for category in categories:
        #class_index = category
        class_index = categories.index(category)  #get category as the index to identify the category
        path = os.path.join(datadir,category)  #path to categories
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img)) #image in gray color
                img_array_rgb = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
                resized_img_array = cv2.resize(img_array_rgb,(img_size,img_size))
                training_data_set.append([resized_img_array,class_index]) #append data to training data set
            except Exception as e:
                pass
    random.shuffle(training_data_set) #shuffle the data
    return training_data_set

img_size = 60 #image size
training_data_set = create_training_data_set(img_size)
def create_x_y_sets(training_data_set,img_size):
    
    x = []
    y = []
    for features,label in training_data_set:
        x.append(features)
        y.append(label)
    x = np.array(x).reshape(8000,img_size,img_size,3)
   # y = np.array(y)
   
       
    return x,y



x,y = create_x_y_sets(training_data_set,img_size)

z = np.zeros((8000, 8), dtype = int)
for i in range(8000):
    n=y[i]
    z[i][n]=1;


    
X_train, X_test, y_train, y_test = train_test_split(x, z, test_size=0.2)

def save_data_x_train(x):
    pickle_out = open("x_train.pickle","wb")
    pickle.dump(x,pickle_out)
    pickle_out.close()
    
def save_data_y_train(y):    
    pickle_out = open("y_train.pickle","wb")
    pickle.dump(y,pickle_out)
    pickle_out.close()
    
def get_data_x_train(location):
    pickle_in = open(location, "rb")
    x = pickle.load(pickle_in)
    return x

def get_data_y_train(location):
    pickle_in = open(location, "rb")
    y = pickle.load(pickle_in)
    return y

def save_data_x_test(x):
    pickle_out = open("x_test.pickle","wb")
    pickle.dump(x,pickle_out)
    pickle_out.close()
    
def save_data_y_test(y):    
    pickle_out = open("y_test.pickle","wb")
    pickle.dump(y,pickle_out)
    pickle_out.close()
    
def get_data_x_test(location):
    pickle_in = open(location, "rb")
    x = pickle.load(pickle_in)
    return x

def get_data_y_test(location):
    pickle_in = open(location, "rb")
    y = pickle.load(pickle_in)
    return y



save_data_x_train(X_train)
save_data_y_train(y_train)
save_data_x_test(X_test)
save_data_y_test(y_test)






          



