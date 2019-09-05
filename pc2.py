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



datadir = "F:/Machine learning/kvasir-dataset-v2" #path to images folder
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
                #resized_img_array = cv2.resize(img_array,(img_size,img_size))
                #plt.imshow(resized_img_array,cmap="gray")
                #plt.show()
                training_data_set.append([resized_img_array,class_index]) #append data to training data set
            except Exception as e:
                pass
    random.shuffle(training_data_set) #shuffle the data
    return training_data_set

img_size = 60 #image size
training_data_set = create_training_data_set(img_size)
#print(len(training_data_set))       
#for i in training_data_set[:10]:
#    print(i[1])

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

def save_data_x(x):
    pickle_out = open("x.pickle","wb")
    pickle.dump(x,pickle_out)
    pickle_out.close()
    
def save_data_y(y):    
    pickle_out = open("y.pickle","wb")
    pickle.dump(y,pickle_out)
    pickle_out.close()
    
def get_data_x(location):
    pickle_in = open(location, "rb")
    x = pickle.load(pickle_in)
    return x

def get_data_y(location):
    pickle_in = open(location, "rb")
    y = pickle.load(pickle_in)
    return y

save_data_x(x)
save_data_y(y)
#z = get_data_y("y.pickle")
#print(z[1])





          



