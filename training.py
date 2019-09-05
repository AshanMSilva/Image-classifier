# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 19:21:23 2019

@author: User
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
#from keras.utils import to_categorical
import pickle
import time

def get_data_x(location):
    pickle_in = open(location, "rb")
    x = pickle.load(pickle_in)
    return x

def get_data_y(location):
    pickle_in = open(location, "rb")
    y = pickle.load(pickle_in)
    return y

x = get_data_x("x.pickle")
#print(x[1])
y = get_data_y("y.pickle")
#print(y[1])
#y_binary = to_categorical(y)
x = x/255.0


            
model = Sequential()
            
model.add(Conv2D(64,(3,3), input_shape =(60,60,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
            
            
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
            
            
            
model.add(Flatten())
            
model.add(Dense(64))
model.add(Activation("relu"))
            
model.add(Dense(8))
model.add(Activation("sigmoid"))
            
model.compile(loss="sparse_categorical_crossentropy", optimizer ="adam", metrics=['accuracy'])
model.fit(x,y,batch_size = 30, epochs=50, validation_split = 0.1)

model.save("ashannew.model")
#score = model.evaluate(x,y, verbose =0)
#print("Test loss:",score[0])
#print("Test accuracy:",score[1])
