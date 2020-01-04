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

def get_data_x_test(location):
    pickle_in = open(location, "rb")
    x = pickle.load(pickle_in)
    return x

def get_data_y_test(location):
    pickle_in = open(location, "rb")
    y = pickle.load(pickle_in)
    return y

def get_data_x_train(location):
    pickle_in = open(location, "rb")
    x = pickle.load(pickle_in)
    return x

def get_data_y_train(location):
    pickle_in = open(location, "rb")
    y = pickle.load(pickle_in)
    return y

X_train = get_data_x_train("x_train.pickle")
#print(x[1])
y_train = get_data_y_train("y_train.pickle")

X_test = get_data_x_test("x_test.pickle")
#print(x[1])
y_test = get_data_y_test("y_test.pickle")
#print(y[1])
#y_binary = to_categorical(y)

X_train = X_train/255.0
X_test = X_test/255.0
           
model = Sequential()
            
model.add(Conv2D(64,(3,3), input_shape =(60,60,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
            
            
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
            
model.add(Flatten())
          
model.add(Dense(16))
model.add(Activation("relu"))

model.add(Dense(16))
model.add(Activation("relu")) 

           
model.add(Dense(8))
model.add(Activation("softmax"))
            
model.compile(loss="categorical_crossentropy", optimizer ="adam", metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size = 64, epochs=5)
model.evaluate(X_test,y_test)
#model.save("ashannew1234567.model")
#score = model.evaluate(x,y, verbose =0)
#print("Test loss:",score[0])
#print("Test accuracy:",score[1])

