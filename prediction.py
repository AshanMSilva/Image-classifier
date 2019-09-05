# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 12:28:03 2019

@author: User
"""

import cv2
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from matplotlib import pyplot as plt

categories = ['dyed-lifted-polyps','dyed-resection-margins','esophagitis','normal-cecum','normal-pylorus','normal-z-line','polyps','ulcerative-colitis']

def prepare(path):
    img_size =60
    img_array = cv2.imread(path) #image in gray color
    img_array_rgb = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
    resized_img_array = cv2.resize(img_array_rgb,(img_size,img_size))
    #plt.imshow(resized_img_array)
    #plt.show()
    x=[]
    x.append(resized_img_array)
    x = np.array(x).reshape(1,img_size,img_size,3)
    return x

model = tf.keras.models.load_model("ashannew.model")
#model.layers[0].input_shape(1,60,60,3)
path = "ac.jpg"
img = image.load_img(path,target_size=(60,60))
plt.imshow(img)
img=np.expand_dims(img, axis =0)
result = model.predict_classes(img)

plt.title(categories[np.argmax(resul)])
plt.show()

#prediction = model.predict([prepare("bb.jpg")])

#y = np.argmax(prediction)

#print(y)
#model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#img = cv2.imread('test.jpg')
#img = cv2.resize(img,(320,240))
#img = np.reshape(img,[1,320,240,3])

#classes = model.predict_classes(prepare("ac.jpg"))

#print(classes)
x =prepare("ac.jpg")
#y=[]
#y.append(2)
#print(x)
#score = model.evaluate(x,y, verbose =0)
#print(score[0])
#print(score[1])

#prediction = model.predict(x)
#print(prediction)