#
# DOBBLE CHALLENGE
#
# References:
#   https://www.kaggle.com/grouby/dobble-card-images
#   https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9
#
# Installation stuff:
#   If Keras is complaining about tensorflor 1.14 <= 2.2:
#      sudo apt install python3-pip
#      pip3 install --upgrade pip
#      python3 -m pip install tensorflow


import cv2
import numpy as np
#import pandas as pd

import matplotlib.pyplot as plt
#%matplotlib inline

import os
import random
import gc

#
# Parameters
#

dir = './dobble_dataset'
#dir = './kaggle/input/dobble-card-images'
#nrows = 146
#ncols = 146
nrows = 224
ncols = 224

nchannels = 3

print("")
print("PARAMETERS:")
print("Normalized shape of images :", ncols, " x ", nrows )

#
# Capture images/labels from data set for training and testing
#

def capture_card_filenames(directory_name):
    subdirs = ['{}/{}'.format(directory_name,i) for i in os.listdir(directory_name) ]
    #labels = os.listdir(directory_name)
    cards = []
    for i,subdir in enumerate(subdirs):
        cards += ['{}/{}'.format(subdir,i) for i in os.listdir(subdir)]
    #random.shuffle(cards)
    del subdirs
    return cards

train1_dir = dir+'/dobble_deck01_cards_57'
train2_dir = dir+'/dobble_deck02_cards_55'
train3_dir = dir+'/dobble_deck03_cards_55'
train4_dir = dir+'/dobble_deck04_cards_55'
train5_dir = dir+'/dobble_deck05_cards_55'
train6_dir = dir+'/dobble_deck06_cards_55'
train7_dir = dir+'/dobble_deck07_cards_55'

train_cards = capture_card_filenames(train1_dir) + capture_card_filenames(train2_dir) + capture_card_filenames(train3_dir) + capture_card_filenames(train4_dir) + capture_card_filenames(train5_dir) + capture_card_filenames(train6_dir)  + capture_card_filenames(train7_dir)
random.shuffle(train_cards)

gc.collect()

#
# Read images and pre-process to fixed size
#

def read_and_process_image(list_of_images):
    X = []
    y = []
    
    for i,image in enumerate(list_of_images):
        X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(nrows,ncols), interpolation=cv2.INTER_CUBIC))
        y_str = image.split('/')[3]
        y.append(int(y_str))
    return X,y

train_X,train_y = read_and_process_image(train_cards)

del train_cards
gc.collect()

#
# Split training data set down into two data sets : training(80%) and validation(20%)
#

train_X = np.array(train_X)
train_y = np.array(train_y)

print("")
print("TRAINING DATA SET:")
print("Shape of training data (X) is :", train_X.shape)
print("Shape of training data (y) is :", train_y.shape)

from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y = train_test_split(train_X,train_y, test_size=0.20, random_state=2)

print("Shape of training data (X) is :", train_X.shape)
print("Shape of training data (y) is :", train_y.shape)
print("Shape of validation data (X) is :", val_X.shape)
print("Shape of validation data (y) is :", val_y.shape)


#
# Create model
#

ntrain = len(train_X)
nval   = len(val_X)
batch_size = 32
nepochs = 100

from keras import layers
from keras import models
from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

# convert labels in range 1-57, to range 0-56, then to one-hot encoding
train_y = to_categorical(train_y-1,57)
val_y = to_categorical(val_y-1,57)

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(nrows,ncols,nchannels)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dense(57))
model.add(layers.Activation('softmax'))

#model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
model.compile(loss='categorical_crossentropy',optimizer='adam')

print("")
print("MODEL SUMMARY:")
model.summary()

print("")
print("TRAIN MODEL:")

train_datagen = ImageDataGenerator( 
	rescale=1./255,
	rotation_range=360, 
	horizontal_flip=True 
	)
val_datagen   = ImageDataGenerator( 
	rescale=1./255
	)


train_generator = train_datagen.flow(train_X,train_y,batch_size=batch_size)
val_generator = val_datagen.flow(val_X,val_y,batch_size=batch_size)

history = model.fit(
	train_generator,
	steps_per_epoch=int(ntrain/batch_size),
	epochs=nepochs,
	validation_data=val_generator,
	validation_steps=int(nval/batch_size)
	)

model.save_weights('dobble_model_weights.h5')
model.save('dobble_model.h5')

#
# Test Model Accuracy
#

def test_accuracy(model, test_n, test_X, test_y):
    ntotal = 0
    ncorrect = 0
    predictions = model.predict(test_X)
    for i in range(test_n):
        y = test_y[i,:]
        pred = predictions[i,:]
        max_y = np.argmax(y)
        max_pred = np.argmax(pred)
        ntotal += 1
        if max_pred == max_y:
            ncorrect += 1
    return ncorrect/ntotal

model.summary()

test1_dir = './dobble_dataset/dobble_test01_cards'
test1_cards = capture_card_filenames(test1_dir)
random.shuffle(test1_cards)

test1_X,test1_y = read_and_process_image(test1_cards)
del test1_cards

ntest1 = len(test1_y)

test1_X = np.array(test1_X)
test1_y = np.array(test1_y)

# normalize images
test1_X = test1_X * (1./255)

# convert labels in range 1-57, to range 0-56, then to one-hot encoding
test1_y = to_categorical(test1_y-1,57)

print("Shape of test1 data (X) is :", test1_X.shape)
print("Shape of test1 data (y) is :", test1_y.shape)


print("")
print("EVALUATE MODEL:")
model.evaluate(test1_X,test1_y)

test1_accuracy = test_accuracy(model,ntest1,test1_X,test1_y)
print(test1_dir," : Test Accuracy = ", test1_accuracy)



