#
# Dobble Buddy - Utilities
#
# References:
#   https://www.kaggle.com/grouby/dobble-card-images
#
# Dependencies:
#   numpy
#   cv2
#   os
#   gc
#   csv
#   collections
#   keras


import numpy as np
import cv2

import os
import gc

import csv
from collections import OrderedDict

from keras import models

#
# Capture images/labels from data set for training and testing
#

def capture_card_filenames(directory_name):
    subdirs = ['{}/{}'.format(directory_name,i) for i in sorted(os.listdir(directory_name)) ]
    cards = []
    for i,subdir in enumerate(subdirs):
        cards += ['{}/{}'.format(subdir,i) for i in os.listdir(subdir)]
    del subdirs
    return cards


#
# Read images and pre-process to fixed size
#

def read_and_process_image(list_of_images,nrows,ncols):
    X = []
    y = []
    
    for i,image in enumerate(list_of_images):
        X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(nrows,ncols), interpolation=cv2.INTER_CUBIC))
        y_str = image.split('/')
        y.append(int(y_str[len(y_str)-2]))
    return X,y


#
# Create collage for cards 01-55 (5 rows, 11 columns)
#

def create_collage(deck_id,cards_X,cards_y):

    cards_idx = np.where(np.logical_and(cards_y>=1, cards_y<=55))
    cards_55 = cards_X[cards_idx]
    
    h,w,z = cards_X[0,:,:,:].shape
    w11 = w * 11
    h5 = h * 5
    collage = np.zeros((h5,w11,3),np.uint8)
    idx = 0
    for r in range(0,5):
        for c in range(0,11):
            collage[r*h:(r+1)*h,c*w:(c+1)*w,:] = cards_55[idx,:,:,:]
            idx = idx + 1

    return collage


#
# Load Symbol labels and Card-Symbol mapping
#

def load_symbol_labels( symbol_filename ):
    symbols = OrderedDict()
    with open(symbol_filename,'r') as file:
        reader = csv.reader(file)
        for row in reader:
            #print(row)
            if row[0] == '\ufeff1': # wierd character occuring on linux
                row[0] = '1'
            if row[0] == 'ï»¿1': # wierd character occuring on windows
                row[0] = '1'
            symbol_id = int(row[0])
            symbol_label = row[1]
            symbols[symbol_id] = symbol_label
    return symbols
    
#
# Load Card-Symbol mapping
#

def load_card_symbol_mapping( mapping_filename ):
    mapping = OrderedDict()
    with open(mapping_filename,'r') as file:
        reader = csv.reader(file)
        for row in reader:
            id = row[0]
            if row[0] == '\ufeff1': # wierd character occuring on linux
                row[0] = '1'
            if row[0] == 'ï»¿1': # wierd character occuring on windows
                row[0] = '1'
            card_id = int(row[0])
            card_mapping = []
            for i,val in enumerate(row[1:]):
                if val=='1':
                    card_mapping.append( i+1 )
            mapping[card_id] = card_mapping
    
    return mapping

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
