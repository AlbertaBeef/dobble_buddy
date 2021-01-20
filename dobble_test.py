# Script to test the accuracy of your model and give error bounds
import cv2
import os
import math
import numpy as np
import keras
from keras.models import load_model
from keras.utils import to_categorical

import dobble_utils as db

# Open dobble model
model = load_model('dobble_model.h5')
dir = './dobble_dataset'
nrows = 224
ncols = 224

nchannels = 3


#  predict Error bounds
_zValues = { .5:.67, .68:1.0, .8:1.28, .9:1.64, .95:1.96, .98:2.33, .99:2.58 }

def get_accuracy_bounds(mean, sampleSize, confidence):
    if mean < 0.0 or mean > 1.0:
        raise UserWarning("mean must be between 0 and 1")

    if sampleSize <= 0:
        raise UserWarning("sampleSize should be positive")
        
    # lookup the zValue depending on the confidence
    zvalue = _zValues.get(confidence)
    # get the standard deviation
    stdev = math.sqrt( (mean * (1 - mean)) / sampleSize) 
    # multiply the standard deviation by the zvalue
    interval = zvalue * stdev
    lower = mean - interval
    upper = mean + interval
    return (lower, upper)

#test_dir = './dobble_dataset/dobble_test01_cards'
test_dir = './dobble_dataset/dobble_test02_cards'


test_cards = db.capture_card_filenames(test_dir)
test_set_size = len(test_cards)
np.random.shuffle(test_cards)

test_X,test_y = db.read_and_process_image(test_cards,nrows,ncols)
del test_cards

ntest = len(test_y)

test_X = np.array(test_X)
test_y = np.array(test_y)

# normalize images
test_X = test_X * (1./255)

# convert labels in range 0-57 to one-hot encoding
test_y = to_categorical(test_y,58)

print("Shape of test data (X) is :", test_X.shape)
print("Shape of test data (y) is :", test_y.shape)


print("")
print("EVALUATE MODEL:")
model.evaluate(test_X,test_y)

test_accuracy = db.test_accuracy(model,ntest,test_X,test_y)
print(test_dir," : Test Accuracy = ", test_accuracy)


for confidence in [.5, .8, .9, .95, .99]:
    (lowerBound, upperBound) = get_accuracy_bounds(test_accuracy, test_set_size, confidence)    
    print(" %.2f%% accuracy bound: %.4f - %.4f" % (confidence, lowerBound, upperBound))
