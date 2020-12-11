#
# DOBBLE DataSet Overview
#
# References:
#   https://www.kaggle.com/grouby/dobble-card-images
#
# Dependencies:
#
# Kaggle:
# python3 -m pip install kaggle

import cv2
import numpy as np

#import matplotlib.pyplot as plt
#%matplotlib inline

import os
import gc
import zipfile


#
# Parameters
#

# Replace the following username with the one from the json file for your Kaggle API Token
os.environ['KAGGLE_USERNAME'] = "aidventure"
# Replace this key with the one from the json file for your Kaggle API Token
os.environ['KAGGLE_KEY'] = "36f92c166af715c1943acbaeb07e18d7"

dir = './dobble_dataset'
#dir = './kaggle/input/dobble-card-images'
nrows = 146
ncols = 146
#nrows = 224
#ncols = 224

nchannels = 3

# Verify that the dataset directory is present, otherwise download it from Kaggle.
if not os.path.isdir(dir):
    print("Downloading workable dataset...")
    # Download the official image set used for network training and validation
    os.system("kaggle datasets download -d grouby/dobble-card-images")
    with zipfile.ZipFile('dobble-card-images.zip', 'r') as zip_ref:
        zip_ref.extractall(dir)
else:
    print("Found local dataset for training and validation:", dir)

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

train1_cards = capture_card_filenames(train1_dir)
train2_cards = capture_card_filenames(train2_dir)
train3_cards = capture_card_filenames(train3_dir)
train4_cards = capture_card_filenames(train4_dir)
train5_cards = capture_card_filenames(train5_dir)
train6_cards = capture_card_filenames(train6_dir)
train7_cards = capture_card_filenames(train7_dir)


#
# Read images and pre-process to fixed size
#

def read_and_process_image(list_of_images):
    X = []
    y = []
    
    for i,image in enumerate(list_of_images):
        X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(nrows,ncols), interpolation=cv2.INTER_CUBIC))
        y_str = image.split('/')
        y.append(int(y_str[len(y_str)-2]))
    return X,y

train1_X,train1_y = read_and_process_image(train1_cards)
train2_X,train2_y = read_and_process_image(train2_cards)
train3_X,train3_y = read_and_process_image(train3_cards)
train4_X,train4_y = read_and_process_image(train4_cards)
train5_X,train5_y = read_and_process_image(train5_cards)
train6_X,train6_y = read_and_process_image(train6_cards)
train7_X,train7_y = read_and_process_image(train7_cards)

train1_X = np.array(train1_X)
train1_y = np.array(train1_y)
train2_X = np.array(train2_X)
train2_y = np.array(train2_y)
train3_X = np.array(train3_X)
train3_y = np.array(train3_y)
train4_X = np.array(train4_X)
train4_y = np.array(train4_y)
train5_X = np.array(train5_X)
train5_y = np.array(train5_y)
train6_X = np.array(train6_X)
train6_y = np.array(train6_y)
train7_X = np.array(train7_X)
train7_y = np.array(train7_y)

print("")
print("TRAINING DATA SET:")
print("Shape of deck01 data (X) is :", train1_X.shape)
print("Shape of deck02 data (X) is :", train2_X.shape)
print("Shape of deck03 data (X) is :", train3_X.shape)
print("Shape of deck04 data (X) is :", train4_X.shape)
print("Shape of deck05 data (X) is :", train5_X.shape)
print("Shape of deck06 data (X) is :", train6_X.shape)
print("Shape of deck07 data (X) is :", train7_X.shape)

#
# Display card decks (cards 1-55)
#

def create_collage(title,cards_X,cards_y):

    h,w,z = cards_X[0,:,:,:].shape
    #print(w,h)
    w11 = w * 11
    h5 = h * 5
    collage = np.zeros((h5,w11,3),np.uint8)
    idx = 0
    for r in range(0,5):
        for c in range(0,11):
            collage[r*h:(r+1)*h,c*w:(c+1)*w,:] = cards_X[idx,:,:,:]
            idx = idx + 1

    cv2.imshow(title,collage)
    cv2.waitKey(0)
    #imgplot = plt.imshow(cv2.cvtColor(collage,cv2.COLOR_BGR2RGB))
    #plt.title(title)
    #plt.show()

print("")
print("VIEW CARD DECKS:")
print("... press any key to continue ...")

create_collage("Deck 01",train1_X,train1_y)
create_collage("Deck 02",train2_X,train2_y)
create_collage("Deck 03",train3_X,train3_y)
create_collage("Deck 04",train4_X,train4_y)
create_collage("Deck 05",train5_X,train5_y)
create_collage("Deck 06",train6_X,train6_y)
create_collage("Deck 07",train7_X,train7_y)

#
# Display 57 cards
#

train_X = np.concatenate( (train1_X,train2_X,train3_X,train4_X,train5_X,train6_X,train7_X), axis=0 )
train_y = np.concatenate( (train1_y,train2_y,train3_y,train4_y,train5_y,train6_y,train7_y), axis=0 )
ntrain = len(train_y)

print("")
print("VIEW CARDS:")
print("... press any key to continue (ESC to quit) ...")

for i in range(1,57+1):
#for i in range(1,5):
    idx = (train_y==i) 
    cards_X = train_X[idx]
    cards_y = train_y[idx]
    ncards = len(cards_y)
    img = cards_X[0,:,:,:]
    for c in range(1,ncards):
       img = cv2.hconcat([img,cards_X[c,:,:,:]])
    title = "Dobble Card " + str(i)
    cv2.imshow(title,img)
    key = cv2.waitKey(0)
    if key == 27:
        break
    #imgplot = plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #plt.title(title)
    #plt.show()

#
# Load Symbol labels and Card-Symbol mapping
#

import csv
from collections import OrderedDict

symbols = OrderedDict()
with open(dir+'/dobble_symbols.txt','r') as file:
    reader = csv.reader(file)
    for row in reader:
        #print(row)
        if row[0] == '\ufeff1': # wierd character occuring on linux
            row[0] = '1'
        if row[0] == 'ï»¿1': # wierd character occuring on windows
            row[0] = '1'
        symbol_id = int(row[0])
        symbol_label = row[1]
        #print(symbol_id,symbol_label)
        symbols[symbol_id] = symbol_label

print("")
print("SYMBOLS:")
print(symbols)

mapping = OrderedDict()
with open(dir+'/dobble_card_symbol_mapping.txt','r') as file:
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
        #print(card_id,card_mapping)
        mapping[card_id] = card_mapping

print("")
print("MAPPING:")
print(mapping)

#
# Display matching symbols
#

import random

print("")
print("VIEW MATCHING SYMBOLS:")
print("... press any key to continue (ESC to quit) ...")


for i in range(100):
#for i in range(10):
    idx1 = random.randrange(0,ntrain)
    idx2 = random.randrange(0,ntrain)
    #print(idx1,idx2)
    card1_X = train_X[idx1,:,:,:]
    card1_y = train_y[idx1]
    card2_X = train_X[idx2,:,:,:]
    card2_y = train_y[idx2]
    if ( card1_y == card2_y ):
        continue;

    card1_mapping = mapping[card1_y]
    card2_mapping = mapping[card2_y]
    symbol_ids = np.intersect1d(card1_mapping,card2_mapping)
    symbol_id = symbol_ids[0]
    symbol_label = symbols[symbol_id]
    img = cv2.hconcat( [card1_X,card2_X] )
    title = "Example " + str(i) + " : Matching Symbol = " + symbol_label
    print(title)
    cv2.imshow(title,img)
    key = cv2.waitKey(0)
    if key == 27:
        break
    #imgplot = plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #plt.title(title)
    #plt.show()

