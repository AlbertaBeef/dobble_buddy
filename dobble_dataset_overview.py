#
# Dobble Buddy - DataSet Overview
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


import cv2
import numpy as np

import os
import gc

import dobble_utils as db

#
# Parameters
#
dir = './dobble_dataset'

nrows = 146
ncols = 146
#nrows = 224
#ncols = 224

nchannels = 3

card_decks = [
    'dobble_deck01_cards_57',
    'dobble_deck02_cards_55',
    'dobble_deck03_cards_55',
    'dobble_deck04_cards_55',
    'dobble_deck05_cards_55',
    'dobble_deck06_cards_55',
    'dobble_deck07_cards_55',
    'dobble_deck08_cards_55',
    'dobble_deck09_cards_55',
    'dobble_deck10_cards_55'
    ]
nb_card_decks = len(card_decks)
print("")
print("PARAMETERS:")
print("Normalized shape of images :", ncols, " x ", nrows )
print("Card Decks : ", nb_card_decks, card_decks)

#
# Capture images/labels from data set for training and testing
#

train_cards = []
for d in range(0,nb_card_decks):
    train_dir = dir+'/'+card_decks[d]
    train_cards.append( db.capture_card_filenames(train_dir) )


#
# Read images and pre-process to fixed size
#

train_X = []
train_y = []
for d in range(0,nb_card_decks):
   X,y = db.read_and_process_image(train_cards[d],nrows,ncols)
   train_X.append( np.array(X) )
   train_y.append( np.array(y) )


print("")
print("TRAINING DATA SET:")
for d in range(0,nb_card_decks):
    print("Shape of deck%02d data (X) is :"%(d+1), train_X[d].shape)

#
# Display card decks (cards 1-55)
#


print("")
print("VIEW CARD DECKS:")
print("... press any key to continue ...")

for d in range(0,nb_card_decks):
    collage = db.create_collage(d+1,train_X[d],train_y[d])
    title = "Deck %02d"%(d+1)
    cv2.imshow(title,collage)
    cv2.waitKey(0)
    

#
# Display 57 cards
#

train_X = np.concatenate( train_X, axis=0 )
train_y = np.concatenate( train_y, axis=0 )
ntrain = len(train_y)

print("")
print("VIEW CARDS:")
print("... press any key to continue (ESC to quit) ...")

for i in range(0,57+1):
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

#
# Load Symbol labels and Card-Symbol mapping
#

symbols = db.load_symbol_labels(dir+'/dobble_symbols.txt')

print("")
print("SYMBOLS:")
print(symbols)

mapping = db.load_card_symbol_mapping(dir+'/dobble_card_symbol_mapping.txt')

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
    idx1 = random.randrange(1,ntrain+1)
    idx2 = random.randrange(1,ntrain+1)
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

