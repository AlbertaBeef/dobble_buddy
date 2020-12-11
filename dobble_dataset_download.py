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

import os
import zipfile

#
# Parameters
#

# Specify the directory where the local training/verification dataset will be 
# located.
dir = './dobble_dataset'

#
# See the official Kaggle documentation on API keys for more information:
#
# https://github.com/Kaggle/kaggle-api
#

# Replace the following username with the one from the json file for your 
# Kaggle API Token
KAGGLE_USERNAME = "default"

# Replace this key with the one from the json file for your Kaggle API Token
KAGGLE_KEY = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# If KAGGLE_USERNAME and KAGGLE_KEY are defined with real values, they will be
# used to access the Kaggle dataset.  If not, then attempt to download using
# any API key info that might be found in the kaggle.json file of the 
# $HOME_PATH/.kaggle/ folder.
# 
if not KAGGLE_USERNAME == "default":
    os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
if not KAGGLE_KEY == "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX":
    os.environ['KAGGLE_KEY'] = KAGGLE_KEY

# Verify that the local dataset directory is present, otherwise download it 
# from Kaggle.
if not os.path.isdir(dir):
    print("Downloading workable dataset...")
    # Download the official image set used for network training and validation
    os.system("kaggle datasets download -d grouby/dobble-card-images")
    with zipfile.ZipFile('dobble-card-images.zip', 'r') as zip_ref:
        zip_ref.extractall(dir)
else:
    print("Found local dataset for training and validation:", dir)
