import numpy as np
from sklearn.model_selection import train_test_split
import time
import shutil
import os
import argparse
import pandas as pd

# -------- SOURCE ---------
REPRODUCIBILITY_DIR = "output/reproducibility/"
RESULTS_BASE_PATH = "output/results/"
IMAGES_DIR = "output/images/"
IMAGES_SUB_PATH = ["Original", "MissingValues", "Imputed","PreImputed"]
MODELS_DIR = "output/models/"

# ----- CONFIGURATION -----
NUMBER_RUNS = 30  # Number of executions for each configuration.
DATA_SPLIT = [0.8, 0.0, 0.2]  # Order: Train, Validation, Test. Values between 0 and 1.
INPUTE_NAN = False  # True places mising data as nan, False pre-imputes with 0.
SOURCE = "CSAW_2000"
START_RUN = 0  # Helpful when resuming an experiment...
NUMBER_IMAGES_TO_SAVE = 50  # Set to 0 to avoid saving images.
RESIZE_DIMENSIONS = (256, 256)  # Size of the saved images.
INVERT = False

from PIL import Image, ImageOps
import glob
import os
import numpy as np
import cv2
import csv
from pathlib import Path
import random
from scipy.stats import bernoulli

def save_images(path, all_images, prefix, labels, max_number_images=None, resize_dimensions=None, invert = False, training = True):
    if not os.path.exists(path):
        os.makedirs(path)

    img_idx = 0

    for img_np in all_images:
        img_np = np.squeeze(img_np)
        img = Image.fromarray((img_np * 255).astype('uint8'))
        if invert:
            img = ImageOps.invert(img)
        if resize_dimensions is not None:
            img = img.resize(resize_dimensions, Image.ANTIALIAS)
        if training: #Path doesnot include run and index
            path_img = path + '/' + labels[img_idx] + '.jpg'
        else:
            path_img = path + '/' + prefix  + '_' + str(img_idx)+ '_' + labels[img_idx] + '.jpg'
        img.save(path_img)
        img_idx += 1

        if max_number_images is not None and img_idx == max_number_images:
            break

def get_run_shuffle(rep_dir, source, run):

    return np.load("output/reproducibility/CSAW/runs_shuffle.npy", allow_pickle=True)[run]
    
def single_run_process_aligned(run, np_images,label):

    images = np_images.copy()

    print(len(label))
    label = list(label)
    labels_new = label

    mlo=[]; cc=[]
    for k in range(0,len(label)):
        if(label[k].split("_")[3]=="CC"):
            cc.append(k)
            mlo_name = label[k].split("_")
            mlo_name[3] = "MLO"
            mlo_name = "_".join(mlo_name)
            mlo.append(label.index(mlo_name))

    print(len(mlo))
    print(len(cc))

    order = get_run_shuffle(REPRODUCIBILITY_DIR, SOURCE, run)

    mlo = [mlo[i] for i in order]
    cc = [cc[i] for i in order]

    mlo_train, mlo_test = train_test_split(mlo, test_size=DATA_SPLIT[2], shuffle=False)
    cc_train, cc_test = train_test_split(cc, test_size=DATA_SPLIT[2], shuffle=False)

    del mlo, cc

    #Intercalate the values between mass and train within the separation (should only vary in the area of separation)

    images_with_mv_train_val = images[mlo_train]; images_train_val = images[cc_train]
    labels_save_train_a = [labels_new[i] for i in mlo_train]
    labels_save_train_b = [labels_new[i] for i in cc_train]
    print(len(labels_save_train_a))
    print(len(labels_save_train_b))

    images_with_mv_test = images[mlo_test]; images_test = images[cc_test]
    labels_save_test_a = [labels_new[i] for i in mlo_test]
    labels_save_test_b = [labels_new[i] for i in cc_test]

    del mlo_train, cc_train


    save_images("data/mammo/train/A/", images_train_val , SOURCE + "_" + str(run), labels_save_train_a, None, None,INVERT, True)
    save_images("data/mammo/train/B/", images_with_mv_train_val , SOURCE + "_" + str(run), labels_save_train_a, None, None,INVERT, True)

    save_images("data/mammo/test/A/", images_test , SOURCE + "_" + str(run), labels_save_test_a, None, None,INVERT, True)
    save_images("data/mammo/test/B/", images_with_mv_test , SOURCE + "_" + str(run), labels_save_test_a, None, None,INVERT, True)

    return(0)
    
parser = argparse.ArgumentParser(description='Easy Implementation of UNIT')

parser.add_argument('--run', type=int, default=0, help='test run')

######################### Main Function
def main():
    args = parser.parse_args()
    run = args.run
    
    #Load Initial Dataset
    with open('CSAW_2000.npy', 'rb') as f:
    	images = np.load(f)

    with open('CSAW_2000_labels.npy', 'rb') as f:
    	label = np.load(f)

    #Create the new datasets for the run
    print('Creating the new datasets...')

    single_run_process_aligned(run,images,label)
    
if __name__ == "__main__":
    main()
    