import util.image as image_utils
import util.reproducibility as rep_utils
import numpy as np
from sklearn.model_selection import train_test_split
import time
import shutil
import os

import train as train
import test as test

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
"  
START_RUN = 0  # Helpful when resuming an experiment...
NUMBER_IMAGES_TO_SAVE = 50  # Set to 0 to avoid saving images.  
RESIZE_DIMENSIONS = (256, 256)  # Size of the saved images. #reduce size
INVERT = False

def single_run_process_unaligned(run, np_images,label): 

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

    order = rep_utils.get_run_shuffle(REPRODUCIBILITY_DIR, SOURCE, run)
    
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

    #aligned_images = np.concatenate((images_with_mv_train_val,images_train_val),axis=2)
    #print(aligned_images.shape)
    del mlo_train, cc_train 

    image_utils.save_images("datasets/trainA/", images_with_mv_train_val , SOURCE + "_" + str(run), labels_save_train_a, None, RESIZE_DIMENSIONS,INVERT, True)
    image_utils.save_images("datasets/trainB/", images_train_val, SOURCE + "_" + str(run), labels_save_train_b, None, RESIZE_DIMENSIONS, INVERT, True)

    image_utils.save_images("datasets/testA/", images_with_mv_test, SOURCE + "_" + str(run), labels_save_test_a, None, RESIZE_DIMENSIONS, INVERT, True)
    image_utils.save_images("datasets/testB/", images_test, SOURCE + "_" + str(run), labels_save_test_b, None, RESIZE_DIMENSIONS, INVERT, True)

    return(0)

def run_smoothly_unaligned(run, data_struct, n_runs, netG, gan_mode):
    conf_exp = 'MLOtoCC_CycleGAN_' + data_struct + '_' + str(2*n_runs) + '_' + netG + '_' + gan_mode 
   
    #Remove the previous datasets
    print('Removing previous datasets...')
    shutil.rmtree('checkpoints/mammo100', ignore_errors=True)
    shutil.rmtree('datasets/trainA', ignore_errors=True)
    shutil.rmtree('datasets/trainB', ignore_errors=True)
    shutil.rmtree('datasets/train', ignore_errors=True)
    shutil.rmtree('datasets/testA', ignore_errors=True)
    shutil.rmtree('datasets/testB', ignore_errors=True)  
    
    single_run_process_unaligned(run,images,label)  
    
    seconds = time.time()
    train.main_train(run,conf_exp, data_struct, n_runs, netG, gan_mode)
    ending = time.time()
    print('Training time (h): ' + str((ending-seconds)/60/60))
    
    rep_utils.write_line_to_csv(
	'Time', ".csv",
              {
                  "RUN": (run + 1),
                  "TYPE": "MLOtoCC",
                  "TIME": (ending-seconds)/60/60
               }) 
        
    #Rename the generator and discriminator for the purposes of transforming B to A
    os.rename('checkpoints/mammo100/latest_net_G_A.pth','checkpoints/mammo100/latest_net_G.pth')
    os.rename('checkpoints/mammo100/latest_net_D_A.pth','checkpoints/mammo100/latest_net_D.pth')  
        
    seconds = time.time()
    print('Beginning testing run '+ str(run) + ' for ' + str(conf_exp))
    test.main_test(run, conf_exp, data_struct, n_runs, netG, gan_mode)
    ending = time.time()
    print('Testing time (h): ' + str((ending-seconds)/60/60))
    
def single_run_process_unaligned2(run, np_images,label): 

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

    order = rep_utils.get_run_shuffle(REPRODUCIBILITY_DIR, SOURCE, run)
    
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

    #aligned_images = np.concatenate((images_with_mv_train_val,images_train_val),axis=2)
    #print(aligned_images.shape)
    del mlo_train, cc_train 

    image_utils.save_images("datasets/trainB/", images_with_mv_train_val , SOURCE + "_" + str(run), labels_save_train_a, None, RESIZE_DIMENSIONS,INVERT, True)
    image_utils.save_images("datasets/trainA/", images_train_val, SOURCE + "_" + str(run), labels_save_train_b, None, RESIZE_DIMENSIONS, INVERT, True)

    image_utils.save_images("datasets/testB/", images_with_mv_test, SOURCE + "_" + str(run), labels_save_test_a, None, RESIZE_DIMENSIONS, INVERT, True)
    image_utils.save_images("datasets/testA/", images_test, SOURCE + "_" + str(run), labels_save_test_b, None, RESIZE_DIMENSIONS, INVERT, True)

    return(0)
    
def run_smoothly_unaligned2(run, data_struct, n_runs, netG, gan_mode):
    conf_exp = 'CCtoMLO_CycleGAN_' + data_struct + '_' + str(2*n_runs) + '_' + netG + '_' + gan_mode 
   
    #Remove the previous datasets
    print('Removing previous datasets...')
    shutil.rmtree('checkpoints/mammo100', ignore_errors=True)
    shutil.rmtree('datasets/trainA', ignore_errors=True)
    shutil.rmtree('datasets/trainB', ignore_errors=True)
    shutil.rmtree('datasets/train', ignore_errors=True)
    shutil.rmtree('datasets/testA', ignore_errors=True)
    shutil.rmtree('datasets/testB', ignore_errors=True)  
    
    single_run_process_unaligned2(run,images,label)  
    
    seconds = time.time()
    train.main_train(run,conf_exp, data_struct, n_runs, netG, gan_mode)
    ending = time.time()
    print('Training time (h): ' + str((ending-seconds)/60/60))
        
    #Rename the generator and discriminator for the purposes of transforming B to A
    os.rename('checkpoints/mammo100/latest_net_G_A.pth','checkpoints/mammo100/latest_net_G.pth')
    os.rename('checkpoints/mammo100/latest_net_D_A.pth','checkpoints/mammo100/latest_net_D.pth')  
        
    seconds = time.time()
    print('Beginning testing run '+ str(run) + ' for ' + str(conf_exp))
    test.main_test(run, conf_exp, data_struct, n_runs, netG, gan_mode)
    ending = time.time()
    print('Testing time (h): ' + str((ending-seconds)/60/60))

##########Start the code :D


#Load Initial Dataset
with open('CBIS_DDSM_image_pares.npy', 'rb') as f:
  images = np.load(f)
#images = np.expand_dims(images, axis=3)

with open('CBIS_DDSM_labels_pares.npy', 'rb') as f:
  label = np.load(f)

#Create the new datasets for the run
print('Creating the new datasets...')

for run in range(0,30):
 run_smoothly_unaligned(run, 'unaligned', 100, 'resnet_6blocks', 'vanilla')
 run_smoothly_unaligned2(run, 'unaligned', 100, 'resnet_6blocks', 'vanilla')
