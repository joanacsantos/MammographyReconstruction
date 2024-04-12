"""General-purpose test script for image-to-image translation.
Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.
It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.
Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.
    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import util, html

import numpy as np
import sys
import ntpath
import time
import util.reproducibility as rep_utils
from skimage.metrics import structural_similarity as ssim
import util.image as image_utils

from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import mse, psnr, uqi, vifp

def mean_absolute_error(x1, x2):
    return np.mean(np.abs(x1 - x2))

def mean_ssim(predicted_data,images_test):
    ssim_values = []
    for img_index in range(0, images_test.shape[0]):
      predicted_image = np.reshape(predicted_data[img_index], (256,256))
      test_image = np.reshape(images_test[img_index], (256,256))
      ssim_metric = ssim(predicted_image, test_image, data_range=1) 
      ssim_values.append(ssim_metric)
    return (sum(ssim_values) / len(ssim_values))

def mean_psnr(predicted_data,images_test):
    values = []
    for img_index in range(0, images_test.shape[0]):
      predicted_image = np.reshape(predicted_data[img_index], (256,256))
      test_image = np.reshape(images_test[img_index], (256,256))
      metric = psnr(predicted_image, test_image, MAX=1) 
      values.append(metric)
    return (sum(values) / len(values))   

def main_test(run, conf_exp, data_struct, n_runs, netG, gan_mode):
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.dataroot = 'datasets/testA'
    opt.run = run
    opt.conf_exp = conf_exp
    opt.no_dropout = True
    #opt.dataset_mode = data_struct
    opt.netG = netG
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # print(opt.run)
    # print(opt.m_rate)

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    images_missing = np.zeros(shape=(len(dataset),256,256,1)) 
    labels_missing = []
    images_fake = np.zeros(shape=(len(dataset),256,256,1))
    counter = 0
    for i, data in enumerate(dataset):
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        
        short_path = ntpath.basename(img_path[0])
        image_name = os.path.splitext(short_path)[0]

        #For real original images (dataset A)
        im_real = util.tensor2im(visuals['real']) 
        labels_missing.append(image_name)
        images_missing[counter] = np.dsplit(im_real,im_real.shape[-1])[0] #teve de vir com 3 layers iguais
        
        if i % 50 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
            #print(im_real.shape)

        #For fake images (dataset B)
        im_fake = util.tensor2im(visuals['fake']) 
        images_fake[counter] = np.dsplit(im_fake,im_fake.shape[-1])[0]
        counter = counter + 1
    images_missing = images_missing/255
    images_fake = images_fake/255
    
    print(len(images_missing))
    print(len(images_fake))

    opt2 = TestOptions().parse()
    opt2.num_threads = 0   
    opt2.batch_size = 1  
    opt2.serial_batches = True 
    opt2.no_flip = True  
    opt2.display_id = -1 
    opt2.dataroot = 'datasets/testB'
    dataset_real = create_dataset(opt2)
    print(opt2.dataroot)
    
    images_real = np.zeros(shape=(len(dataset_real),256,256,1))
    labels_real = []
    counter = 0
    for i, data in enumerate(dataset_real):
      im_real = util.tensor2im(data['A']) 
      images_real[counter] = np.dsplit(im_real,im_real.shape[-1])[0] #teve de vir com 3 layers iguais
      short_path = ntpath.basename(data['A_paths'][0])
      image_name = os.path.splitext(short_path)[0]
      labels_real.append(image_name)

      counter = counter + 1

    images_real = images_real/255

    image_utils.save_images("output/images/", np.concatenate((images_real, images_missing,images_fake),axis=2), conf_exp +"_" + str(opt.run), labels_missing, 20, None, False, False)

    np.savez_compressed('output/images/images_'+str(run)+'_'+ conf_exp, images_fake)
    np.savez_compressed('output/images/labels_'+str(run)+'_'+ conf_exp, labels_missing)
    
    write_line_to_csv(
          'CSAW_'+SETUP+'/results',SETUP + ".csv",
                {
                    "RUN": (run + 1),
                    "MAE": mean_absolute_error(images_missing,images_fake),
                    "PSNR": mean_psnr(images_missing,images_fake),
                    "SSIM": mean_ssim(images_missing,images_fake)
                }) 
