import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
from util.util import tensor2im
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

def save_images(path, all_images, prefix, labels, max_number_images=None, resize_dimensions=None, invert = False, training = True):
    if not os.path.exists(path):
        os.makedirs(path)

    img_idx = 0

    for img_np in all_images:
        img_np = np.squeeze(img_np)
        img = Image.fromarray((img_np*255).astype('uint8'))
        if invert:
            img = ImageOps.invert(img)
        if resize_dimensions is not None:
            img = img.resize(resize_dimensions, Image.ANTIALIAS)
        if training: #Path doesnot include run and index
            path_img = path + '/' + labels[img_idx] 
        else:
            path_img = path + '/' + prefix  + '_' + str(img_idx)+ '_' + labels[img_idx]
        img.save(path_img)
        img_idx += 1

        if max_number_images is not None and img_idx == max_number_images:
            break
            
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

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    
    images_real_A = np.zeros(shape=(len(dataset), 256, 256,1))
    images_fake_B = np.zeros(shape=(len(dataset), 256, 256,1))
    images_real_B = np.zeros(shape=(len(dataset), 256, 256,1))
    labels_real = []
    counter = 0

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        
        visuals=model.get_current_visuals()

        images_real_A[counter] = np.expand_dims(visuals['real_A'][:,:,2],axis=-1)
        images_fake_B[counter] = np.expand_dims(visuals['fake_B'][:,:,2],axis=-1)
        images_real_B[counter] = np.expand_dims(visuals['real_B'][:,:,2],axis=-1)

        #visuals['real_A']=visuals['real_A'][:,:,0:3]
        #visuals['real_B']=visuals['real_B'][:,:,0:3]
        #visuals['fake_B']=visuals['fake_B'][:,:,0:3]    
        img_path = model.get_image_paths()
        img_path=img_path[0]
        img_path = img_path.split('/')[-1]
        labels_real.append(img_path)
        
        counter = counter + 1
        print('%04d: process image... %s' % (i, img_path))
    images_real_A = images_real_A /255
    images_real_B = images_real_B /255
    images_fake_B = images_fake_B /255
    save_images("output/images/", np.concatenate((images_real_A, images_fake_B,images_real_B),axis=2), opt.conf_exp +"_" + str(opt.run), labels_real, 20, None, False, False)
    
    np.savez_compressed('output/images/images_'+str(opt.run)+'_'+ opt.conf_exp, images_fake_B)
    np.savez_compressed('output/images/labels_'+str(opt.run)+'_'+ opt.conf_exp, labels_real)
    
    write_line_to_csv(
        'CSAW_'+SETUP+'/results',SETUP + ".csv",
              {
                  "RUN": (run + 1),
                  "MAE": mean_absolute_error(images_real_B,images_fake_B),
                  "PSNR": mean_psnr(images_real_B,images_fake_B),
                  "SSIM": mean_ssim(images_real_B,images_fake_B)
               }) 