import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from network import Generator

import torch
from PIL import Image, ImageOps
import glob
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
            path_img = path + '/' + labels[img_idx] + '.png'
        else:
            path_img = path + '/' + prefix  + '_' + str(img_idx)+ '_' + labels[img_idx] + '.png'
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

parser = argparse.ArgumentParser(description='DiscoGAN in One Code')

# Task
parser.add_argument('--task', required=True, help='task or root name')

# Hyper-parameters
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')

# misc
parser.add_argument('--model_path', type=str, default='./models')  # Model Tmp Save
parser.add_argument('--sample_path', type=str, default='./test_results')  # Results
parser.add_argument('--run', type=int, default=0, help='test run')
parser.add_argument('--confexp', type=str, default='None', help='configuration of the experiment')

##### Helper Functions for Data Loading & Pre-processing
class ImageFolder(data.Dataset):
    def __init__(self, opt):
        self.task = opt.task
        self.transformP = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
                                             #transforms.Normalize((0.5, 0.5, 0.5),
                                                                  #(0.5, 0.5, 0.5))])
        self.transformS = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
                                             #transforms.Normalize((0.5, 0.5, 0.5),
                                                                  #(0.5, 0.5, 0.5))])
        self.image_len = None

        self.dir_base = './datasets'
        if self.task.startswith('mam'):
            self.root = os.path.join(self.dir_base, self.task)
            self.dir_AB = os.path.join(self.root, 'test')  # ./maps/train
            self.image_paths = list(map(lambda x: os.path.join(self.dir_AB, x), os.listdir(self.dir_AB)))
            self.image_len = len(self.image_paths)

        elif self.task == 'handbags2shoes': # handbags2shoes
            self.rootA = os.path.join(self.dir_base, 'edges2handbags')
            self.rootB = os.path.join(self.dir_base, 'edges2shoes')
            self.dir_A = os.path.join(self.rootA, 'val')
            self.dir_B = os.path.join(self.rootB, 'val')
            self.image_paths_A = list(map(lambda x: os.path.join(self.dir_A, x), os.listdir(self.dir_A)))
            self.image_paths_B = list(map(lambda x: os.path.join(self.dir_B, x), os.listdir(self.dir_B)))
            self.image_len = min(len(self.image_paths_A), len(self.image_paths_B))

        else: # facescrubs
            self.root = os.path.join(self.dir_base, 'facescrub')
            self.rootA = os.path.join(self.root, 'actors')
            self.rootB = os.path.join(self.root, 'actresses')
            self.dir_A = os.path.join(self.rootA, 'val') # You Should make your OWN Validation Set
            self.dir_B = os.path.join(self.rootB, 'val')
            self.image_paths_A = list(map(lambda x: os.path.join(self.dir_A, x), os.listdir(self.dir_A)))
            self.image_paths_B = list(map(lambda x: os.path.join(self.dir_B, x), os.listdir(self.dir_B)))
            self.image_len = min(len(self.image_paths_A), len(self.image_paths_B))

    def __getitem__(self, index):
        if self.task.startswith('mam'):
            AB_path = self.image_paths[index]
            AB = Image.open(AB_path).convert('L')
            AB = self.transformP(AB)

            w_total = AB.size(2)
            w = int(w_total / 2)

            A = AB[:, :256, :256]
            B = AB[:, :256, w:w + 256]

        elif self.task == 'handbags2shoes': # handbags2shoes
            A_path = self.image_paths_A[index]
            B_path = self.image_paths_B[index]
            A = Image.open(A_path).convert('RGB')
            B = Image.open(B_path).convert('RGB')

            A = self.transformP(A)
            B = self.transformP(B)

            w_total = A.size(2)
            w = int(w_total / 2)

            A = A[:, :64, w:w+64]
            B = B[:, :64, w:w+64]

        else: # Facescrubs
            A_path = self.image_paths_A[index]
            B_path = self.image_paths_B[index]
            A = Image.open(A_path).convert('RGB')
            B = Image.open(B_path).convert('RGB')

            A = self.transformS(A)
            B = self.transformS(B)

        return {'A': A, 'B': B, 'label': AB_path}

    def __len__(self):
        return self.image_len

##### Helper Function for GPU Training
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

##### Helper Function for Math
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

######################### Main Function
def main():
    # Pre-settings
    cudnn.benchmark = True
    global args
    args = parser.parse_args()
    print(args)

    dataset = ImageFolder(args)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=args.batchSize,
                                  shuffle=True,
                                  num_workers=2)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    # Networks
    g_pathAtoB = os.path.join(args.model_path, 'generatorAtoB.pkl')
    g_pathBtoA = os.path.join(args.model_path, 'generatorBtoA.pkl')

    generator_AtoB = Generator()
    generator_BtoA = Generator()

    generator_AtoB.load_state_dict(torch.load(g_pathAtoB))
    generator_AtoB.eval()

    generator_BtoA.load_state_dict(torch.load(g_pathBtoA))
    generator_BtoA.eval()

    if torch.cuda.is_available():
        generator_AtoB = generator_AtoB.cuda()
        generator_BtoA = generator_BtoA.cuda()

    testing_labels = []
    testing_images = np.zeros((200,256,256,1))
    realA = np.zeros((200,256,256,1))
    realB = np.zeros((200,256,256,1))

    """Train generator and discriminator."""
    total_step = len(data_loader) # For Print Log
    iter = 0
    img = 0
    for i, sample in enumerate(data_loader):
        input_A = sample['A']
        input_B = sample['B']

        real_A = to_variable(input_A)
        real_B = to_variable(input_B)

        # ===================== Forward =====================#
        fake_B = generator_AtoB(real_A)
        
        labels = []
        for k in sample['label']:
          label = k.split('/')[-1]
          labels.append(label.split('.')[0])
          testing_labels.append(label.split('.')[0])

        fake_B = denorm(fake_B)
        print(fake_B.shape)
        fake_B = fake_B.permute((0,2,3,1))
        fake_B = fake_B.cpu().detach().numpy()
        print(fake_B.shape)

        real_A = denorm(real_A)
        real_A = real_A.permute((0,2,3,1))
        real_A = real_A.cpu().detach().numpy()

        real_B = denorm(real_B)
        real_B = real_B.permute((0,2,3,1))
        real_B = real_B.cpu().detach().numpy()

        for k in range(0,len(labels)):
          testing_images[img] = fake_B[k]
          realA[img] = real_A[k]
          realB[img] = real_B[k]

          img = img + 1

        # print the log info
        print('Validation[%d/%d]' % (i + 1, total_step))
        # save the sampled images
        #res = torch.cat((torch.cat((real_A, fake_B), dim=3), real_B), dim=3)
        #torchvision.utils.save_image(denorm(res.data), os.path.join(args.sample_path, sample['label']))

    save_images("output/images/", np.concatenate((realA, testing_images,realB),axis=2), "images_"+str(args.run)+'_'+args.confexp, testing_labels, max_number_images=20, resize_dimensions=None, invert = False, training = False)
    np.savez_compressed('output/images/images_'+str(args.run)+'_'+args.confexp, testing_images)
    np.savez_compressed('output/images/labels_'+str(args.run)+'_'+args.confexp, testing_labels)
            
    write_line_to_csv(
          'CSAW_'+SETUP+'/results',SETUP + ".csv",
                {
                    "RUN": (run + 1),
                    "MAE": mean_absolute_error(realB,testing_images),
                    "PSNR": mean_psnr(realB,test_images),
                    "SSIM": mean_ssim(realB,testing_images)
                }) 

if __name__ == "__main__":
    main()