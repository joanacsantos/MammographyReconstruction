import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
from PIL import Image, ImageOps
import glob
import os
import numpy as np
import cv2
import csv
from pathlib import Path
import random
from skimage.metrics import structural_similarity as ssim
from scipy.stats import bernoulli
import time

def save_images(path, all_images, prefix, labels, max_number_images=None, resize_dimensions=None, invert = False, training = False):
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
        if training: #Path does not include run and index
            path_img = path + '/' + labels[img_idx] + '.png'
        else:
            path_img = path + '/' + prefix  + '_' + str(img_idx)+ '_' + labels[img_idx] + '.png'
        img.save(path_img)
        img_idx += 1

        if max_number_images is not None and img_idx == max_number_images:
            break

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

import numpy as np
import os
import csv
from pathlib import Path
import random
from scipy.stats import bernoulli
from sewar.full_ref import mse, psnr, uqi, vifp

def write_line_to_csv(dir_path, file, data_row):

    file_path = dir_path + file
    file_exists = Path(file_path).is_file()

    if file_exists:
        file_csv = open(file_path, 'a')
    else:
        file_csv = open(file_path, 'w')

    writer = csv.DictWriter(file_csv, delimiter=',', fieldnames=[*data_row], quoting=csv.QUOTE_NONE)

    if not file_exists:
        writer.writeheader()

    writer.writerow(data_row)

    file_csv.flush()
    file_csv.close()
    
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

def mean_squared_error(predicted_data,images_test):
    values = []
    for img_index in range(0, images_test.shape[0]):
      predicted_image = np.reshape(predicted_data[img_index], (256,256))
      test_image = np.reshape(images_test[img_index], (256,256))
      metric = mse(predicted_image*255, test_image*255) 
      values.append(metric)
    return (sum(values) / len(values))

def mean_psnr(predicted_data,images_test):
    values = []
    for img_index in range(0, images_test.shape[0]):
      predicted_image = np.reshape(predicted_data[img_index], (256,256))
      test_image = np.reshape(images_test[img_index], (256,256))
      metric = psnr(predicted_image, test_image, MAX=1) 
      values.append(metric)
    print(values)
    return (sum(values) / len(values))  
    
def mean_uqi(predicted_data,images_test):
    values = []
    for img_index in range(0, images_test.shape[0]):
      predicted_image = np.reshape(predicted_data[img_index], (256,256))
      test_image = np.reshape(images_test[img_index], (256,256))
      metric = uqi(predicted_image*255, test_image*255) 
      values.append(metric)
    return (sum(values) / len(values))  
    
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, default=0, help="run that is being trained")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="mammo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_pixel = torch.nn.L1Loss()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Dimensionality (channel-wise) of image embedding
shared_dim = opt.dim * 2 ** opt.n_downsample

# Initialize generator and discriminator
shared_E = ResidualBlock(features=shared_dim)
E1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
E2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
shared_G = ResidualBlock(features=shared_dim)
G1 = Generator(dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)
G2 = Generator(dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)
D1 = Discriminator(input_shape)
D2 = Discriminator(input_shape)

if cuda:
    E1 = E1.cuda()
    E2 = E2.cuda()
    G1 = G1.cuda()
    G2 = G2.cuda()
    D1 = D1.cuda()
    D2 = D2.cuda()
    criterion_GAN.cuda()
    criterion_pixel.cuda()

# Load pretrained models
E1.load_state_dict(torch.load("saved_models/%s/E1_%s.pth" % (opt.dataset_name, "final")))
E2.load_state_dict(torch.load("saved_models/%s/E2_%s.pth" % (opt.dataset_name, "final")))
G1.load_state_dict(torch.load("saved_models/%s/G1_%s.pth" % (opt.dataset_name, "final")))
G2.load_state_dict(torch.load("saved_models/%s/G2_%s.pth" % (opt.dataset_name, "final")))
D1.load_state_dict(torch.load("saved_models/%s/D1_%s.pth" % (opt.dataset_name, "final")))
D2.load_state_dict(torch.load("saved_models/%s/D2_%s.pth" % (opt.dataset_name, "final")))

# Loss weights
lambda_0 = 10  # GAN
lambda_1 = 0.1  # KL (encoded images)
lambda_2 = 100  # ID pixel-wise
lambda_3 = 0.1  # KL (encoded translated images)
lambda_4 = 100  # Cycle pixel-wise

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(E1.parameters(), E2.parameters(), G1.parameters(), G2.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2),
)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = [
    #transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    #transforms.RandomCrop((opt.img_height, opt.img_width)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
]

# Test data loader
val_dataloader = DataLoader(
    ImageDataset("data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=False, mode="test"),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)

# ----------
#  Testing
# ----------
labelsA = []
labelsB = []
realA = np.zeros((200,256,256,1))
fakeA = np.zeros((200,256,256,1))
realB = np.zeros((200,256,256,1))
fakeB = np.zeros((200,256,256,1))

for i, batch in enumerate(val_dataloader):
  X1 = Variable(batch["A"].type(Tensor))
  X2 = Variable(batch["B"].type(Tensor))
  _, Z1 = E1(X1)
  _, Z2 = E2(X2)
  fake_X1 = G1(Z2)
  fake_X2 = G2(Z1)
  #img_sample = torch.cat((X1.data, fake_X2.data, X2.data, fake_X1.data), 0)
  #save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)

  label = batch["label_A"][0].split('/')[-1]
  labelsA.append(label.split('.')[0])
  label = batch["label_B"][0].split('/')[-1]
  labelsB.append(label.split('.')[0])

  realA[i] = denorm(X1.permute((0,2,3,1))).cpu().detach().numpy()
  realB[i] = denorm(X2.permute((0,2,3,1))).cpu().detach().numpy()

  fakeA[i] = denorm(fake_X1.permute((0,2,3,1))).cpu().detach().numpy()
  fakeB[i] = denorm(fake_X2.permute((0,2,3,1))).cpu().detach().numpy()


confexp = "CCtoMLO_UNIT"
save_images("output/images", np.concatenate((realA, fakeB,realB),axis=2), "images_"+str(opt.run)+'_'+confexp, labelsB, max_number_images=20, resize_dimensions=None, invert = False, training = False)
np.savez_compressed('output/images/images_'+str(opt.run)+'_'+confexp, fakeB)
np.savez_compressed('output/images/labels_'+str(opt.run)+'_'+confexp, labelsB)
write_line_to_csv(
    'output/CSAW_',confexp + ".csv",
        {
            "RUN": str(opt.run),
            "MAE": mean_absolute_error(realB,fakeB),
            "PSNR": mean_psnr(realB,fakeB),
            "SSIM": mean_ssim(realB,fakeB)
        })

confexp = "MLOtoCC_UNIT"
save_images("output/images/", np.concatenate((realB, fakeA,realA),axis=2), "images_"+str(opt.run)+'_'+confexp, labelsA, max_number_images=20, resize_dimensions=None, invert = False, training = False)
np.savez_compressed('output/images/images_'+str(opt.run)+'_'+confexp, fakeA)
np.savez_compressed('output/images/labels_'+str(opt.run)+'_'+confexp, labelsA)

write_line_to_csv(
    'output/CSAW_',confexp + ".csv",
        {
            "RUN": str(opt.run),
            "MAE": mean_absolute_error(realA,fakeA),
            "PSNR": mean_psnr(realA,fakeA),
            "SSIM": mean_ssim(realA,fakeA)
        }) 
        
        