import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import random
import torch
import torchvision
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from PIL import Image
from network import Generator
import numpy as np
from PIL import Image, ImageOps
import glob
import os
import numpy as np
import cv2
import csv
from pathlib import Path
import random
from scipy.stats import bernoulli
from skimage.metrics import structural_similarity as ssim
import time

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

def mean_psnr(predicted_data,images_test):
    values = []
    for img_index in range(0, images_test.shape[0]):
      predicted_image = np.reshape(predicted_data[img_index], (256,256))
      test_image = np.reshape(images_test[img_index], (256,256))
      metric = psnr(predicted_image, test_image, MAX=1) 
      values.append(metric)
    return (sum(values) / len(values))   
  

parser = argparse.ArgumentParser(description='Easy Implementation of Pix2Pix')

"""
Task                RawSize     Batch Size      Epochs      EpochsInPaper       RandomCrop&Mirroring    #ofSamples      #ofSamples(Validation)
Map-Aerial          600*600     1               100         200                 Yes                     1096            1098
Edges-Bag           256*256     4               15          15                  No                      138567          200
Semantic-Photo      256*256     1               100         200                 Yes                     2975            500
"""

# Task
parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders train, val, etc)')
parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument('--run', type=int, default=0, help='test run')
parser.add_argument('--confexp', type=str, default='None', help='configuration of the experiment')

# Options
parser.add_argument('--no_resize_or_crop', action='store_true', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batchSize', type=int, default=1, help='test Batch size')

# misc
parser.add_argument('--model_path', type=str, default='./models')
parser.add_argument('--sample_path', type=str, default='./test_results')

##### Helper Functions for Data Loading & Pre-processing
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

##### Helper Functions for Data Loading & Pre-processing
class ImageFolder(data.Dataset):
    def __init__(self, opt):
        # os.listdir Function gives all lists of directory
        self.root = opt.dataroot
        self.no_resize_or_crop = opt.no_resize_or_crop
        self.no_flip = opt.no_flip
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,),
                                                                  (0.5,))]) 
        self.dir_AB = os.path.join(opt.dataroot, 'test') 
        self.image_paths = list(map(lambda x: os.path.join(self.dir_AB, x), os.listdir(self.dir_AB)))

    def __getitem__(self, index):
        AB_path = self.image_paths[index]
        AB = Image.open(AB_path).convert('L')

        if(not self.no_resize_or_crop):
            AB = AB.resize((286 * 2, 286), Image.BICUBIC)
            AB = self.transform(AB)

            w = int(AB.size(2) / 2)
            h = AB.size(1)
            w_offset = random.randint(0, max(0, w - 256 - 1))
            h_offset = random.randint(0, max(0, h - 256 - 1))

            A = AB[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
            B = AB[:, h_offset:h_offset + 256, w + w_offset:w + w_offset + 256]
        else:
            AB = self.transform(AB)
            w_total = AB.size(2)
            w = int(w_total / 2)

            A = AB[:, :256, :256]
            B = AB[:, :256, w:w + 256]

        if (not self.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        return {'A': A, 'B': B,'label': AB_path}

    def __len__(self):
        return len(self.image_paths)

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

    g_path = os.path.join(args.model_path, 'generator.pkl')

    # Load pre-trained model
    generator = Generator(args.batchSize)
    generator.load_state_dict(torch.load(g_path))
    generator.eval()

    if torch.cuda.is_available():
        generator = generator.cuda()

    testing_labels = []
    testing_images = np.zeros((244,256,256,1))
    realA = np.zeros((244,256,256,1))
    realB = np.zeros((244,256,256,1))

    img = 0
    total_step = len(data_loader) # For Print Log
    for i, sample in enumerate(data_loader):
        AtoB = args.which_direction == 'AtoB'
        input_A = sample['A' if AtoB else 'B']
        input_B = sample['B' if AtoB else 'A']

        real_A = to_variable(input_A)
        fake_B = generator(real_A)
        real_B = to_variable(input_B)

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

    save_images("output/images/", np.concatenate((realA, testing_images,realB),axis=2), "images_"+str(args.run)+'_'+args.confexp, testing_labels, max_number_images=20, resize_dimensions=None, invert = False, training = False)
    np.savez_compressed('output/images/images_'+str(args.run)+'_'+args.confexp, testing_images)
    np.savez_compressed('output/images/labels_'+str(args.run)+'_'+args.confexp, testing_labels)

    write_line_to_csv(
        'CBIS_DDSM_'+args.confexp+'/results',args.confexp + ".csv",
              {
                  "RUN": (args.run + 1),
                  "MAE": mean_absolute_error(realB,testing_images),
                  "PSNR": mean_psnr(realB,testing_images),
                  "SSIM": mean_ssim(realB,testing_images)
               }) 


if __name__ == "__main__":
    main()