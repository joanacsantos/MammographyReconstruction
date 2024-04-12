# Read DICOM files from the cbis-dataset

import os
import glob
thisdir = *** Define your directory ***

names = []
names_complete = []
for name in os.listdir(thisdir):
    name_1 = name.split('\\')[-1]
    names_complete.append(name_1)
    name_2 = '_'.join(name_1.split('_')[0:4])
    names.append(name_2)
    
names = list(set(names))
print(len(names))

#Eliminate duplicates
pares = []
pares_complete = []
patients = []
for k in names:
    if((k+'_MLO' in names_complete) & (k+'_CC' in names_complete)):
        if(k.split('_')[2] not in patients):
            pares.append(k)
            pares_complete.append(k+'_MLO')
            pares_complete.append(k+'_CC')
            patients.append(k.split('_')[2])

print(len(pares))

import matplotlib.pyplot as plt
import pydicom
import numpy as np
import matplotlib
import cv2

images = np.zeros((len(pares)*2, 256,256,1))
labels = []
count = 0
for k in pares:
    print(k)
    mlo_dir = thisdir+'/'+k+'_MLO'+'/'
    cc_dir = thisdir+'/'+k+'_CC'+'/'
    labels.append(k+'_MLO')
    labels.append(k+'_CC')

    res = []
    for (dir_path, dir_names, file_names) in os.walk(mlo_dir):
        res.append(dir_path)
    dataset = pydicom.dcmread(res[2] + '/1-1.dcm')
    d_mlo = np.array(dataset.pixel_array)

    res = []
    for (dir_path, dir_names, file_names) in os.walk(cc_dir):
        res.append(dir_path)
    dataset2 = pydicom.dcmread(res[2] + '/1-1.dcm')
    d_cc = np.array(dataset2.pixel_array)

    resized_mlo = cv2.resize(d_mlo, (256,256), interpolation = cv2.INTER_AREA)
    resized_cc = cv2.resize(d_cc, (256,256), interpolation = cv2.INTER_AREA)

    normalized_mlo = ((resized_mlo - np.amin(resized_mlo))/(np.amax(resized_mlo) - np.amin(resized_mlo)) * 255)
    normalized_cc = ((resized_cc - np.amin(resized_cc))/(np.amax(resized_cc) - np.amin(resized_cc)) * 255)

    images[count]= np.expand_dims(normalized_mlo, axis=(0,3))
    count = count + 1
    images[count]= np.expand_dims(normalized_cc, axis=(0,3))
    count = count + 1
images_original = images / 255

tt = 0

images_clean = np.zeros(images_original.shape)

for img in range(0,images_original.shape[0]):
    
    image = images_original[img]
    image = np.reshape(image,(256,256))
    image = (image*255).astype('uint8')
    print(image.shape)
    
    ret,thresh1 = cv2.threshold(np.array(image), 20, 255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels_im = cv2.connectedComponents(opening)
    values = []
    for value in range(1,num_labels):
        values.append(len(np.where(labels_im==value)[0]))
        
    final_mask = labels_im == np.where(values == np.amax(values))[0][0] +1
    
    print(final_mask.shape)

    image = image*final_mask
    print(image.shape)

    image = (image - np.min(image))/np.max(image)

    print(np.min(image))
    print(np.max(image))


    tt = tt +1
    if(tt%100 == 0):
        print('Saving image: {} of 410'.format(tt))
    images_clean[img] = np.expand_dims(image, axis=(0,3))

np.save('CBIS_DDSM_image_pares_clean.npy', images_clean)
np.save('CBIS_DDSM_labels_pares.npy', labels)
