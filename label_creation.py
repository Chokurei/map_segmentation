#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 21:26:57 2018

create label image by provided layers

@author: kaku
"""

import numpy as np
import glob, os, cv2
import matplotlib.pyplot as plt
import tifffile as tiff

label_path = '../Data/Shibuya/a_shibuya/'
label_file_names = glob.glob(os.path.join(label_path, '*.tif'))

label_images = []
for i in range(len(label_file_names)):
    label_image_temp = tiff.imread(label_file_names[i])[:,:,0]
    label_image_temp[label_image_temp != 0] = i + 1
    label_images.append(label_image_temp)
    
label_image = np.clip(sum(label_images), 0, len(label_file_names) + 1)

tiff.imsave('../Data/Shibuya/a_shibuya.tif', label_image)


