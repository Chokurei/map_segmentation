#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 20:50:31 2018

@author: kaku
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import cv2, glob, os

from keras.utils.np_utils import to_categorical 
from keras.models import Model
from keras.layers import Input, merge, MaxPooling2D, UpSampling2D, Add, concatenate
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.models import model_from_json
from keras import backend as K

from sklearn.model_selection import train_test_split

smooth = 1e-12

# big image shape = 3456 x 3456
# proposed patch size = 128 x 128
img = tiff.imread(img_path[0])[row_range[0]:row_range[-1],col_range[0]:col_range[-1],:]
label = plt.imread(label_path[0])[row_range[0]:row_range[-1],col_range[0]:col_range[-1]]

tiff.imsave('/Users/kaku/Desktop/r_img.tif', img)
tiff.imsave('/Users/kaku/Desktop/a_img.tif', label)

def image_get(img, patch_size):
    stride = patch_size//2
    img_num = (img.shape[0] // stride - 1) * (img.shape[1] // stride - 1)
    if np.ndim(img) == 3:
        imgs = np.zeros((img_num, patch_size, patch_size, 3))
        for i in range(img.shape[0] // stride - 1):
            for j in range(img.shape[1] // stride - 1):
                img_patch = img[i*stride:(i*stride + patch_size), j*stride:(j*stride + patch_size),:] 
                imgs[i * (img.shape[0] // stride - 1) + j] = img_patch
    else:
        imgs = np.zeros((img_num, patch_size, patch_size))
        for i in range(img.shape[0] // stride - 1):
            for j in range(img.shape[1] // stride - 1):
                img_patch = img[i*stride:(i*stride + patch_size), j*stride:(j*stride + patch_size)] 
                imgs[i * (img.shape[0] // stride - 1) + j] = img_patch
    return imgs

def get_unet(patch_size, num_class):
    """
    Build a mini U-Net architecture
    Return U-Net model

    Notes
    -----
    Shape of output image is similar with input image
    Output img bands: N_Cls
    Upsampling is important
#    """
    ISZ = patch_size
    N_Cls = num_class
    inputs = Input((ISZ, ISZ, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

#    keras.layers.concatenate(inputs, axis=-1)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(N_Cls, (1, 1), activation='softmax')(conv9)
    
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    return model

def jaccard_coef(y_true, y_pred):

    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

data_path = '../Data/Shibuya/'
img_path = glob.glob(os.path.join(data_path, 'r*.tif'))
label_path = glob.glob(os.path.join(data_path, 'a*.tif'))
row_range = [26, -26]
col_range = [649, -749]

PATCH_SIZE = 128
N_CLASS = 12
CV_RATIO = 0.2
BATCH_SIZE = 32
EPOCH = 2

imgs = image_get(img, PATCH_SIZE)
labels = image_get(label, PATCH_SIZE)
labels = to_categorical(labels, 12)

X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.4, random_state=0)

model = get_unet(PATCH_SIZE, N_CLASS)

History = model.fit(x = X_train, y = y_train, batch_size=BATCH_SIZE, nb_epoch = EPOCH, verbose=1, validation_split=CV_RATIO)
        









