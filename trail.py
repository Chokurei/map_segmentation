#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:39:59 2017

@author: kaku
"""

import cv2
import matplotlib.pyplot as plt
import tifffile as tiff

image_path = '/Users/kaku/Desktop/Work/map_segmentation/Data/UseZone_raster4326/533935762.tif' 
label_path = '/Users/kaku/Desktop/Work/map_segmentation/Data/UseZone_vector4326/533935762.tif'

image = tiff.imread(image_path)
label = tiff.imread(label_path)

