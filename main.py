# importing stuff
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from starter_code.visualize import visualize
from starter_code.utils import load_case

import tensorflow as tf

# internal library imports


# Load data
volume, segmentation = load_case(123)
img = volume.get_data()
# from visualizeSlider import *
# cube_show_slider(volume.get_fdata(), axis=0, cmap='gray')

IMG_WIDTH = img.shape[2]
IMG_HEIGHT = img.shape[1]
IMG_CHANNELS = img.shape[0]
print(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

#Build the model
from neuralNetwork import *
model = makeModel(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()







