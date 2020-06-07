import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from starter_code.visualize import visualize
from starter_code.utils import load_case
from tqdm import tqdm
import tensorflow as tf

volume, segmentation = load_case(123)
X = volume.get_data()
X = np.expand_dims(X, 3)
model = tf.keras.models.load_model('model_big_1')


segmentation_pred = model.predict(X)
nifty = nib.Nifti1Image(segmentation_pred, volume.affine, volume.header)

from visualizeSlider import *
cube_show_slider(cube=segmentation_pred[:,:,:,0], axis=0, cmap='gray')
cube_show_slider(cube=segmentation_pred[:,:,:,1], axis=0, cmap='gray')
cube_show_slider(cube=segmentation.get_data(), axis=0, cmap='gray')