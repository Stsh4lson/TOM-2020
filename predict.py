import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from starter_code.visualize import visualize
from starter_code.utils import load_case
from tqdm import tqdm
import os
import tensorflow as tf
from modules.visualizeSlider3Way import cube_show_slider
from modules.preprocessing import preprocess_X


def visualize_case(case_num, save=None):

    volume, segmentation = load_case(case_num)
    X = preprocess_X(volume)
    y = segmentation.get_fdata()
    model = tf.keras.models.load_model('saved_models\model_big_4_test')

    print(X.shape)
    print(y.shape)

    segmentation_pred = model.predict(X)
    print('mean of kidneys ', np.mean(segmentation_pred[:,:,:,0]))
    print('mean of cancer ', np.mean(segmentation_pred[:,:,:,1]))

    # (T, segmentation_pred_cancer) = cv2.threshold(
    #     segmentation_pred[:, :, :, 1], 0.004, 2, cv2.THRESH_BINARY)
    # (T, segmentation_pred_kidney) = cv2.threshold(
    #     segmentation_pred[:, :, :, 0], 0.2, 1, cv2.THRESH_BINARY)
    # segmentation_pred = np.clip(
    #     (segmentation_pred_kidney+segmentation_pred_cancer), 0, 2)

    cube_show_slider(cube=X[:, :, :, 0], cube1=y,
                     cube2=segmentation_pred[:,:,:,0], axis=0, cmap='gray')
    if save:
        nifty_img = nib.Nifti1Image(
            segmentation_pred, volume.affine, volume.header)
        nib.save(nifty_img, str(save) + "\case{}".format(case_num))
visualize_case(123)