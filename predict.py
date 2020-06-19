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
from modules.NeuralNetwork import makeModel, dice_coef, dice_coef_loss


def visualize_case(case_num, save=None):

    volume, segmentation = load_case(case_num)
    X = preprocess_X(volume)
    y = segmentation.get_fdata()
    model = tf.keras.models.load_model(r'saved_models\checkpoints\model8\unet_without_edge_detection_with_batch_norm_1592592855', custom_objects={
        'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    

    segmentation_pred = model.predict(X)
    # print('mean of kidneys ', np.mean(segmentation_pred[:,:,:,0]))
    # print('mean of cancer ', np.mean(segmentation_pred[:,:,:,1]))
  
    # (T, segmentation_pred_cancer) = cv2.threshold(
    #     segmentation_pred[:, :, :, 1], 0.001, 2, cv2.THRESH_BINARY)
    # (T, segmentation_pred_kidney) = cv2.threshold(
    #     segmentation_pred[:, :, :, 0], 0.1, 1, cv2.THRESH_BINARY)
    # segmentation_pred = np.clip(
    #     (segmentation_pred_kidney+segmentation_pred_cancer), 0, 2)
    print(X.shape)
    print(y.shape)
    print(segmentation_pred.shape)
    print(np.min(segmentation_pred))
    print(np.max(segmentation_pred))
    print(np.mean(segmentation_pred)    )
    cube_show_slider(cube=X[:, :, :, 0], cube1=y,
                     cube2=segmentation_pred[:,:,:,1], axis=0, cmap='gray')
    cube_show_slider(cube=segmentation_pred[:,:,:,0], cube1=segmentation_pred[:,:,:,1],
                     cube2=segmentation_pred[:,:,:,2], axis=0, cmap='gray')
    if save:
        nifty_img = nib.Nifti1Image(
            segmentation_pred, volume.affine, volume.header)
        nib.save(nifty_img, str(save) + "\case{}".format(case_num))
visualize_case(123)