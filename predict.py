import cv2
import matplotlib.pyplot as plt
import numpy as np
from starter_code.visualize import visualize
from starter_code.utils import load_case
import sys
import os
import tensorflow as tf
from modules.visualizeSlider3Way import cube_show_slider
from modules.preprocessing import preprocess_X
from modules.NeuralNetwork import makeModel, dice_coef, dice_coef_loss

def visualize_case(case_num, save=None):
    def minMax(array):
        return (array - np.min(array))/(np.max(array) - np.min(array))

    volume, segmentation = load_case(case_num)
    X = preprocess_X(volume)
    y = segmentation.get_fdata()
    model = tf.keras.models.load_model(r'MODEL.h5', custom_objects={
        'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    

    segmentation_pred = model.predict(X)
    kidney = cv2.threshold(segmentation_pred[:,:,:,1],0.2,1,cv2.THRESH_BINARY)[1]
    cancer = cv2.threshold(segmentation_pred[:,:,:,2],0.06,1,cv2.THRESH_BINARY)[1]
    seg_img = np.clip((kidney*0.5 + cancer), 0, 1)
    
    cube_show_slider(cube=X[:, :, :, 0], cube1=y,
                     cube2=seg_img, axis=0, cmap='gray')
    if save:
        nifty_img = nib.Nifti1Image(
            segmentation_pred, volume.affine, volume.header)
        nib.save(nifty_img, str(save) + "\case{}".format(case_num))


if __name__ == "__main__":
    case_num_arg = int(sys.argv[1])
    visualize_case(case_num_arg)