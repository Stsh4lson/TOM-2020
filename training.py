import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import time

# ---------kits19 modules--------------------------------------------------
from starter_code.visualize import visualize
from starter_code.utils import load_case

# ----------own modules---------------------------------------------------
from modules.NeuralNetwork import makeModel, dice_coef, dice_coef_loss


from modules.visualizeSlider import cube_show_slider

# ----------REDUNDANT - USED ONLY IN DATA GENERATORS'
from modules.dataGenerators import trainGenerator, validationGenerator

import keras.backend as K
# K.set_floatx('float16')
# K.set_epsilon(1e-4)
# -------------CONFIG-----------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])

model = makeModel(512, 512, 1)
# model = tf.keras.models.load_model('saved_models/checkpoints/unet_without_edge_detection_with_batch_norm_1592345709', custom_objects={
#     'soft_dice_loss': soft_dice_loss})

'------0-179 Train set, 180-209 val_set, 210-299 test'
'------AMOUNT OF SLICES FOR TRAIN DATA == 38650'
'------AMOUNT OF SLICES FOR VALID DATA == 4520'

NAME = "unet_without_edge_detection_with_batch_norm_{}".format(int(time.time()))

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME)),
    tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/checkpoints/model8/{}'.format(NAME), monitor='val_loss', verbose=1,
        save_best_only=True, mode='auto')
    ]

# '------lists data points for train and validation'
# deleting broken cases
case_numbers = np.delete(np.arange(0, 209, 1), [158, 159, 170, 202])
case_numbers_val = case_numbers[179:]
case_numbers_train = case_numbers[:179]

batch_size = 8
data_size = 38650
val_data_size = 4520
results = model.fit(
    trainGenerator(case_numbers_train, batch_size),
    steps_per_epoch = data_size // batch_size,
    epochs = 12,
    callbacks = callbacks,
    verbose = 1,
    validation_data = validationGenerator(case_numbers_val, batch_size),
    validation_steps = val_data_size // batch_size
    )

model.save('saved_models\model8_from_{}'.format(time.time()))