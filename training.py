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
from modules.dataGenerators import trainGenerator, validationGenerator

import tensorflow.keras.backend as K
# K.set_floatx('float16')
# K.set_epsilon(1e-3)

# -------------CONFIG-----------------------------------------------------
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])


model = makeModel(512, 512, 1)
# model = tf.keras.models.load_model(r'saved_models\modelunetwithoutBNscal2loss160epochs1592768773_1592770013.8964128.h5', custom_objects={
        # 'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

'------0-179 Train set, 180-209 val_set, 210-299 test'
'------AMOUNT OF SLICES FOR TRAIN DATA == 38650'
'------AMOUNT OF SLICES FOR VALID DATA == 4520'
'------AMOUNT OF SLICES FOR TRAIN DATA WITH KIDNEY PRESENT == 12009'
'------AMOUNT OF SLICES FOR VALID DATA WITH KIDNEY PRESENT == 3547'

NAME = "unetADAMdiceSLICED_{}".format(int(time.time()))

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='logs\\{}'.format(NAME)),
    tf.keras.callbacks.ModelCheckpoint(filepath='saved_models\\checkpoints\\{}.h5'.format(NAME), monitor='val_loss', verbose=1,
        save_best_only=True, mode='auto')
    ]

# '------lists data points for train and validation'
# deleting broken cases
case_numbers = np.delete(np.arange(0, 209, 1), [158, 159, 170, 202])
case_numbers_val = case_numbers[149:]
case_numbers_train = case_numbers[:149]

batch_size = 8
data_size = 12009
val_data_size = 3547
num_epochs = 24
results = model.fit(
    trainGenerator(case_numbers_train, batch_size),
    steps_per_epoch = data_size // batch_size,
    epochs = num_epochs,
    callbacks = callbacks,
    verbose = 1,
    validation_data = validationGenerator(case_numbers_val, batch_size),
    validation_steps = val_data_size // batch_size
    )

model.save('saved_models\\model{}_{}.h5'.format(NAME, time.time()))