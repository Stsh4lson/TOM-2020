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
X = volume.get_data()
y = segmentation.get_data()
# from visualizeSlider import *
# cube_show_slider(volume.get_fdata(), axis=0, cmap='gray')

IMG_WIDTH = X.shape[2]
IMG_HEIGHT = X.shape[1]
IMG_SLICES = X.shape[0]
X = np.expand_dims(X, 3)
#Build the model
from neuralNetwork import *
model = makeModel(IMG_WIDTH, IMG_HEIGHT, 1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#ModelCheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model1.h5', verbose=1, save_best_only=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]
results = model.fit(X, y, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks, verbose=1)






