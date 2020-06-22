import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

# def scale(array, maxv=1):
#     return ((array - K.min(array))/(K.max(array) - K.min(array)))*maxv
def scale(array):
    return (array-np.min(array))/4095

def edge_detect(img):
    X_edges = []
    for i in range(len(img[:,0,0,0])):        
        X_edges.append(
            cv2.medianBlur(         
                cv2.bitwise_not(
                    cv2.adaptiveThreshold(
                        scale(img[i,:,:,0], 255).astype(np.uint8),
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        15, 2
                    )
                )
            ,9)
        )
    return X_edges

def find_segmentation(y):
    y_slices=len(y[:,0,0])-1
    for slice in range(y_slices):
        if np.sum(y[slice,:,:])>0:
            begining_slice=slice
            break
    for slice in range(y_slices):
        if np.sum(y[(y_slices-slice),:,:])>0:
            ending_slice=y_slices-slice
            break
    return begining_slice, ending_slice

#CURRENTLY WINDOWING AND EDGE DETECTION TURNED OFF
def preprocess_X(volume):
    #load as array
    X = volume.get_fdata()
    X = scale(X)
    # X = tf.cast(X, dtype=tf.float16)
    X = np.expand_dims(X, 3)
    if not X.shape[1] == 512 or not X.shape[2] == 512:
        X = tf.image.resize(X, size=(512, 512))
    
    # X_window = scale(X_window)
    # edge detection 
    # CURRENTLY TURNED OFF    
    # X_edges = edge_detect(X)
    # X_edges = scale(X_edges, 1)
    # X_edges = np.expand_dims(X_edges, 3)

    # concat contrast with edges
    # X = np.concatenate((X_window, X_edges), axis=3).astype(np.float32)
    return X

def preprocess_y(segmentation):    
    y = segmentation.get_fdata()
    begining, end = find_segmentation(y)
    # y = tf.cast(y, dtype=tf.float16)
    y = tf.keras.utils.to_categorical(y)
    if not y.shape[1] == 512 or not y.shape[2] == 512:
        y = tf.image.resize(y, size=(512, 512))
    
    return y, begining, end