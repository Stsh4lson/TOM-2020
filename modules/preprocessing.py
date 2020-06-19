import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def scale(array, maxv=1):
    return ((array - np.min(array))/(np.max(array) - np.min(array)))*maxv

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
            
#CURRENTLY WINDOWING AND EDGE DETECTION TURNED OFF
def preprocess_X(volume):
    #load as array
    X = volume.get_fdata()
    X = scale(X)
    #dealling with case 160
    if not X.shape[1]==512 or not X.shape[2]==512:
        X_placeholder = np.zeros((X.shape[0], 512, 512))
        for i in range(X.shape[0]):            
            X_placeholder[i,:,:] = cv2.resize(X[i,:,:], (512, 512))
        X = np.expand_dims(X_placeholder, 3)
        # window kidney
        # X_window = np.clip(X, 0.39, 0.9)
    else:
        X = np.expand_dims(X, 3)
        # window kidney
        # X_window = np.clip(X, 0.39, 0.9)

    # X_window = scale(X_window)
    # edge detection 
    # CURRENTLY TURNED OFF    
    # X_edges = edge_detect(X)
    # X_edges = scale(X_edges, 1)
    # X_edges = np.expand_dims(X_edges, 3)

    # concat contrast with edges
    # X = np.concatenate((X_window, X_edges), axis=3).astype(np.float32)
    return scale(X.astype(np.float32))

def preprocess_y(segmentation):
    y = segmentation.get_fdata()
    if not y.shape[1]==512 or not y.shape[2]==512:
        y_placeholder = np.zeros((y.shape[0], 512, 512))
        for i in range(y.shape[0]):            
            y_placeholder[i,:,:] = cv2.resize(y[i,:,:], (512, 512))
        y = y_placeholder
    y = tf.keras.utils.to_categorical(y)
    
    return y