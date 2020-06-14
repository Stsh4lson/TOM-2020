import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def scale(array, maxv=1):
    return ((array - np.min(array))/(np.max(array) - np.min(array)))*maxv

def auto_canny(image, sigma=0.1):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def preprocess_X(volume):
    #load as array
    X = volume.get_fdata()
    X = scale(X, 4096)
    #dealling with case 160
    if not X.shape[1]==512 or not X.shape[2]==512:
        X_placeholder = np.zeros((X.shape[0], 512, 512))
        for i in range(X.shape[0]):            
            X_placeholder[i,:,:] = cv2.resize(X[i,:,:], (512, 512))
        X = np.expand_dims(X_placeholder, 3)
        # window kidney
        X_window = np.clip(X, 1500, 4096)
    else:
        X = np.expand_dims(X, 3)
        # window kidney
        X_window = np.clip(X, 1500, 4096)

    # edge detection
    X_edges = []
    for i in range(len(X[:,0,0,0])):
        X_edges.append( auto_canny(
            cv2.GaussianBlur(
                scale(X_window[i,:,:,0],255), (15, 15), 0).astype(np.uint8))
            )
    X_edges = np.array(X_edges)
    X_edges = cv2.morphologyEx(X_edges, cv2.MORPH_CLOSE, (10, 10))
    X_edges = np.expand_dims(X_edges, 3)

    # concat contrast with edges
    X = np.concatenate((X_window, X_edges), axis=3).astype(np.float32)
    return X

def preprocess_y(segmentation):
    y = segmentation.get_fdata()
    if not y.shape[1]==512 or not y.shape[2]==512:
        y_placeholder = np.zeros((y.shape[0], 512, 512))
        for i in range(y.shape[0]):            
            y_placeholder[i,:,:] = cv2.resize(y[i,:,:], (512, 512))
        y = y_placeholder
    y = tf.keras.utils.to_categorical(y)[:,:,:,1:]
    
    return y