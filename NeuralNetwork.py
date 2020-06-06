import cv2
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from starter_code.visualize import visualize
from starter_code.utils import load_case


def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

volume, segmentation = load_case("case_00123")
plt.imshow(volume.dataobj[105,:,:], cmap='gray')
plt.show()

img_raw = np.array(volume.dataobj)
print(int(np.floor(img_raw.shape[0]/4)))

img = np.resize(img_raw, (
    int(np.floor(img_raw.shape[0]/16)),
    int(np.floor(img_raw.shape[1]/16)),
    int(np.floor(img_raw.shape[2]/16))
))

import plotly.graph_objects as go
import numpy as np







