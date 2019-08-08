import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

sr=misc.imread("/home/xulingchuan/chuan/SRCNN-Tensorflow/sample/test_image.png", flatten=True, mode="YCbCr").astype(np.float)
hr=misc.imread("/home/xulingchuan/chuan/SRCNN-Tensorflow/Test/Set5/bird_GT.bmp", flatten=True, mode="YCbCr").astype(np.float)
plt.imshow(sr)
plt.show()
