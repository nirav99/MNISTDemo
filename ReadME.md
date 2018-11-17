MNISTDemo
=========

Classify hand-written digits from MNIST dataset using Tensorflow

'''python
import tensorflow as tf
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
m = x_train.shape[0] # m is the total number of samples

print("m = " + str(m))

print("Shape of training data x_train: " + str(x_train.shape) + " y_train : " + str(y_train.shape))
print("Shape of test data : " + str(x_test.shape))
'''

Print some random digits

'''python
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

digit_index = 729
some_digit = x_train[digit_index]

plt.imshow(some_digit, cmap = matplotlib.cm.binary, interpolation="nearest")
#plt.axis("off")
plt.show()

print("Y Label : " + str(y_train[digit_index]))
'''

![png](digit4.png)


