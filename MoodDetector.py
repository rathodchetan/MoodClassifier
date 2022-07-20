#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)


# In[ ]:


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


# In[ ]:


index = 1
plt.imshow(X_train_orig[index]) #display sample training image
plt.show()


# In[ ]:


def happyModel():
    
    model = tf.keras.Sequential([
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            tfl.ZeroPadding2D(padding=(3,3),input_shape=(64,64,3)),
            ## Conv2D with 32 7x7 filters and stride of 1
            tfl.Conv2D(32,(7,7),strides=(1,1)),
            ## BatchNormalization for axis 3
            tfl.BatchNormalization(axis=3),
            ## ReLU
            tfl.ReLU(),
            ## Max Pooling 2D with default parameters
            tfl.MaxPool2D(),
            ## Flatten layer
            tfl.Flatten(),
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            tfl.Dense(1,activation='sigmoid')
        ])
    
    return model


# In[ ]:


happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])


# In[ ]:


happy_model.summary()


# In[ ]:


happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)


# In[ ]:


happy_model.evaluate(X_test, Y_test)


# In[ ]:





# In[ ]:




