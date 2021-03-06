# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gR7Tl41CNk-1FaJkCQCUZ09o4ST4vNR9
"""

import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow import keras

pip install tensorflow.io

show_dataset = tfds.list_builders()
print(show_dataset)

dataset_train, dataset_info = tfds.load("beans", batch_size=-1, with_info=True, as_supervised=True, split="train")
dataset_test, dataset_info = tfds.load("beans", batch_size=-1, with_info=True, as_supervised=True, split="test")
dataset_validation, dataset_info = tfds.load("beans", batch_size=-1, with_info=True, as_supervised=True, split="validation")

dataset_info

dataset_train = tfds.as_numpy(dataset_train)
dataset_test = tfds.as_numpy(dataset_test)
dataset_validation = tfds.as_numpy(dataset_validation)

train_image = dataset_train[0][:]
train_label = dataset_train[1][:]
train_image[1033].shape

test_image = dataset_test[0][:]
test_label = dataset_test[1][:]
test_image[127].shape

validation_image = dataset_validation[0][:]
validation_label = dataset_validation[1][:]
validation_image[132].shape

plt.imshow(train_image[0].astype(np.float32)/255.0, cmap=plt.get_cmap())
plt.show()
train_label[0]

plt.imshow(test_image[60].astype(np.float32)/255.0, cmap=plt.get_cmap())
plt.show()
test_label[60]

plt.imshow(validation_image[100].astype(np.float32)/255.0, cmap=plt.get_cmap())
plt.show()
validation_label[100]

validation_label_one_hot = tf.one_hot(validation_label, 3)

validation_label_one_hot[1]

