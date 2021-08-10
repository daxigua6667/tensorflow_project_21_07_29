import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow import keras

print(tf.version.VERSION)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
show_dataset = tfds.list_builders()

print(show_dataset)

dataset= tfds.load("lsun")



# tensorboard --logdir logs/fit