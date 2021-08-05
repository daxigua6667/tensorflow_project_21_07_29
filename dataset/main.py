
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
print(tf.version.VERSION)
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import  layers, models
import matplotlib.pyplot as plt
from tensorflow import keras

show_dataset = tfds.list_builders()

# dataset = tfds.load("coil100", split="train", data_dir="C:/Users/daxigua/tensorflow_datasets/", download=False)
dataset, dataset_info = tfds.load("coil100", split="train",with_info=True, as_supervised=True)
# dataset_test = tfds.load("coil100", split="test")
# dataset_test[:,:] = dataset[0: 30]


    # dataset = tfds.load(name="coil100", split="train", data_dir="C:/Users/daxigua/tensorflow_datasets/", download=False)
assert isinstance(dataset, tf.data.Dataset)
print(dataset)



# for dataset_example in dataset.take(1):  # 只取一个样本
#     image, label = dataset_example["image"], dataset_example["label"]
#
#     plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
#     plt.show()
#     print("Label: %d" % label.numpy())

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

dataset = dataset.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.cache()
# dataset = dataset.shuffle(dataset_info.splits['train'].num_examples)
dataset = dataset.batch(128)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

print(dataset)


# dataset_test_image_normalizaion,  dataset_test_lable_normalizaion= dataset_test["image"]/255.0, dataset["label"]

dataset_builder = tfds.builder("coil100")
dataset_builder.download_and_prepare()
# dataset_train = dataset_builder.as_dataset(split="train")

dataset_train = dataset

# dataset_train = dataset_train.repeat().shuffle(1024).batch(32)

# prefetch 将使输入流水线可以在模型训练时异步获取批处理。
# dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)

# 现在你可以遍历数据集的批次并在 mnist_train 中训练批次：
#   ...

info = dataset_builder.info
print("-----------------------------------------------")
print(info)
print("-----------------------------------------------")

print("-----------------------------------------------")
print("features", info.features)
print("-----------------------------------------------")
print("num_classes", info.features["label"].num_classes)
print("-----------------------------------------------")
print("names", info.features["label"].names)
print("-----------------------------------------------")


dataset_test = tfds.load("coil100", split="train")
fig = tfds.show_examples(info, dataset_test)

model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((3, 3)))
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))


model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(dataset, epochs=300)

# test_loss, test_acc = model.evaluate(train_image, train_label, verbose=2)
#
# print('\nTest accuracy:', test_acc)
#
# probability_model = tf.keras.Sequential([model,
#                                          tf.keras.layers.Softmax()])
#
# # test_images = train_image[13]
# # test_label = train_label[13]
#
# predictions = probability_model.predict(test_images)
#
# np.argmax(predictions[0])
#
# test_label[0]