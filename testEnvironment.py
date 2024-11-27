import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image_dataset_from_directory

GPUs = tf.config.list_physical_devices('GPU')
print("Number of GPUs Available: ", len(GPUs))
#if len(GPUs) > 0:
#    tf.config.experimental.set_memory_growth(GPUs[0], True)

# Check the images directory to make sure it's readable
IMAGE_DIRECTORY = '/home/ubuntu/files/data/frames_224'
files = os.listdir(IMAGE_DIRECTORY)


# Check some images to see if the data sets are working
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SEED = 42

#train_dataset = image_dataset_from_directory(IMAGE_DIRECTORY,
#                                             shuffle=True,
#                                             batch_size=BATCH_SIZE,
#                                             image_size=IMG_SIZE,
#                                             label_mode='binary',
#                                             validation_split=0.1,
#                                             subset='training',
#                                             seed=SEED)

# total non training images
validation_dataset = image_dataset_from_directory(IMAGE_DIRECTORY,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             label_mode='binary',
                                             validation_split=0.1,
                                             subset='validation',
                                             seed=SEED)

dev_dataset = validation_dataset.take(3000)
test_dataset = validation_dataset.skip(3000)

# check the class names
class_names = test_dataset.class_names
print (class_names)

plt.figure(figsize=(10, 10))
for images, labels in test_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
        




