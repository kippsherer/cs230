#!/usr/bin/env python

import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory

DATASET_DIRECTORY = '/home/ubuntu/files/data/'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SEED = 42

dataset_cache = {'train':False, 'dev':False, 'test':False}

# returns (images, labels), where 
# images has shape (batch_size, image_size[0], image_size[1], num_channels), and 
# labels are a float32 tensor of 1s and 0s of shape (batch_size, 1)
def get_dataset_train():
    if dataset_cache['train'] == False:
        dataset_cache['train'] = image_dataset_from_directory(DATASET_DIRECTORY+'train',
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             label_mode='binary',
                                             seed=SEED)

    return dataset_cache['train']


# returns (images, labels), where 
# images has shape (batch_size, image_size[0], image_size[1], num_channels), and 
# labels are a float32 tensor of 1s and 0s of shape (batch_size, 1)
def get_dataset_dev():
    if dataset_cache['dev'] == False:
        dataset_cache['dev'] = image_dataset_from_directory(DATASET_DIRECTORY+'dev',
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             label_mode='binary',
                                             seed=SEED)

    return dataset_cache['dev']


# returns (images, labels), where 
# images has shape (batch_size, image_size[0], image_size[1], num_channels), and 
# labels are a float32 tensor of 1s and 0s of shape (batch_size, 1)
def get_dataset_test():
    if dataset_cache['test'] == False:
        dataset_cache['test'] = image_dataset_from_directory(DATASET_DIRECTORY+'test',
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             label_mode='binary',
                                             seed=SEED)

    return dataset_cache['test']


def show_dataset_statistics(dataset):
    print ('Class names:')
    print (dataset.class_names)




def show_dataset_samples(dataset, cnt=1):
    print ('Here')

#plt.figure(figsize=(32, 32))
#for images, labels in test_dataset.take(1):
#    for i in range(9):
#        ax = plt.subplot(3, 3, i + 1)
#        plt.imshow(images[i].numpy().astype("uint8"))
#        plt.title(class_names[labels[i]])
#        plt.axis("off")
        




# image standardization
def standardize_image(image, label):
    #return tf.image.per_image_standardization(image), label
    return tf.cast(image, tf.float32) / 255.0, label