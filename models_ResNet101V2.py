#!/usr/bin/env python

# In[ ]:
# %cd /home/ubuntu/files/cs230

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import matplotlib.pyplot as plt

import datasets as ds

# In[ ]:
# to enable GPUs
#GPUs = tf.config.list_physical_devices('GPU')
#print("Number of GPUs Available: ", len(GPUs))
#tf.config.experimental.set_memory_growth(GPUs[0], True)

#AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SHAPE = ds.IMG_SIZE + (3,)


# image standardization
# used for MobileNetV2 models
def standardize_image(image, label):
    return tf.keras.applications.resnet.preprocess_input(image), label


# create model ResNet101_1a
# regular ResNet101 trained on imagenet, with just last layers replace to be single sigmoid out
def get_model_ResNet101_1a():
    #start with base ResNet101
    base_model = tf.keras.applications.ResNet101(
        input_shape=IMG_SHAPE,
        include_top=False,
        alpha=1.0,
        weights="imagenet",
        input_tensor=None
    )

    base_model.trainable = False
    # tune which layers are adjusted during training
    #base_modelodel.trainable = True
    #fine_tune_at = 120
    #for layer in base_model.layers[:fine_tune_at]:
    #    layer.trainable = False

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)      # using .2 dropout for this model
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    return model


# create model ResNet101_1b
# regular ResNet101 trained on imagenet, with layers above 120 trainable
# last layers replace to be single sigmoid out
def get_model_ResNet101_1b():
    #start with base ResNet101
    base_model = tf.keras.applications.ResNet101(
        input_shape=IMG_SHAPE,
        include_top=False,
        alpha=1.0,
        weights="imagenet",
        input_tensor=None
    )

    # tune which layers are adjusted during training
    base_model.trainable = True
    fine_tune_at = 120
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)      # using .2 dropout for this model
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    return model

