#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import datasets as ds
IMG_SHAPE = ds.IMG_SIZE + (3,)


# image standardization
# used for our custom models
def standardize_image(image, label):
    #return tf.cast(image, tf.float32) / 255.0, label
    return tf.image.per_image_standardization(image), label


# create model
# 3 Conv layers, 2 FC layers
def get_model_Custom_1a():

    model = keras.Sequential(
        [
            keras.Input(shape=(224,224,3)),
            layers.Conv2D(32, (3,3), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(64, (3,3), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(128, (3,3), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Flatten(),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid'),

        ]
    )
    model.name = "Custom_1a"

    return model


# create model
# 6 Conv layers, 3 FC layers
def get_model_Custom_2a():

    model = keras.Sequential(
        [
            keras.Input(shape=(224,224,3)),
            layers.Conv2D(32, (3,3), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(48, (3,3), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(48, (3,3), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(64, (3,3), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(64, (3,3), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(128, (3,3), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid'),

        ]
    )
    model.name = "Custom_2a"

    return model


# create model
# 6 Conv layers, 3 FC layers, larger filters
def get_model_Custom_2b():

    model = keras.Sequential(
        [
            keras.Input(shape=(224,224,3)),
            layers.Conv2D(32, (5,5), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(48, (5,5), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(48, (3,3), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(64, (5,5), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(64, (3,3), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(128, (3,3), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid'),

        ]
    )
    model.name = "Custom_2b"

    return model

