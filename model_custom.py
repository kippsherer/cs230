#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#import time
#import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import matplotlib.pyplot as plt
#from tensorflow.python.framework.ops import EagerTensor
#from tensorflow.python.ops.resource_variable_ops import ResourceVariable

import datasets as ds

# to enable GPUs
GPUs = tf.config.list_physical_devices('GPU')
print("Number of GPUs Available: ", len(GPUs))
tf.config.experimental.set_memory_growth(GPUs[0], True)


AUTOTUNE = tf.data.experimental.AUTOTUNE

# retrieve our datasets
ds_train = ds.get_dataset_train()
ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)
ds_train_std = ds_train.map(ds.standardize_image)

ds_dev = ds.get_dataset_dev()
ds_dev_std = ds_dev.map(ds.standardize_image)

#ds_test = ds.get_dataset_test()
#ds_test_std = ds_test.map(ds.standardize_image)


# iterating over dataset
#for epochs in range(10):
#    for x,y in ds_train
#        #training
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


print (model.summary())


model.compile(
    loss=keras.losses.BinaryCrossentropy(
            #from_logits=True,
            label_smoothing=0.0,
            axis=-1,
            reduction='sum_over_batch_size',
            name='binary_crossentropy'
        ),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.F1Score(), 
            tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.BinaryAccuracy(), 
            tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives() ],
)

model.fit(ds_train_std, epochs=10, verbose=2)

model.evaluate(ds_dev_std, verbose=2)



