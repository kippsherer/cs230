#!/usr/bin/env python

# In[ ]:

# %cd /home/ubuntu/files/cs230

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
import matplotlib.pyplot as plt

import datasets as ds
#import models_MobileNetV2 as mnv2
#import models_ResNet101 as rn
import models_Custom as cus


# to enable GPUs
GPUs = tf.config.list_physical_devices('GPU')
print("Number of GPUs Available: ", len(GPUs))
tf.config.experimental.set_memory_growth(GPUs[0], True)

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SHAPE = ds.IMG_SIZE + (3,)


# get the model we want to train, set the preprocessing function
#standardize_image = mnv2.standardize_image
#model = mnv2.get_model_MobileNetV2_1a()
#model = mnv2.get_model_MobileNetV2_1b()
#model = mnv2.get_model_MobileNetV2_1c()

#standardize_image = rn.standardize_image
#model = rn.get_model_ResNet101_1b()


standardize_image = cus.standardize_image
model = cus.get_model_Custom_1a()


# compile the model
model.compile(
    loss=keras.losses.BinaryCrossentropy(
            #from_logits=True,
            label_smoothing=0.0,
            axis=-1,
            reduction='sum_over_batch_size',
            name='binary_crossentropy'
        ),
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.F1Score(), 
            tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives() ],
)



# retrieve our training dataset
ds_train = ds.get_dataset_train()
ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)
ds_train_std = ds_train.map(standardize_image)


# standardize the dataset and train the model
history = model.fit(ds_train_std, epochs=7, verbose=2)
print (history.history)




# evaluate the model with the dev dataset
statistics = ['binary_crossentropy', 'accuracy', 'precision', 'recall', 'f1_score', 'false_negatives', 'false_positives']
print ("Dev dataset")
ds_dev = ds.get_dataset_dev()
ds_dev_std = ds_dev.map(standardize_image)
result = model.evaluate(ds_dev_std, verbose=2)
print ( dict(zip(statistics, result)) )



# evaluate the model with the test dataset
print ("Test dataset")
ds_test = ds.get_dataset_test()
ds_test_std = ds_test.map(standardize_image)
result = model.evaluate(ds_test_std, verbose=2)
print ( dict(zip(statistics, result)) )



# evaluate the model with the test dataset
print ("Test Dark dataset")
ds_test_dark = ds.get_dataset_test_dark()
ds_test_dark_std = ds_test_dark.map(standardize_image)
result = model.evaluate(ds_test_dark_std, verbose=2)
print ( dict(zip(statistics, result)) )





