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
GPUs = tf.config.list_physical_devices('GPU')
print("Number of GPUs Available: ", len(GPUs))
tf.config.experimental.set_memory_growth(GPUs[0], True)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# In[ ]:
# retrieve our training dataset
ds_train = ds.get_dataset_train()
ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)
ds_train_std = ds_train.map(ds.standardize_image)


# In[ ]:
# create model
# 3 Conv layers, 2 FC layers
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


# In[ ]:
# compile the model
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
            tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives() ],
)

# In[ ]:
# train the model
history = model.fit(ds_train_std, epochs=10, verbose=2)
print (history.history)


# In[ ]:
# evaluate the model with the dev dataset
statistics = ['binary_crossentropy', 'accuracy', 'precision', 'recall', 'f1_score', 'false_negatives', 'false_positives']
print ("Dev dataset")
ds_dev = ds.get_dataset_dev()
ds_dev_std = ds_dev.map(ds.standardize_image)
result = model.evaluate(ds_dev_std, verbose=2)
print ( dict(zip(statistics, result)) )


# In[ ]:
# evaluate the model with the test dataset
print ("Test dataset")
ds_test = ds.get_dataset_test()
ds_test_std = ds_test.map(ds.standardize_image)
result = model.evaluate(ds_test_std, verbose=2)
print ( dict(zip(statistics, result)) )


# In[ ]:
# evaluate the model with the test dataset
print ("Test Dark dataset")
ds_test_dark = ds.get_dataset_test_dark()
ds_test_dark_std = ds_test_dark.map(ds.standardize_image)
result = model.evaluate(ds_test_dark_std, verbose=2)
print ( dict(zip(statistics, result)) )



