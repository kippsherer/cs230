#!/usr/bin/env python

# In[ ]:

# %cd /home/ubuntu/files/cs230

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

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
DROPOUT_PROP = 0.2
IMG_SHAPE = ds.IMG_SIZE + (3,)

base_model = tf.keras.applications.ResNet152V2(
    input_shape=IMG_SHAPE,
    include_top=False,
    alpha=1.0,
    weights="imagenet",
    input_tensor=None
)


#model.trainable = True
#fine_tune_at = 120
#for layer in base_model.layers[:fine_tune_at]:
#    layer.trainable = False


base_model.trainable = False
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(DROPOUT_PROP)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)


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


acc = [0.] + history.history['accuracy']
loss = history.history['loss']

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
#plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
#plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training Loss')
plt.xlabel('epoch')
plt.show()



