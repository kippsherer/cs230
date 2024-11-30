#!/usr/bin/env python

# In[ ]:

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import numpy as np
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
import matplotlib.pyplot as plt

# %cd /home/ubuntu/files/cs230

import performance as stats


# create training plots
plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.title('Training Plots')
plt.legend(loc='lower right')
plt.xlabel('epoch')
plt.ylabel('Accuracy')


for model in stats.model_statistics:
    #print (model)

    plt.plot(stats.model_statistics[model]['training']['accuracy'], label=model)


plt.ylim([min(plt.ylim()),1.01])
plt.show()