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


# create plots
plt.figure(figsize=(10, 10))


# create training accuracy plots
plt.subplot(2, 1, 1)
plt.title('Training Accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')

for model in stats.model_statistics:
    #print (model)
    plt.plot(stats.model_statistics[model]['training']['accuracy'], label=model+" (" + str(stats.model_statistics[model]['training']['accuracy'][-1]) + ")")

plt.ylim([min(plt.ylim()),1.01])
plt.legend(loc='lower right')


# create training accuracy plots
plt.subplot(2, 1, 2)
plt.title('Training Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')

for model in stats.model_statistics:
    #print (model)
    plt.plot(stats.model_statistics[model]['training']['loss'], label=model+" (" + str(stats.model_statistics[model]['training']['loss'][-1]) + ")")

plt.ylim([min(plt.ylim()),max(plt.ylim())])
plt.legend(loc='upper right')

plt.show()




plt.figure(figsize=(5, 15))
fig, ax = plt.subplots()

table_data=[]
row_labels=[]
for model in stats.model_statistics:
    #table_data.append( [model] + list(stats.model_statistics[model]['test'].values()) )
    table_data.append( list(stats.model_statistics[model]['test'].values()) )
    row_labels.append(model)

col_labels = list( stats.model_statistics['MobileNetV2_1a']['test'].keys() )

table = ax.table(cellText=table_data, colLabels=col_labels, rowLabels=row_labels, loc='center')

table.set_fontsize(14)
table.scale(2,2)
ax.axis('off')
plt.show()