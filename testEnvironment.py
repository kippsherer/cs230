import matplotlib.pyplot as plt
#import json
import os
import cv2
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import image_dataset_from_directory

GPUs = tf.config.list_physical_devices('GPU')
print("Number of GPUs Available: ", len(GPUs))
if len(GPUs) > 0:
    tf.config.experimental.set_memory_growth(GPUs[0], True)


BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SEED = 42

img = cv2.imread("D:\\storage\\Data\\MLdata\\dashcam\\frames_224\\pos\\pos_20241102145051_000040Aframe1412_0_0.png", cv2.IMREAD_ANYCOLOR)
while True:
    cv2.imshow("test", img)
    cv2.waitKey(0)
    sys.exit() # to exit from all the processes
 
cv2.destroyAllWindows() # destroy all windows

exit()

directory = 'D:\\storage\\Data\\MLdata\\dashcam\\frames_224'
train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             label_mode='binary',
                                             validation_split=0.1,
                                             subset='training',
                                             seed=SEED)

# total non training images
validation_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             label_mode='binary',
                                             validation_split=0.1,
                                             subset='validation',
                                             seed=SEED)

dev_dataset = validation_dataset.take(3000)

test_dataset = validation_dataset.skip(3000)


print ( train_dataset.info )

print ( train_dataset.cardinality() )

#for file_path in train_dataset.take(5):
#    print(file_path)


#test_ds = val_ds.take((2*val_batches) // 3)
#val_ds = val_ds.skip((2*val_batches) // 3)




#class_names = train_dataset.class_names

#plt.figure(figsize=(10, 10))
#for images, labels in train_dataset.take(1):
#    for i in range(9):
#        ax = plt.subplot(3, 3, i + 1)
#        plt.imshow(images[i].numpy().astype("uint8"))
#        plt.title(class_names[labels[i]])
#        plt.axis("off")
        




