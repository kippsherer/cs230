#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import numpy as np
import random
import string
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import datasets as ds
import performance as stats

import models_MobileNetV2 as mnv2
import models_ResNet101 as rn
import models_Custom as cus

statistics = ['binary_crossentropy', 'accuracy', 'precision', 'recall', 'f1_score', 'false_negatives', 'false_positives']


# NOTE: ResNet models exceed GitHub size limits, so those are excluded from the repo.
# ['MobileNetV2_1a_nuky', 'MobileNetV2_1a_nvhi', 'MobileNetV2_1b_yrqb', 'MobileNetV2_1b_elvb', 'MobileNetV2_1b_pnli', 'MobileNetV2_1c_nhsg', 'MobileNetV2_2a_pulz', 'MobileNetV2_2a_apxw', 'MobileNetV2_2a_etti', 
#  'ResNet101_1a_hqtc', 'ResNet101_1b_gtpj', 'ResNet101_1c_hkkp', 'ResNet101_1c_fbce', 
#  'Custom_1a_lctx', 'Custom_2a_qdix', 'Custom_2b_qtve', 'Custom_1a_zhyk', 'Custom_2a_nvxs', 'Custom_2b_oyrn']
models_to_evaluate = ['Custom_1a_zhyk', 'Custom_2a_nvxs', 'Custom_2b_oyrn']

# get our dataset
dataset = ds.get_dataset_test()

# set our standardization
#standardize_image = mnv2.standardize_image
#standardize_image = rn.standardize_image
standardize_image = cus.standardize_image


for model in stats.model_statistics:
    #print (model)
    if model not in models_to_evaluate:
        continue

    # load model
    model = tf.keras.models.load_model('models/' + model + '.keras')

    print (model)

    # standardize images
    ds_std = dataset.map(standardize_image)
    result = model.evaluate(ds_std, verbose=2)


