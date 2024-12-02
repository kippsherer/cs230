# CS230 Deep Learning Project
Every year vehicle collisions with large livestock cause untold damage to life and property. In
Arizona, like in other open-range states, many drivers don’t realize the real hazard and frequency of
livestock on the road until it is too late.
Studies have shown that motorist alert signage that is activated specifically at the time and place of the
hazard have higher effectiveness in preventing collisions with wildlife than signs alone. Current radar
based systems activate whenever there is a car even if there is no current animal hazard, essentially a
false positive from the point of view of the motorist.
Using video to detect the hazard and notify drivers of impending hazard with alert activated signs
could be a cost effective method to prevent a number accidents.
What makes this project interesting from a deep learning point of view are the challenges which make
it more complicated than a normal object detection problem.

# Quick Start

# Description of Files
* Comparisons.ipynb - Jupyter notebook used while comparing models
* datasets.py - contains functions for retrieving the training, dev, test, and test_dark datasets
* evaluate.py - TBD
* performance.py - contains dictionary of metrics collected during training and testing of various models

* train_models.py - executable used to train a model and run tests against the other datasets 
* test_environment.py - executable used while testing the environment 

* models_Custom.py - contains functions to retrieve various models
* models_MobileNetV2.py - contains functions to retrieve various models based on MobileNetV2
* models_ResNet101.py - contains functions to retrieve various models based on ResNet101


* ImageAugmentation Directory
    * frameExtraction.py - executable used to extract frames and augment the images used in the datasets
    * positiveFrameExtraction.py - executable used to extract and augment the positive images since they get more augmentation





