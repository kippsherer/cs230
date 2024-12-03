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
    * Train a model
        1. edit datasets.py, set DATASET_DIRECTORY to correct directory
        2. edit train_model.py, include correct models_ file
        3. edit train_model.py, adjust standardization and model to be used
        4. execute train_model.py

    * Evaluate model(s)
        1. edit datasets.py, set DATASET_DIRECTORY to correct directory
        2. edit evaluate.py, set models_to_evaluate to desired models.
        3. execute evaluate.py

    * Graph model(s)
        1. open Comparisons.ipynb Jupyter notebook
        2. follow comments in the notebook

# Description of Files
* train_model.py - Executable used to train a model and run tests against the other datasets 
* test_environment.py - Executable used while testing the environment 
* evaluate.py - Executable used to test and graph models
&nbsp;   
* Comparisons.ipynb - Jupyter notebook used while comparing models
* datasets.py - Contains functions for retrieving the training, dev, test, and test_dark datasets
* performance.py - Contains dictionary of metrics collected during training and testing of various models
&nbsp;  
* models_Custom.py - Contains functions to retrieve various models
* models_MobileNetV2.py - Contains functions to retrieve various models based on MobileNetV2
* models_ResNet101.py - Contains functions to retrieve various models based on ResNet101
&nbsp;  
* ImageAugmentation Directory
    * frameExtraction.py - Executable used to extract frames and augment the images used in the datasets
    * positiveFrameExtraction.py - Executable used to extract and augment the positive images since they get more augmentation

* models Directory
Directory where the trained models are saved in .keras format after running train_model.py. NOTE: ResNet models exceed GitHubs size limit and are excluded

# Retrieve Datasets
* Train - http://kippsherer.com/images/train.zip (3G)
* Dev - http://kippsherer.com/images/dev.zip (174M)
* Test - http://kippsherer.com/images/test.zip (174M)

