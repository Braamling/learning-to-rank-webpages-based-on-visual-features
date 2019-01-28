# Learning to Rank Webpages Based on Visual Features
This repository contains the code required to reproduce the results described in the The Web Conference 2019 paper "Learning to Rank Webpages Based on Visual Features" by Bram van den Akker, Ilya Markov and Maarten de Rijke. The implementation has been written by Bram van den Akker.  

# Preprocessing
The code for the preprocessing can be found in three separate repositories. 

## Wayback machine screenshot scaper
[web-screenshot-scraper](https://github.com/Braamling/web-screenshot-scraper) contains all the code that was used to create the screenshot from the Wayback machine and [ClueWeb12 rendering service](https://lemurproject.org/clueweb12/services.php). The ClueWeb12 rendering service is a closed service, so might not be available for your purposes. 

## Saliency heatmap predictions
[web-page-saliency-prediction-using-two-stage-transfer-learning](https://github.com/Braamling/web-page-saliency-prediction-using-two-stage-transfer-learning) contains a pytorch implementation for predicting web page saliency: *Shan, Wei, et al. "Two-Stage Transfer Learning of End-to-End Convolutional Neural Networks for Webpage Saliency Prediction." International Conference on Intelligent Science and Big Data Engineering. Springer, Cham, 2017.*. This code has been used to create the saliency heatmaps based on the VITOR dataset. 

## Contextual features
[contextual-search-features-for-large-datasets-in-spark](https://github.com/Braamling/contextual-search-features-for-large-datasets-in-spark) contains all the neccary Scala Spark code to generate the contextual features in VITOR from the raw ClueWeb12 dataset. The resulting data will be in the same format as the LETOR dataset.

# Experiments code
Given that all preprocessing steps have been performed, the rest of this README is dedicated to the actual experiments using the code in this repository. 

## Prerequisites 
Please make sure to have python 3 and all the requirements in requirements.txt installed. 

> pip3 install -r requirements.txt

Make sure to download the VITOR dataset (dataset will be published shortly) and place the content in a new directory called `storage`.

## Introduction to code structure
Each architecture is build as a combination of two models. The `prepare_model` class in `train.py` initializes a feature extraction model and scoring model taken from the `/models` directory. The input of each model should always be an rgb image or arbitrary size and a letor style folded directory. The output is always a single document score. 

The Clueweb12Dataset class in `/utils/cluewebDataset.py` interfaces the VITOR dataset by overloading the standard PyTorch dataset class. This class makes sure than only entries with an existing image are used during training and evaluation.  

During training, Evaluate objects from `evaluate.py` are created during each epoch. The evaluate class uses an interface on top of the dataset to evaluate train, validate and test data given a certain trained model. 


## Running the experiments
TODO: Write this part
