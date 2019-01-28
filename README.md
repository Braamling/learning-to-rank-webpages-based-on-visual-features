# Human-Saliency-on-Snapshots-in-Web-Relevance
This repository contains all the code to reproduce the results and preprocessing steps described in the paper <PLACEHOLDER>. 

# Paper
The LaTeX source and corresponding images can be found in `/paper`. 

# Preprocessing
All instructions can be found in the corresponding directories, a small description can be found below
TODO: Put these in separate repositories, change their names and place the link here.

## highlightGenerator
`/preprocessing/highlightGenerator` contains all the code to scrape the wayback machine and online rendering service. 

## saliencyGenerator
`/preprocessing/saliencyGenerator` contains a pytorch implementation for predicting web page saliency: *Shan, Wei, et al. "Two-Stage Transfer Learning of End-to-End Convolutional Neural Networks for Webpage Saliency Prediction." International Conference on Intelligent Science and Big Data Engineering. Springer, Cham, 2017.*


## contextualFeaturesGenerator
`/preprocessing/contextualFeaturesGenerator` contains Spark and Python code that was used to calculate the content features and transform it to the LETOR format.


# Experiments code
Given that all preprocessing steps have been performed, the rest of this README is dedicated to the actual experiment codebase. 

## Prerequisites 
Please make sure to have python 3 and all the requirements in requirements.txt installed. 

> pip3 install -r requirements.txt

Make sure to download the <PLACEHOLDER> dataset <TODOHERE> and place the content in a new directory called `storage` (TODO: create download script).

## Introduction to code structure
Each architecture is build as a combination of two models. The `prepare_model` class in `train.py` initializes a feature extraction model and scoring model taken from the `/models` directory. The input of each model should always be an rgb image or arbitrary size and a letor style folded directory. The output is always a single document score. 

The Clueweb12Dataset class in `/utils/cluewebDataset.py` interfaces the <PLACEHOLDER> dataset by overloading the standard PyTorch dataset class. This class makes sure than only entries with an existing image are used during training and evaluation.  

During training, Evaluate objects from `evaluate.py` are created during each epoch. The evaluate class uses an interface on top of the dataset to evaluate train, validate and test data given a certain trained model. 


## Running the experiments
TODO: Write this part