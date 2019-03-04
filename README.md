[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2559229.svg)](https://doi.org/10.5281/zenodo.2559229)

# Learning to Rank Webpages Based on Visual Features
This repository contains the code required to reproduce the results described in the The Web Conference 2019 paper "Learning to Rank Webpages Based on Visual Features" by Bram van den Akker, Ilya Markov and Maarten de Rijke. The implementation has been written by Bram van den Akker.  

# Dataset
The dataset can be found [here](https://github.com/Braamling/learning-to-rank-webpages-based-on-visual-features/blob/master/dataset.md).

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
The model can be trained using one of two methods: i) train a single model based on a a single setup using `train.py`, or ii) train multiple models specified in `tune.py`. The methods of running the code will need a bit of rework in the future to make it more user friendly. In order to speedup the training process, it is recommended to cache the frozen layers from the visual feature extractor using `cache_frozen_layers.py` 

### cache_frozen_layers.py
Using this file is fairly straight forward, it will automatically cache `vgg16`, `resnet152`, `resnet18` feature vectors for all the images in a specified folder. Make sure that the input images are of size `224x224` before starting. 

```
usage: cache_frozen_layers.py [-h] [--image_folder IMAGE_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --image_folder IMAGE_FOLDER
                        The location of all the images. 
```

### train.py
`train.py` has a very broad set of options and can therefor be used in a couple of ways. Let me introduce you to the two standard methods: i) training using images, and ii) training using cache layers with a custom transformation compontent. The available parameters can be found at the end of this section.



#### Training using images (Example)

```bash
python3 train.py 
  --learning_rate 0.00005 
   --batch_size 100 
   --content_feature_size 11 
   --description resnet152_1_layers  
   --query_specific False 
   --image_path storage/images_224x224/snapshots/  
   --visual_dropout 0.1 
   --classification_dropout 0.1 
   --load_images True 
   --model resnet152 
   --content_feature_dir storage/clueweb12_web_trec  
```

#### Training using cached layers (Example)
Note: When using a cache it is possible to provide provide hidden sizes seperated by an 'x' to transform the outputs of the visual layers to another hidden representation. ie. `--visual_layers 4096x4096` together with `--cache_vector_size 2048` and `--visual_features 30` will be converted to a small feed forward neural network that converts the cached vector of size 2048 to a hidden layer of size 4096, then another hidden layer of size 4096 and finally a visual feature vector of size 30. 

```bash
python3 train.py 
  --content_feature_size 11 
  --description resnet152_highlights_d_10_4096x4096_lr_00001 
  --query_specific True 
  --image_path storage/images_224x224/highlights/ 
  --load_images True
  --model transform_cache 
  --content_feature_dir storage/clueweb12_web_trec
  --cache_path storage/model_cache/resnet152-highlights-cache 
  --epochs 20 
  --learning_rate 0.00001 
  --batch_size 100 
  --dropout 0.10 
  --cache_vector_size 2048 
  --visual_layers 4096x4096
```

|       Parameter        |         Default Value          |                                                               Description                                                               |
|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| content_feature_dir    | 'storage/clueweb12_web_trec/', | The location of all the folds with train, test and validation files.                                                                    |
| folds                  | 5,                             | The amounts of folds to train on.                                                                                                       |
| sessions_per_fold      | 1,                             | The amount of training sessions to average per fold.                                                                                    |
| image_path             | 'storage/images/snapshots/',   | The location of the salicon images for training.                                                                                        |
| saliency_dir           | None,                          | [optional] The path of the directory where the saliency images are stored. No saliency images will be stored the argument is not passed |
| saliency_cache_path    | None,                          | [optional] The path of the directory where the saliency images cache is stored. ie 'storage/model_cache/restnet152-saliency-cache.'     |
| cache_path             | None,                          | Provide the path of a feature extractor cache path in order to speed up training ie. 'storage/model_cache/restnet152-saliency-cache.    |
| batch_size             | 3,                             | The batch size used for training.                                                                                                       |
| epochs                 | 10,                            | The amount of epochs used to train.                                                                                                     |
| description            | 'example_run',                 | The description of the run, for logging, output and weights naming.                                                                     |
| learning_rate          | 0.01,                          | The learning rate to use for the experiment                                                                                             |
| content_feature_size   | 11,                            | The amount of context features                                                                                                          |
| model                  | "features_only",               | chose the model to train, (features_only, ViP)                                                                                          |
| load_images            | "True",                        | set whether the images should be loaded during training and evaluation.                                                                 |
| only_with_image        | "True",                        | set whether all documents without images should be excluded from the dataset                                                            |
| query_specific         | "False",                       | set whether the images are query specific, which means that there might be multiple screenshot for a single document (ie. when using highlighted screenshots)                                                         |
| log_dir                | 'storage/logs/{}',             | The location to place the tensorboard logs.                                                                                             |
| optimized_scores_path  | 'storage/logs/',               | The location to store the scores that were optimized.                                                                                   |
| optimize_on            | 'ndcg@5',                      | Give the measure to optimize the model on (ndcg@1, ndcg@5, ndcg@10, p@1, p@5, p@10, map).                                               |
| grayscale              | 'False',                       | Flag whether to convert the images to grayscale.                                                                                        |
| classification_dropout | .1,                            | The dropout to use in the classification layer.                                                                                         |
| visual_dropout         | .1,                            | The dropout to use in the visual feature layer.                                                                                         |
| hidden_size            | 10,                            | The amount of hidden layers in the classification layer                                                                                 |
| visual_layers          | "2048x2048",                   | [cached only] Provide hidden sizes seperated by an 'x' that transforms the visual outputs to a hidden representation                    |
| cache_vector_size      | 25088,                         | [cached only] the size of the output vectors stored in the cache                                                                        |
| visual_features        | 30,                            | The size of the visual feature vector                                                                                                   |
| finetune_n_layers      | 1,                             | For resnet152 and inception, define the amount of layers at the end to be fine tuned.                                                   |



### tune.py
The `tune.py` file can be used as an example to create a grid search. It currently doesn't have a proper configuration file. however, the different grid search parameters can be configured at the end of the file itself. An example of the parameters can be found below. The parameters are explained in the `train.py` instructions.

```python
    FLAGS.parameters = {
        '--learning_rate':          ['0.0001', '0.00005'],
        '--classification_dropout': ['0.10', '0.20'],
        '--visual_dropout':         ['0.50', '0.10'],
        '--visual_layers':          ['4096x4096x4096x4096', '4096x4096x4096', '2048x2048x2048x2048',
                                     '1024x1024x1024x1024x1024x1024',
                                     '1024x1024x1024x1024', '2048x4096x2048'],
        '--optimize_on':            ['p@10'],
        '--batch_size':             ['100']
    }
```

```
usage: tune.py [-h] [--log_name LOG_NAME] [--model MODEL]
               [--infrastructure_type INFRASTRUCTURE_TYPE]
               [--input_type INPUT_TYPE]
               [--cache_vector_size CACHE_VECTOR_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --log_name LOG_NAME   the name to use as a prefix while logging the results
                        (usually model name)
  --model MODEL         Name of the visual feature model to train ("vgg16", "resnet152", "resnet18", "inception", "features_only", "
  --infrastructure_type INFRASTRUCTURE_TYPE
                        (Experimental) The infrastruce can be set to `saliency_add` to combine saliency and another visual feature vector.
  --input_type INPUT_TYPE
                        Type of input images to use (saliency, snapshots, highlights or None)
  --cache_vector_size CACHE_VECTOR_SIZE
                        Name of the input to train on.

```
