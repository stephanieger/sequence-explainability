This repository contains implementations of the models discussed in the paper 
[Reproducibility of Attention-Based Explanations for RNNs"](https://arxiv.org/TBD)
by Stephanie Ger,Diego Klabjan and Jean Utke. 

## Table of Contents
* Data
* Training Classification Models
* Computing Explanations with Explainability Models
* Valley Tracking Low-Loss Path
* Examining Low-Loss Paths Between Trained Weights
* Using Scripts for Other Datasets

## Data 
Models were evaluated on two public datasets and these datasets are available [here](https://northwestern.app.box.com/folder/121557674519). The amazon-polarity.tar.gz file contains the Amazon review data and the IMDB.tar.gz file contains the IMDB review dataset. Each .zip file contains the training data, validation and test data batched by sequence length and additional inputs necessary for running equal time models. All data files are stored as numpy arrays.

## Training Classification Models
Sequence-to-sequence models with attention or hierarchical attention can be run with seq2seq_atten.py or seq2seq_hier.py respectively. Hyperparameters for model architecture such as 
number of layers or units in a given layer are set in the configuration file located at util/config.py. The random seed for the model can be set in util/config.py by changing the 
SEED_VALUE variable. Model accuracy and loss can be computed using the model_outputs.py script.

## Computing Explanations with Explainability Models

## Valley Tracking Low-Loss Path

## Examining Low-Loss Paths Between Trained Weights

## Using Scripts for Other Datasets
