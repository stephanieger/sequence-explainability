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
Models were evaluated on two public datasets and these datasets are available [here](https://northwestern.app.box.com/folder/121557674519). The amazon-polarity.tar.gz file contains the Amazon review data and the IMDB.tar.gz file contains the IMDB review dataset. Each .zip file contains the training data, validation and test data batched by sequence length and additional inputs necessary for running equal time models. Note that for the IMDB dataset, files must be unzipped and moved out of the respective folders so that all numpy arrays located in batch/. All data files are stored as numpy arrays.

## Training Classification Models
Sequence-to-sequence models with attention or hierarchical attention can be run with seq2seq_avg.py or seq2seq_hier.py respectively. Hyperparameters for model architecture such as number of layers or units in a given layer are set in the configuration file located at util/config.py. The random seed for the model can be set in util/config.py by changing the SEED_VALUE variable. Model accuracy and loss can be computed using the model_outputs.py script.

## Computing Explanations with Explainability Models

There are four explainability methods that can be used to generate model explanations. These methods with the associated model script are listed here:

1. Attention (seq2seq_avg_atten.py, run on avg model)
2. Hierarchical Atten (seq2seq_hier_atten.py, run on hier model)
3. Gradient (seq2seq_avg_grad.py or seq2seq_hier_grad.py, run on avg or hier model respectively)
4. Integrated Gradient (seq2seq_avg_int_grad.py, run on avg model)

The arguments for each of these explainability scripts are the path to where the trained model is stored and the path to the best trained weights for that model. Once the explainability model has been run, explanations can be batched and clustered to get a single file for the explanation vectors of each model.In this batching process, a single explanation value is computed for event types that occur multiple times in a single sequence for the attention, gradient and integrated gradient models. To run this batching and clustering process, different scripts must be run for the attention based models and the gradient based models. 

## Valley Tracking Low-Loss Path

## Examining Low-Loss Paths Between Trained Weights

## Using Scripts for Other Datasets
