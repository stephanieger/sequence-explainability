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

The arguments for each of these explainability scripts are the path to where the trained model is stored and the path to the best trained weights for that model. Once the explainability model has been run, explanations can be batched and clustered to get a single file for the explanation vectors of each model.In this batching process, a single explanation value is computed for event types that occur multiple times in a single sequence for the attention, gradient and integrated gradient models. To run this batching and clustering process, different scripts must be run for the attention based models and the gradient based models. For the attention based models, we run getExplainabilityAttenBatches.py and getExplainabilityAttenClustering.py sequentially. For the gradient model we first run getExplainabilityGradBatches.py and for the integrated gradient model we first run getExplainabilityBatchedGradBatches.py. For all gradient models, we next run getExplainabilityGradClustering.py. After the batching and clustering code has been run, to visualize clustering, getExplainabilityClusterPlots.py and getExplainabilityComparison.py can be run.

## Valley Tracking Low-Loss Path

To compute a low-loss path between trained models, use the weight_interpolation.py script in the explainability-interpolation/ directory. To run this script, we need trained weights from two different models that we want to interpolate between as well as maximum allowable increase in loss (epsilon_loss). Setting the appropriate epsilon_loss takes some trial and error. Depending on the epsilon_loss, the model training may take a few days and the explainbility-interpolation/weight_interpolation_res.py can be used to restart training. To compute a straight line interpolation path between trained weights to compare the low-loss path against, use the explainability-interpolation/straight_line_interp.py. 

## Examining Low-Loss Paths Between Trained Weights

The scripts used for examining the low-loss paths between trained weights are located in the examine-interpolation/ directory. There are a few ways to examine the interpolation plots. Once the interpolation model has been run, the seq2seq_avg_atten_interp.py and seq2seq_hier_atten_interp.py scripts can be used to compute the attention weights along the path and getExplainabilityAttenBatchClusterInterpolation.py is used to batch and compute the explanation vectors from attention. Once we have the attention vectors, the scripts with ARI in the name can used to compute the ARI score along the interpolation, either with respect to an endpoint, or to the neighboring point. The getExplainabilityF1ScoreInterpolation.py script is used to compute the F1-score for the weights along the interpolation path and once examine-interpolation/getExplainabilityViolinTimestepsSetsHypAll.py is run, the ViolinPlot files can be used to plot the violin plot morphing along the interpolation path. Finally, the MorphPlot scripts are used to plot how the cluster membership changes along the interpolation path. 

## Using Scripts for Other Datasets
To use these scripts for other datasets, the following inputs must be created.
X: Data of the shape (batch, timesteps, features), in the case of word data, an embedding like BERT should be used to get a vector representation for each word. 
Y: Label for the data of the shape (batch, labels)
F: One hot encoding of the event types of the data of the shape (batch, timesteps, number_of_events), in the case of words we used hypernyms from the NLTK wordnet embedding. 
E: Event Type of the shape (batch,timesteps) where we have an ordered list of event types and the index corresponds to a given event type. 
