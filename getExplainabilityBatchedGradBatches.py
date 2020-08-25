import numpy as np
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
import argparse
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os

'''
This assumes that intGradAnalyzer.py has already been run using seq2seq_avg_int_grad.py, use this to get data batches 
so that the grad Clustering file can be run. 
'''

parser = argparse.ArgumentParser()
parser.add_argument("model_folder", help="folder with saved model", type=str)
parser.add_argument("grad_folder", help="folder with gradients", type=str)
parser.add_argument("grad_save_folder", help="folder with new gradients", type=str)
parser.add_argument("val", help="validation set - all lower case: val1_sub for example", type=str)
parser.add_argument("data", help="dataset", type=str)
args = parser.parse_args()

model_folder = args.model_folder
grad_save_folder = args.model_folder + args.grad_save_folder
if not os.path.exists(grad_save_folder):
	os.makedirs(grad_save_folder)
val = args.val

# get config file for model run
sys.path.append(model_folder)
from util.config import *
if args.data == 'Amazon':
	Config = AmazonConfig()
elif args.data == 'Sentiment':
	Config = SentimentConfig()

VAL_BATCHES = Config.VAL_BATCHES
TEST_BATCHES = Config.TEST_BATCHES

def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)

# make loading in validation and test data easy
vs_batches = [VAL_BATCHES, TEST_BATCHES]
val_sets = Config.DATASETS
vs_dict = Config.DATASETS_DICT

grad_base = args.model_folder + args.grad_folder + val + '_grad_batch_'
x_base = 'Config.F_' + val.upper()

for i in range(vs_dict[val]):
	print(i)

	x_val = np.load(eval(x_base) + str(i) + '.npy')
	grad = np.load(grad_base + str(i) + '.npy')
	seq_len = x_val.shape[1]

	avg_grad_vec = []
	for policy_id in range(len(grad)):
		# average gradients
		event_grad = [[] for _ in range(Config.NUMBER_OF_EVENT_TYPES)]

		# append events to storage for the right event id
		for j in range(len(grad[policy_id])):
			tmp = x_val[policy_id][j]
			tmp_idx = np.where(tmp != 0)[0]
			for i_ in tmp_idx:
				event_grad[i_] += [np.sum(grad[policy_id][j] * tmp[i_])]

		# average attention
		avg_grad = []
		for j in range(len(event_grad)):
			if len(event_grad[j]) == 0:
				avg_grad += [0.]
			else:
				avg_grad += [np.mean(np.array(event_grad[j]))]
		avg_grad = np.array(avg_grad)
		avg_grad_vec += [avg_grad]

	avg_grad_vec = np.array(avg_grad_vec)
	np.save(grad_save_folder + 'X_' + val + '_grad_batch_' + str(i) + '.npy', avg_grad_vec)

