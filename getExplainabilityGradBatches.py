import numpy as np
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
import argparse
import sys
import matplotlib
matplotlib.use('Agg')
import os

'''
This assumes that gradAnalyzer.py has already been run using seq2seq_avg_grad.py or seq2seq_hier_grad.py. Get grad for
each batch. 
'''

parser = argparse.ArgumentParser()
parser.add_argument("model_folder", help="folder with saved model", type=str)
parser.add_argument("val", help="validation set - all lower case: val1_sub for example", type=str)
parser.add_argument("data", help="dataset", type=str)
args = parser.parse_args()

model_folder = args.model_folder
grad_save_folder = args.model_folder + 'grad/'
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

# make loading in validation and test data easy
vs_batches = [VAL_BATCHES, TEST_BATCHES]
val_sets = Config.DATASETS
vs_dict = Config.DATASETS_DICT

grad = np.load(model_folder + val + '_norm_grad_weights.npy')
x_base = 'Config.F_' + val.upper()

for i in range(vs_dict[val]):

	x_val = np.load(eval(x_base) + str(i) + '.npy')
	seq_len = x_val.shape[1]

	avg_grad_vec = []
	for policy_id in range(len(grad[i])):
		# average gradients
		event_grad = [[] for _ in range(Config.NUMBER_OF_EVENT_TYPES)]

		# append events to storage for the right event id
		for j in range(len(grad[i][policy_id])):
			tmp = x_val[policy_id][j]
			tmp_idx = np.where(tmp != 0)[0]
			for i_ in tmp_idx:
				event_grad[i_] += [np.sum(grad[i][policy_id][j] * tmp[i_])]

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

