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
This assumes that attenAnalyzer.py has already been run using seq2seq_avg_atten.py or seq2seq_hier_atten.py. Reorganize
explanation vector into data batches.
'''

parser = argparse.ArgumentParser()
parser.add_argument("model_folder", help="folder with saved model", type=str)
parser.add_argument("atten_type", help="avg or hier", type=str)
parser.add_argument("val", help="validation set - all lower case: val1_sub for example", type=str)
parser.add_argument("data", help="dataset", type=str)
args = parser.parse_args()

model_folder = args.model_folder
atten_save_folder = args.model_folder + 'atten/'
atten_type = args.atten_type
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

if not os.path.exists(atten_save_folder):
	os.makedirs(atten_save_folder)

atten = np.load(model_folder + val + '_atten_weights.npy')
x_base = 'Config.F_' + val.upper()

if atten_type.lower() == 'avg':
	for i in range(vs_dict[val]):
		print(i)

		x_val = np.load(eval(x_base) + str(i) + '.npy')
		seq_len = x_val.shape[1]
		print(atten[i].shape)
		print(x_val.shape)

		avg_atten_vec = []
		for policy_id in range(len(atten[i])):
			event_atten = [[] for _ in range(Config.NUMBER_OF_EVENT_TYPES)]
			for j in range(atten[i].shape[2]):
				tmp = x_val[policy_id][j]
				idx = np.where(tmp != 0)[0]
				for i_ in idx:
					event_atten[i_] += [atten[i][policy_id, 0, j, 0]]

			avg_atten = []
			for j in range(len(event_atten)):
				if len(event_atten[j]) == 0:
					avg_atten += [0.]
				else:
					avg_atten += [np.sum(np.array(event_atten[j]))]
			avg_atten = np.array(avg_atten)
			avg_atten_vec += [avg_atten]

		np.save(atten_save_folder + 'X_' + val + '_atten_batch_' + str(i)+'.npy', avg_atten_vec)

elif atten_type.lower() == 'hier':
	for i in range(vs_dict[val]):
			avg_atten_vec = atten[i][:, 0]
			np.save(atten_save_folder + 'X_' + val + '_atten_batch_' + str(i)+'.npy', avg_atten_vec)