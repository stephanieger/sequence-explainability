import numpy as np
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
import argparse
import sys
import matplotlib
matplotlib.use('Agg')
import glob
import os

'''
This assumes that attenAnalyzer.py has already been run using seq2seq_avg_atten_interpolation.py or
seq2seq_hier_atten_interpolation.py (respectively). 
'''

parser = argparse.ArgumentParser()
parser.add_argument("model_folder", help="folder with config file", type=str)
parser.add_argument("atten_folder", help="folder with saved w_path attention outputs", type=str)
parser.add_argument("atten_type", help="avg or hier", type=str)
parser.add_argument("val", help="validation set - all lower case: val1_sub for example", type=str)
parser.add_argument("data", help="dataset", type=str)
args = parser.parse_args()

model_folder = args.model_folder

# get config file for model run
sys.path.append(model_folder)
# from util.config import *
from config import *
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

TIMESTEPS = Config.TIMESTEPS

dirs = glob.glob(args.atten_folder + 'interp-step-*')
for _dir in dirs:
	save_folder = _dir + '/explainability/'

	# atten_save_folder = _dir + '/atten/'
	atten_type = args.atten_type
	val = args.val

	if not os.path.exists(save_folder):
		os.makedirs(save_folder)

	atten = np.load(_dir + '/'+ val + '_atten_weights.npy')
	x_base = 'Config.F_' + val.upper()
	y_base = 'Config.Y_' + args.val.upper()

	atten_vec = []
	churn_atten_vec = []
	non_churn_atten_vec = []

	atten_feat_vec = []
	churn_atten_feat_vec = []
	non_churn_atten_feat_vec = []

	if atten_type.lower() == 'avg':
		for i in range(vs_dict[val]):
			print(i)

			x_val = np.load(eval(x_base) + str(i) + '.npy')
			y_val = np.load(eval(y_base) + str(i) + '.npy')

			lab = np.argmax(y_val, axis=2)
			lab = lab[:, 0]

			seq_len = x_val.shape[1]
			print(atten[i].shape)
			print(x_val.shape)

			feat_atten_vec = np.squeeze(atten[i], axis=(1, 3))
			seq_len = feat_atten_vec.shape[1]
			feat_atten_vec = np.pad(feat_atten_vec, ((0, 0), (TIMESTEPS - seq_len, 0)), 'constant', constant_values=(0))

			atten_feat_vec += [feat_atten_vec]

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

				atten_vec += [avg_atten]
				# old_grad_vec += [old_grad[policy_id]]

				# Consider churn policies separately
				if lab[policy_id] != 0:
					churn_atten_vec += [avg_atten]
					churn_atten_feat_vec += [feat_atten_vec[policy_id]]
				else:
					non_churn_atten_vec += [avg_atten]
					non_churn_atten_feat_vec += [feat_atten_vec[policy_id]]



		churn_atten_vec = np.array(churn_atten_vec)
		non_churn_atten_vec = np.array(non_churn_atten_vec)
		atten_vec = np.array(atten_vec)
		churn_atten_feat_vec = np.array(churn_atten_feat_vec)
		non_churn_atten_feat_vec = np.array(non_churn_atten_feat_vec)
		atten_feat_vec = np.concatenate(atten_feat_vec)


		# save arrays
		np.save(save_folder + args.val + '_churn_atten_vec.npy', churn_atten_vec)
		np.save(save_folder + args.val + '_non_churn_atten_vec.npy', non_churn_atten_vec)
		np.save(save_folder + args.val + '_atten_vec.npy', atten_vec)
		np.save(save_folder + args.val + '_churn_atten_feat_vec.npy', churn_atten_feat_vec)
		np.save(save_folder + args.val + '_non_churn_atten_feat_vec.npy', non_churn_atten_feat_vec)
		np.save(save_folder + args.val + '_atten_feat_vec.npy', atten_feat_vec)

	elif atten_type.lower() == 'hier':
		for i in range(vs_dict[val]):
			avg_atten_vec = atten[i][:, 0]
			y_val = np.load(eval(y_base) + str(i) + '.npy')

			lab = np.argmax(y_val, axis=2)
			lab = lab[:, 0]

			for policy_id in range(len(avg_atten_vec)):

				atten_vec += [avg_atten_vec[policy_id]]
				# old_grad_vec += [old_grad[policy_id]]

				# Consider churn policies separately
				if lab[policy_id] != 0:
					churn_atten_vec += [avg_atten_vec[policy_id]]
				else:
					non_churn_atten_vec += [avg_atten_vec[policy_id]]

		churn_atten_vec = np.squeeze(np.array(churn_atten_vec))
		non_churn_atten_vec = np.squeeze(np.array(non_churn_atten_vec))
		atten_vec = np.squeeze(np.array(atten_vec))

		np.save(save_folder + args.val + '_churn_atten_vec.npy', churn_atten_vec)
		np.save(save_folder + args.val + '_non_churn_atten_vec.npy', non_churn_atten_vec)
		np.save(save_folder + args.val + '_atten_vec.npy', atten_vec)
				# np.save(atten_save_folder + 'X_' + val + '_atten_batch_' + str(i)+'.npy', avg_atten_vec)