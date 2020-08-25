import numpy as np
import glob
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument("atten_folder", help="folder with atten batches", type=str)  # .../violin-plot/'
parser.add_argument("type_dict", help="type dictionary", type=str)
parser.add_argument("--percent", help="min number of event types", default=0.05, type=float)
args = parser.parse_args()


load_folder = args.atten_folder + 'cluster-dat-all-sets/'
cluster_plot_dir = args.atten_folder + 'all-sets-timeseries-keys/'
save_folder = args.atten_folder + 'all-sets-timeseries-keys-types-per-' + str(args.percent) + '-all/'


if not os.path.exists(save_folder):
	os.makedirs(save_folder)

# load in dictionary
with open(args.type_dict, 'rb') as f:
	_dict = pickle.load(f)

# use index to get keys for each cluster
key = list(_dict.keys())
val = np.argsort(np.array(list(_dict.values())))
key = [key[t] for t in val]

N = len(key)

indices = np.load(args.atten_folder + 'IDX.npy')

NN = [int(len(ii)*args.percent) for ii in indices]
datasets = np.load(cluster_plot_dir + 'DATASETS.npy')
np.save(save_folder + 'IDX.npy', indices)
np.save(save_folder + 'DATASETS.npy', datasets)

num = len(glob.glob(load_folder + 'DAT_*'))
print(num)

for n in range(num):
	DAT = np.load(load_folder + 'DAT_' + str(n) + '.npy')

	MEAN_DEN = [(D != 0).sum(0) for D in DAT]
	IDX_TMP = [np.where(MD > NN[ii])[0] for ii, MD in enumerate(MEAN_DEN)]
	MEAN_VEC = [np.true_divide(DAT[i].sum(0), MEAN_DEN[i]) for i in range(len(DAT))]

	# MEAN_VEC = [D[IDX_TMP[i]] if len(IDX_TMP[i]) > NN else D for i, D in enumerate(DD)]
	_IDX = [np.argsort(-MV) for MV in MEAN_VEC]
	IDX = [[I for I in _IDX[i] if I in IDX_TMP[i]] for i in range(len(_IDX))]
	# print(MEAN_VEC[0].shape)
	MEAN_VAL = [MEAN_VEC[i][IDX[i]] for i in range(len(MEAN_VEC))]
	KEYS = [[key[i] for i in idx] for idx in IDX]
	np.save(save_folder + 'KEYS-' + str(n) + '.npy', KEYS)
	np.save(save_folder + 'MEAN-' + str(n) + '.npy', MEAN_VAL)
