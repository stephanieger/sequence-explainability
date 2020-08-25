import numpy as np
import glob
import matplotlib.pyplot as plt
import argparse
import os

'''
Plot violin plots along interpolation path, run once getExplainabilityViolinTimestepsSetsHypAll.py has been run.
'''
parser = argparse.ArgumentParser()
parser.add_argument("violin_folder", help="folder with attention batches", type=str)  # list of model folders in order)
parser.add_argument("val", help="which dataset", type=str)
parser.add_argument("type_dict", help="type dictionary", type=str)
parser.add_argument("--N", help="number of event types to consider", default=10, type=int)
args = parser.parse_args()

violin_folder = args.violin_folder
cluster_data_dir = violin_folder + 'cluster-dat-all-sets/'
save_dir = violin_folder + 'all-sets-timeseries-keys/'
N = args.N

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

KEYS = np.load(violin_folder + 'KEYS.npy')
num = len(glob.glob(cluster_data_dir + 'DAT_*'))
dataset = np.load(cluster_data_dir + 'DATASETS.npy')

DAT_VEC = np.load(cluster_data_dir + 'DAT_0.npy')
MEAN_VEC = [np.mean(t, axis=0) for t in DAT_VEC]
IDX = [np.argsort(-t) for t in MEAN_VEC]

for i, ds in enumerate(dataset):
	print(ds)
	_ds = ds.split('_')
	_pred = _ds[0]
	_clus = _ds[1]
	_lab = _ds[2]
	_key = KEYS[i]

	if len(DAT_VEC[i]) != 0:
		for feat in range(N):
			print(feat)
			dat_vec = []
			for n in range(num):
				if n % 8 == 0:
					DAT_VEC = np.load(cluster_data_dir + 'DAT_' + str(n) + '.npy')
					dat_vec += [DAT_VEC[i][:, IDX[i][feat]]]

			if num - 1 % 8 != 0:
				DAT_VEC = np.load(cluster_data_dir + 'DAT_' + str(num - 1) + '.npy')
				dat_vec += [DAT_VEC[i][:, IDX[i][feat]]]
			plt.figure()
			r = plt.violinplot(dat_vec, list(range(len(dat_vec))), widths=0.9, showmeans=True, showmedians=True,
							   showextrema=True, bw_method='silverman')
			r['cmeans'].set_color('g')
			r['cmedians'].set_color('b')
			plt.ylim(0, 1.1)
			_k = _key[feat].split('\'')[1].split('.')[0]
			plt.title('N cluster: ' + _clus + ' Pred: ' + _pred + ' Lab: ' + _lab + ' Feat: ' + _k)
			plt.savefig(save_dir + 'violin_cluster_' + ds + '_feat_' + str(feat) + '.png')
			plt.close()