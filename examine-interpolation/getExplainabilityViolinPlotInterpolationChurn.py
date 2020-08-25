import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import argparse
import glob
import os

'''		
This script is designed to compute violin plots for samples who's cluster ID does not change, and how the morph during
the weight interpolation path. Only works for n_clusters=2
'''

parser = argparse.ArgumentParser()
parser.add_argument("atten_folder", help="folder with attention batches", type=str)  # list of model folders in order
parser.add_argument("cluster_lab_folder", help="folder with the cluster identification for each sample", type=str)
parser.add_argument("save_folder", help="where to save the data", type=str)
parser.add_argument("val", help="which dataset", type=str)
parser.add_argument("type_dict", help="type dictionary", type=str)
parser.add_argument("--N", help="number of event types to consider", default=10, type=int)
args = parser.parse_args()

N = args.N
save_folder = args.save_folder
if not os.path.exists(save_folder):
	os.makedirs(save_folder)

cluster_data_dir = save_folder + 'churn-cluster-dat/'
if not os.path.exists(cluster_data_dir):
	os.makedirs(cluster_data_dir)
cluster_dir = [save_folder + 'churn-cluster-' + str(t) + '/' for t in range(2)]
for _dir in cluster_dir:
	if not os.path.exists(_dir):
		os.makedirs(_dir)

# make save folders for figs in each cluster

# load in dictionary
with open(args.type_dict, 'rb') as f:
	_dict = pickle.load(f)

# figure out the number of samples that stay in the same cluster
num_steps = len(glob.glob(args.cluster_lab_folder + 'churn_cluster_labels_*.npy'))
init_label = np.load(args.cluster_lab_folder + 'churn_cluster_labels_0.npy')
labels = []
labels += [init_label]

for num in range(0, num_steps):
	lab = np.load(args.cluster_lab_folder + 'churn_cluster_labels_' + str(num) + '.npy')
	mat = contingency_matrix(init_label, lab)
	_mat = np.max(mat) - mat
	r,c = linear_sum_assignment(_mat)
	if (c == np.array([1,0])).all():
		lab = 1 - lab
	labels += [lab]
	init_label = np.copy(lab)

LAB = np.array(labels)
SUM = np.sum(LAB, axis=0)
# get indices for each cluster label
idx_one = np.where(SUM == LAB.shape[0])[0]
idx_zero = np.where(SUM == 0)[0]

atten_dirs = args.atten_folder.split(',')
dirs = []
for atten_dir in atten_dirs:
	dirs += [glob.glob(atten_dir + 'interp-step-*')]

cnt = 0
# load in initial point to get IDX
dat = np.load(atten_dirs[0] + 'interp-step-0/explainability/' + args.val + '_churn_atten_vec.npy')
DAT = [dat[idx_zero], dat[idx_one]]

np.save(cluster_data_dir + 'CHURN_DAT_' + str(cnt) + '.npy', DAT)
MEAN_VEC = [np.mean(t, axis=0) for t in DAT]
IDX = [np.argsort(-t) for t in MEAN_VEC]

# use index to get keys for each cluster
key = list(_dict.keys())
KEYS = [[key[i] for i in idx][:N] for idx in IDX]
np.save(save_folder + 'CHURN_KEYS.npy', KEYS)

for i, atten_dir in enumerate(atten_dirs):
	num_steps = len(dirs[i])
	print('num steps:', num_steps)

	for j in range(num_steps):
		print(j)
		# load in data and compute necc. components for violin plot for each cluster
		dat = np.load(atten_dir + 'interp-step-' + str(j) + '/explainability/' + args.val + '_churn_atten_vec.npy')
		DAT = [dat[idx_zero], dat[idx_one]]
		MEAN_VEC = [np.mean(t, axis=0) for t in DAT]
		STD_VEC = [np.mean(t, axis=0) for t in DAT]

		MEAN_VEC = [t[IDX[u]][:N] for u, t in enumerate(MEAN_VEC)]
		STD_VEC = [t[IDX[u]][:N] for u, t in enumerate(STD_VEC)]
		DAT_VEC = [t[:, IDX[u]][:, :N] for u, t in enumerate(DAT)]
		np.save(cluster_data_dir + 'CHURN_DAT_' + str(cnt) + '.npy', DAT)

		for k, dat_vec in enumerate(DAT_VEC):
			if len(dat_vec) != 0:
				plt.figure()
				r = plt.violinplot(dat_vec, list(range(N)), widths=0.9, showmeans=True, showmedians=True, showextrema=True,
							   bw_method='silverman')
				r['cmeans'].set_color('g')
				r['cmedians'].set_color('b')
				plt.ylim(0, 1.1)
				plt.title('step:' + str(cnt))
				plt.savefig(cluster_dir[k] + 'violin_' + str(cnt) + '.png')
				plt.close()
		cnt += 1
