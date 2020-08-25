import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
import os
from scipy.optimize import linear_sum_assignment
import argparse
import glob

'''
Plot how cluster morphs along interpolation plot. Run after interpolation path is determined. 
'''

parser = argparse.ArgumentParser()
parser.add_argument("atten_folder", help="folder with attention batches", type=str)  # list of model folders in order
parser.add_argument("save_folder", help="where to save the data", type=str)
parser.add_argument("val", help="which dataset", type=str)
parser.add_argument("--n_clusters", help="num clusters", default=2, type=int)
args = parser.parse_args()

if not os.path.exists(args.save_folder):
	os.makedirs(args.save_folder)

n_clusters = args.n_clusters

# get two
atten_dirs = args.atten_folder.split(',')
dirs = []
for atten_dir in atten_dirs:
	dirs += [glob.glob(atten_dir + 'interp-step-*')]

cnt = 0

for i, atten_dir in enumerate(atten_dirs):
	num_steps = len(dirs[i])
	print(num_steps)

	for j in range(num_steps - 1):
		clusterer_i = KMeans(n_clusters=n_clusters, random_state=10)
		clusterer_j = KMeans(n_clusters=n_clusters, random_state=10)
		dat_i = np.load(atten_dir + 'interp-step-' + str(j) + '/explainability/' + args.val + '_atten_vec.npy')
		dat_j = np.load(atten_dir + 'interp-step-' + str(j + 1) + '/explainability/' + args.val + '_atten_vec.npy')
		cnt += 1

		cluster_labels_i = clusterer_i.fit_predict(dat_i)
		cluster_labels_j = clusterer_j.fit_predict(dat_j)

		mat = contingency_matrix(cluster_labels_i, cluster_labels_j)
		_mat = np.max(mat) - mat
		row_ind, col_ind = linear_sum_assignment(_mat)

		if i == 0:
			if j == 0:
				np.save(args.save_folder + 'cluster_labels_0.npy', cluster_labels_i)
		np.save(args.save_folder + 'cluster_labels_' + str(cnt) + '.npy', cluster_labels_j)
		np.save(args.save_folder + 'col_ind_' + str(cnt) + '.npy', col_ind)


	if i != len(atten_dirs) - 1:  ## if it's not the last restart of the directory, compute ARI score between dirs
		clusterer_i = KMeans(n_clusters=n_clusters, random_state=10)
		clusterer_j = KMeans(n_clusters=n_clusters, random_state=10)
		dat_i = np.load(atten_dir + 'interp-step-' + str(num_steps - 1) + '/explainability/' + args.val +
						'_atten_vec.npy')
		dat_j = np.load(atten_dirs[i + 1] + 'interp-step-' + str(0) + '/explainability/' + args.val +
						'_atten_vec.npy')
		cnt += 1

		cluster_labels_i = clusterer_i.fit_predict(dat_i)
		cluster_labels_j = clusterer_j.fit_predict(dat_j)

		mat = contingency_matrix(cluster_labels_i, cluster_labels_j)
		_mat = np.max(mat) - mat
		row_ind, col_ind = linear_sum_assignment(_mat)

		if i == 0:
			if j == 0:
				np.save(args.save_folder + 'cluster_labels_0.npy', cluster_labels_i)

		np.save(args.save_folder + 'cluster_labels_' + str(cnt) + '.npy', cluster_labels_j)
		np.save(args.save_folder + 'col_ind_' + str(cnt) + '.npy', col_ind)