import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import glob

'''
Compute the ARI between the two trained weights for each interpolation model
'''
parser = argparse.ArgumentParser()
parser.add_argument("model_folders", help="folder with saved model", type=str)
parser.add_argument("atten_folder", help="folder with attention batches", type=str)  # list of model folders in order
parser.add_argument("val", help="validation set", type=str)
parser.add_argument("data", help="dataset", type=str)
args = parser.parse_args()

atten_dirs = args.atten_folder.split(',')
dirs = []
for atten_dir in atten_dirs:
	dirs += [glob.glob(atten_dir + 'interp-step-*')]

# ARI = []
dat_i = np.load(atten_dirs[0] + 'interp-step-' + str(0) + '/explainability/' + args.val + '_atten_vec.npy')
dat_j = np.load(atten_dirs[-1] + 'interp-step-' + str(len(dirs[-1]) - 1) + '/explainability/' + args.val +
				'_atten_vec.npy')

n_clusters = 2
clusterer = KMeans(n_clusters=n_clusters, random_state=10)
cluster_labels_i = clusterer.fit_predict(dat_i)
cluster_labels_j = clusterer.fit_predict(dat_j)
ARI = [adjusted_rand_score(cluster_labels_i, cluster_labels_j)]

for atten_dir in atten_dirs:
	np.save(atten_dir + 'max-diff-ARI.npy', ARI)