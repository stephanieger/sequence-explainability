import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
from sklearn.cluster import KMeans
import os

'''
Plot violin plots for each cluster for top 10 attentions along valley tracking low-loss path. 
'''

parser = argparse.ArgumentParser()
parser.add_argument("hyp_dict", help='hypernym dictionary', type='str')
parser.add_argument("atten_folders", help='folder with attentions, in order and comma separated', type='str')
parser.add_argument("save_folder", help='folder to save plots', type=str)
args = parser.parse_args()

with open(args.hyp_dict, 'rb') as f:
	_dict = pickle.load(f)

if not os.path.exists(args.save_folder):
	os.makedirs(args.save_folder)

key = list(_dict.keys())

atten_dirs = args.atten_folder.split(',')
dirs = []
for atten_dir in atten_dirs:
	dirs += [glob.glob(atten_dir + 'interp-step-*')]

vec = np.load(atten_dir + 'interp-step-0/explainability/' + args.val + '_atten_vec.npy')
clusterer = KMeans(n_clusters=2)
lab = clusterer.fit_predict(vec)

idx = []
idx += np.where(lab == 0)[0]
idx += np.where(lab == 1)[0]

VEC = []
VEC += [vec[idx[0]]]
VEC += [vec[idx[1]]]

mean_vec = []
std_vec = []
mean_vec += [np.mean(VEC[0], axis=0)]
std_vec += [np.std(VEC[0], axis=0)]
mean_vec += [np.mean(VEC[1], axis=0)]
std_vec += [np.std(VEC[1], axis=0)]

N = 10
IDX = []
IDX += [np.argsort(-mean_vec[0])]
IDX += [np.argsort(-mean_vec[1])]

keys = [key[i] for i in idx][:N]

plt.figure()
plt.violinplot(VEC[0][:N,:].transpose(), list(range(N)), widths=0.9, showmeans=True, showmedians=True, showextrema=True,
			   bw_method='silverman')
plt.ylim(0,1.1)
plt.title('Violin Plots for top ' + str(N) + ' Hypernyms at Step: 0 for Cluster 0')
plt.savefig(args.save_folder + 'cluster-0-0.png')

plt.figure()
plt.violinplot(VEC[1][:N,:].transpose(), list(range(N)), widths=0.9, showmeans=True, showmedians=True, showextrema=True,
			   bw_method='silverman')
plt.ylim(0,1.1)
plt.title('Violin Plots for top ' + str(N) + ' Hypernyms at Step: 0 for Cluster 1')
plt.savefig(args.save_folder + 'cluster-1-0.png')

cnt = 0
for i, atten_dir in enumerate(atten_dirs):
	num_steps = len(dirs[i])
	print(num_steps)
	for j in range(1, num_steps):
		cnt += 1

		vec = np.load(atten_dir + 'interp-step-' + str(j) + '/explainability/' + args.val + '_atten_vec.npy')

		clusterer = KMeans(n_clusters=2)
		lab = clusterer.fit_predict(vec)

		idx = []
		idx += np.where(lab == 0)[0]
		idx += np.where(lab == 1)[0]

		VEC = []
		VEC += [vec[idx[0]]]
		VEC += [vec[idx[1]]]

		mean_vec = []
		std_vec = []
		mean_vec += [np.mean(VEC[0], axis=0)]
		std_vec += [np.std(VEC[0], axis=0)]
		mean_vec += [np.mean(VEC[1], axis=0)]
		std_vec += [np.std(VEC[1], axis=0)]

		N = 10
		IDX = []
		IDX += [np.argsort(-mean_vec[0])]
		IDX += [np.argsort(-mean_vec[1])]

		keys = [key[i] for i in idx][:N]

		plt.figure()
		plt.violinplot(VEC[0][:N, :].transpose(), list(range(N)), widths=0.9, showmeans=True, showmedians=True,
					   showextrema=True, bw_method='silverman')
		plt.ylim(0, 1.1)
		plt.title('Violin Plots for top ' + str(N) + ' Hypernyms at Step: ' + str(cnt) + ' for Cluster 0')
		plt.savefig(args.save_folder + 'cluster-0-' + str(cnt) + '.png')

		plt.figure()
		plt.violinplot(VEC[1][:N, :].transpose(), list(range(N)), widths=0.9, showmeans=True, showmedians=True,
					   showextrema=True, bw_method='silverman')
		plt.ylim(0, 1.1)
		plt.title('Violin Plots for top ' + str(N) + ' Hypernyms at Step: ' + str(cnt) + ' for Cluster 1')
		plt.savefig(args.save_folder + 'cluster-1-' + str(cnt) + '.png')

