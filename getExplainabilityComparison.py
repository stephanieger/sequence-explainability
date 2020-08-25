import numpy as np
import argparse
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import manifold

'''
Consider violin, tsne and cluster_stats for all explanations. This assumes Batching and Clustering has been done 
for all explainability models. Int Grad isn't computed for hierarchical attention, hence the flag. 
'''

parser = argparse.ArgumentParser()
parser.add_argument("model_folder", help="folder with saved model", type=str)
parser.add_argument("n_clusters", help="number of clusters", type=int)
parser.add_argument("val", help="validation set", type=str)
parser.add_argument("int_grad", help='do we have int_grad? 1 if yes, 0 otherwise', type=int)
args = parser.parse_args()

load_folder = args.model_folder + 'explainability/'

names = ['churn_atten', 'churn_grad', 'atten', 'grad']
if args.int_grad == 1:
	names += ['churn_int_grad', 'int_grad']

all_cluster_lab = []

for i in range(len(names)):
	print(names[i])

	# box plots
	print('compute box plots')
	idx = np.load(load_folder + args.val + '_' + names[i] + '_idx.npy')
	dat = np.load(load_folder + args.val + '_norm_' + names[i] + '_vec.npy')
	dat = np.squeeze(dat)
	cluster_dat = dat[:, idx]
	cluster_dat = abs(cluster_dat)
	cluster_dat = np.squeeze(cluster_dat)
	with open(load_folder + args.val + '_' + names[i] + '_event_desc.txt', 'r') as f:
		event_desc = f.readlines()

	for j in range(9):
		plt.figure()
		plt.boxplot(cluster_dat[:, j * 10:(j + 1) * 10])
		plt.ylim(0, 1)
		plt.xlabel('features in range ' + str(j * 10) + ' to ' + str((j + 1) * 10))
		plt.ylabel('data spread')
		plt.title('feature variations')
		plt.savefig(load_folder + args.val + '_' + names[i] + '_boxplot_feat_set_' + str(j) + '.png')
		plt.close()

	# violin plots
	print('compute violin plots')
	clusterer = KMeans(n_clusters=args.n_clusters, random_state=10)
	cluster_labels = clusterer.fit_predict(dat)

	cluster_size = []
	for i_ in range(args.n_clusters):
		cluster_size += [len(np.where(cluster_labels == i_)[0])]

	np.savetxt(load_folder + args.val + '_' + names[i] + '_cluster_size.csv', cluster_size, delimiter=',')

	cluster_size = []
	box_plot_all = []
	for i_ in range(args.n_clusters):

		cluster_n_samples = cluster_dat[cluster_labels == i_]
		cluster_size += [len(cluster_n_samples)]
		box_plot_feat = []

		for j in range(len(idx)):
			box_plot_feat += [cluster_n_samples[:, j]]

		for j in range(10):
			pos = list(range(10))
			plt.figure()
			plt.violinplot(box_plot_feat[j * 10:(j + 1) * 10], pos, widths=0.9, showmeans=True, showmedians=True,
						   showextrema=True, bw_method='silverman')
			plt.xlabel('features in range ' + str(j * 10) + ' to ' + str((j + 1) * 10))
			plt.ylabel('data spread')
			plt.ylim(0, 1)
			plt.title('feature variation in cluster' + str(i_))
			plt.savefig(
				load_folder + args.val + '_' + names[i] + '_violinplot_cluster_' + str(i_) + '_feat_set_' + str(
					j) + '.png')
			plt.close()

	cluster_size = np.array(cluster_size)
	np.save(load_folder + args.val + '_' + names[i] + '_cluster_size.npy', cluster_size)

	# TSNE plots
	print('t-SNE plots')

	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	tsne_dat = tsne.fit_transform(dat)
	np.save(load_folder + args.val + '_' + names[i]+ '_tsne.npy', tsne_dat)

	clusterer = KMeans(n_clusters=args.n_clusters, random_state=10)
	cluster_labels = clusterer.fit_predict(dat)
	all_cluster_lab += [cluster_labels]
	cluster_centers = clusterer.cluster_centers_
	np.save(load_folder + args.val + '_' + names[i] + '_centroids.csv', cluster_centers)

	print('fit tsne')
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	tsne_dat = tsne.fit_transform(np.concatenate([dat, cluster_centers], axis=0))
	print(tsne_dat.shape)
	cmap = plt.cm.jet
	bounds = np.linspace(0, args.n_clusters - 1, args.n_clusters)
	norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
	fig, ax = plt.subplots(1, 1, figsize=(6, 6))
	sc = ax.scatter(tsne_dat[:-args.n_clusters, 0], tsne_dat[:-args.n_clusters, 1], c=cluster_labels,
					cmap=cmap)  # [0:4000], cmap=cmap)
	sc1 = ax.scatter(tsne_dat[-args.n_clusters:, 0], tsne_dat[-args.n_clusters:, 1], c='k')
	txt = list(range(args.n_clusters))
	# annotate cluster labels
	for i_, x in enumerate(txt):
		ax.annotate(x, (tsne_dat[-args.n_clusters + i_, 0], tsne_dat[-args.n_clusters + i_, 1]))

	ax.set_title('TSNE plot')
	plt.savefig(load_folder + args.val + '_' + names[i] + '_tsne.png')
	plt.close()


