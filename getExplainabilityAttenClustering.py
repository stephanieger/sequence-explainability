import numpy as np
import argparse
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

'''
This assumes getExplainabilityAttenBatches.py has been run, get explanation vector for all data as well as positive 
reviews. 
'''

parser = argparse.ArgumentParser()
parser.add_argument("model_folder", help="folder with saved model", type=str)
parser.add_argument("val", help="validation set", type=str)
parser.add_argument("data", help="which dataset", type=str)
args = parser.parse_args()

save_folder = args.model_folder + 'explainability/'

if not os.path.exists(save_folder):
	os.makedirs(save_folder)

# load in config file for associated
sys.path.append(args.model_folder)
from config import *
# get config file
if args.data == 'Amazon':
	Config = AmazonConfig()
elif args.data == 'Sentiment':
	Config = SentimentConfig()

# types of validation sets
VAL_BATCHES = Config.VAL_BATCHES
TEST_BATCHES = Config.TEST_BATCHES

vs_batches = [VAL_BATCHES, TEST_BATCHES]
val_sets = Config.DATASETS
vs_dict = Config.DATASETS_DICT

num_batches = vs_dict[args.val]
N = Config.ATTN_N

x_base = 'Config.X_' + args.val.upper()
y_base = 'Config.Y_' + args.val.upper()
atten_base = args.model_folder + 'atten/' + 'X_' + args.val + '_atten_batch_'

atten_vec = []
churn_atten_vec = []

for i in range(vs_dict[args.val]):
	x_val = np.load(eval(x_base)+str(i)+'.npy')
	y_val = np.load(eval(y_base)+str(i)+'.npy')

	atten = np.load(atten_base + str(i)+'.npy')

	lab = np.argmax(y_val, axis=2)
	lab = lab[:, 0]

	for policy_id in range(len(x_val)):
		atten_vec += [atten[policy_id]]
		# Consider churn policies separately
		if lab[policy_id] != 0:
			churn_atten_vec += [atten[policy_id]]

# turn everything back into arrays
churn_atten_vec = np.array(churn_atten_vec)
atten_vec = np.array(atten_vec)

# save arrays
np.save(save_folder + args.val + '_churn_atten_vec.npy', churn_atten_vec)
np.save(save_folder + args.val + '_atten_vec.npy', atten_vec)

# get mean and standard deviation for each set
vectors = ['churn_atten_vec', 'atten_vec']
names = ['churn_atten', 'atten']

for i in range(len(vectors)):
	print(names[i])
	dat = eval(vectors[i])
	mean_ = np.mean(dat, axis=0)
	std_ = np.std(dat, axis=0)
	idx = np.argsort(-abs(mean_), axis=0)
	dat = np.squeeze(dat)

	# save mean and standard deviation
	np.save(save_folder + args.val +'_' + names[i] + '_mean.npy', mean_)
	np.save(save_folder + args.val + '_' + names[i] + '_std.npy', std_)

	# normalize data and save
	scaler = StandardScaler()
	norm_dat = scaler.fit_transform(dat)
	np.save(save_folder + args.val + '_norm_' + names[i] + '_vec.npy', norm_dat)

	# do cluster analysis
	score = []
	sil_avg = []
	range_n_clusters = range(2,20)

	norm_dat = np.squeeze(norm_dat)

	print('determine number of clusters')
	for n_clusters in range_n_clusters:
		# clustering
		clusterer = KMeans(n_clusters=n_clusters, random_state=10)
		cluster_labels = clusterer.fit_predict(dat)
		silhouette_avg = silhouette_score(dat, cluster_labels)
		sample_silhouette_values = silhouette_samples(dat, cluster_labels)
		score += [clusterer.score(dat)]
		sil_avg += [silhouette_avg]

	# plot the score
	plt.plot(np.array(range_n_clusters), score)
	plt.xlabel('number of clusters')
	plt.ylabel('explained variance')
	plt.title('Scree Plot')
	plt.savefig(save_folder + args.val + '_' + names[i] + '_scree.png')
	plt.close()

	# plot the silhouette avg
	plt.plot(np.array(range_n_clusters), sil_avg)
	plt.xlabel('Number of Clusters')
	plt.ylabel('Silhouette Average')
	plt.title('Silhouette Score for N Clusters')
	plt.savefig(save_folder + args.val + '_' + names[i] + '_sil_avg.png')
	plt.close()
