import numpy as np
import argparse
import sys
import matplotlib
matplotlib.use('Agg')
import os

'''
Get model predictions from trained endpoints for interpolations between two weights. 
'''

parser = argparse.ArgumentParser()
parser.add_argument("model_folder", help="folder with saved model", type=str)
parser.add_argument("weight_folder", help="folder with saved interp model", type=str)  # list of model folders in order"
parser.add_argument("val", help="validation set", type=str)
parser.add_argument("model", help='model - hier or avg', type=str)
parser.add_argument("data", help="dataset", type=str)
args = parser.parse_args()

# load in arguments
val = args.val

# get config file
sys.path.append(args.model_folder)
from config import *

if args.data == 'Amazon':
	Config = AmazonConfig()
	VAL_BATCHES = eval('Config.' + val.upper() + '_BATCHES')
elif args.data == 'Sentiment':
	Config = SentimentConfig()
	VAL_BATCHES = eval('Config.' + val.upper() + '_BATCHES')

# set number of batches
IDX = list(range(VAL_BATCHES))

## load in model
if args.model == 'hier':
	from seq2seq.seq2seqHier import Seq2seqModel
elif args.model == 'avg':
	from seq2seq.seq2seqAvg import Seq2seqModel

# set up data loader
def data_loader(idx):
	X_val = np.load(Config.DATA_FOLDER + 'x_' + str(val) + '_' + str(idx) + '.npy')
	Y_val = np.load(Config.DATA_FOLDER + 'y_' + str(val) + '_' + str(idx) + '.npy')
	decoder_input_data = np.zeros((len(Y_val), 1, Config.NUM_DECODER_TOKENS))

	if args.model == 'hier':
		F_val = np.load(Config.DATA_FOLDER + 'f_' + str(val) + '_' + str(idx) + '.npy')
		E_val = np.load(Config.DATA_FOLDER + 'e_' + str(val) + '_' + str(idx) + '.npy')
		inputs = [X_val, F_val, E_val, decoder_input_data]
	elif args.model == 'avg':
		inputs = [X_val, decoder_input_data]

	return inputs, Y_val

seq2seqModel = Seq2seqModel()
seq2seq_model = seq2seqModel.createModel(Config)

# define F1 score function
def compute_pred(w_hat, idx, _dir, model=seq2seq_model):
	model.set_weights(w_hat)
	print('weights:', _idx)

	churn_idx = []
	nochurn_idx = []

	y_val = []
	y_pred = []
	for i in IDX:
		print(i)
		inputs, y = data_loader(i)
		y_pred += [model.predict(inputs, batch_size=Config.BATCH_SIZE)]
		y_val += [y]

	y_val = np.concatenate(y_val, axis=0)
	y_pred = np.concatenate(y_pred, axis=0)

	lab = np.argmax(y_val, axis=2)
	lab = lab[:, 0]

	for i, l in enumerate(lab):
		if l == 1:
			churn_idx += [i]
		else:
			nochurn_idx += [i]

	prd = np.argmax(y_pred, axis=-1)
	act = np.argmax(y_val, axis=-1)

	np.save(_dir + 'prd_' + str(idx) + '.npy', prd)
	np.save(_dir + 'act_' + str(idx) + '.npy', act)
	np.save(_dir + 'churn_idx.npy', churn_idx)
	np.save(_dir + 'nochurn_idx.npy', nochurn_idx	)
	return


weight_dirs = args.weight_folder.split(',')

save_folder = 'endpoint-pred/'
if not os.path.exists(save_folder):
	os.makedirs(save_folder)

w_1 = np.load(weight_dirs[0] + 'weights_1.npy')
w_2 = np.load(weight_dirs[0] + 'weights_2.npy')

compute_pred(w_1, 1, save_folder)
compute_pred(w_2, 2, save_folder)




