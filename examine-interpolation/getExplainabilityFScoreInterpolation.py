import numpy as np
import argparse
import sys
import matplotlib
matplotlib.use('Agg')
import glob
from sklearn.metrics import f1_score

'''
Compute F1-score along interpolation path (either for straight line or for valley-tracking low-loss path).
'''

parser = argparse.ArgumentParser()
parser.add_argument("model_folder", help="folder with saved model", type=str)
parser.add_argument("weight_folder", help="folder with saved interp model", type=str)  # list of model folders in order"
parser.add_argument("val", help="validation set", type=str)
parser.add_argument("save_folder", help="where to save outputs", type=str)
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
def compute_f1(w_hat, model=seq2seq_model):
	model.set_weights(w_hat)

	y_val = []
	y_pred = []
	for i in IDX:
		inputs, y = data_loader(i)
		y_pred += [model.predict(inputs, batch_size=Config.BATCH_SIZE)]
		y_val += [y]

	y_val = np.concatenate(y_val, axis=0)
	y_pred = np.concatenate(y_pred, axis=0)

	prd = np.argmax(y_pred, axis=-1)
	act = np.argmax(y_val, axis=-1)

	return f1_score(act, prd)


weight_dirs = args.weight_folder.split(',')
dirs = []
for weight_dir in weight_dirs:
	dirs += [glob.glob(weight_dir + 'w_path/w_hat*')]

f1 = []

w_init = np.load(weight_dirs[0] + 'weights_1.npy')
f1 += [compute_f1(w_init)]

for i, weight_dir in enumerate(weight_dirs):
	num_steps = len(dirs[i])
	print('num steps:', num_steps)

	for j in range(1, num_steps):
		weights = np.load(weight_dir + 'w_path/w_hat_' + str(j) + '.npy')
		f1 += [compute_f1(weights)]
		print(f1[-1])

f1 = np.array(f1)
np.save(args.save_folder + 'interp-f1.npy', f1)




