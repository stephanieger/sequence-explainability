## packages to import
import numpy as np
import argparse
import sys
import tensorflow as tf
import random
random.seed(0)
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
import os
import time


## arguments
parser = argparse.ArgumentParser()
parser.add_argument("data", help="data set", type=str)
parser.add_argument("model_folder_1", help="where saved model is", type=str)
parser.add_argument("h5_1", help="best weights", type=str)
parser.add_argument("h5_2", help="best weights", type=str)
parser.add_argument("stepsize",  help="steps for path", type=float)  # 0.01
parser.add_argument("model", help='model - hier or avg', type=str)
parser.add_argument("model_str", help = 'for file saving', type=str)
args = parser.parse_args()

start_time = time.time()

## save directories
save_dir = '../interp-out/' + args.model_str + '/'
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
w_save_dir = save_dir + 'w_path/'
if not os.path.exists(w_save_dir):
	os.makedirs(w_save_dir)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

## arguments
_lambda = 1
stepsize = args.stepsize

# hyperparemeters to save:
hyp_to_save = ['stepsize']

with open(save_dir + 'hyperparameters.txt', 'w') as f:
	for hyp in hyp_to_save:
		f.write(hyp + ': ' + str(eval(hyp)) + '\n')

## build model
sys.path.append(args.model_folder_1)
from config import *

# get config file
if args.data == 'Amazon':
	Config = AmazonConfig()
	TRAIN_BATCHES = Config.TRAIN_BATCHES
elif args.data == 'Sentiment':
	Config = SentimentConfig()
	TRAIN_BATCHES = 100

## load in model

if args.model == 'hier':
	from seq2seq.seq2seqHier import Seq2seqModel, Seq2seqModelInterp
elif args.model == 'avg':
	from seq2seq.seq2seqAvg import Seq2seqModel, Seq2seqModelInterp

## load in data
idx = list(range(TRAIN_BATCHES))


def data_loader(idx):
	X_train = np.load(Config.DATA_FOLDER + 'x_train_' + str(idx) + '.npy')
	Y_train = np.load(Config.DATA_FOLDER + 'y_train_' + str(idx) + '.npy')
	decoder_input_data = np.zeros((len(Y_train), 1, Config.NUM_DECODER_TOKENS))

	if args.model == 'hier':
		F_train = np.load(Config.DATA_FOLDER + 'f_train_' + str(idx) + '.npy')
		E_train = np.load(Config.DATA_FOLDER + 'e_train_' + str(idx) + '.npy')
		inputs = [X_train, F_train, E_train, decoder_input_data]
	elif args.model == 'avg':
		inputs = [X_train, decoder_input_data]

	return inputs, Y_train

## get loss function
def getLoss(w_hat):
	model.set_weights(w_hat)
	l = 0
	n = 0

	for i in idx:
		# print(i)
		inputs, Y_train = data_loader(i)
		hist = model.evaluate(x=inputs, y=Y_train, batch_size=Config.BATCH_SIZE, verbose=0)
		l += hist[0] * len(Y_train)
		n += len(Y_train)

	return float(l) / float(n)

def dist(w_hat, w_target):
	w_hat_all = []
	w_target_all = []
	for i in range(len(w_hat)):
		w_hat_all += [w_hat[i].flatten()]
		w_target_all += [w_target[i].flatten()]
	w_hat_all = np.concatenate(w_hat_all)
	w_target_all = np.concatenate(w_target_all)
	d = np.linalg.norm(w_target_all - w_hat_all)

	return d

## compute loss for weight sets w_1 and w_2 and get max loss
seq2seqModel = Seq2seqModel()
model = seq2seqModel.createModel(Config)

## load in model weights and determine max loss
model.load_weights(args.h5_1)
weights_1 = model.get_weights()

model.load_weights(args.h5_2)
weights_2 = model.get_weights()

np.save(save_dir + 'weights_1.npy', weights_1)
np.save(save_dir + 'weights_2.npy', weights_2)

loss_1 = getLoss(weights_1)
loss_2 = getLoss(weights_2)
print('loss_1:', loss_1)
print('loss_2:', loss_2)

############################################# STRAIGHT LINE INTERPOLATION ##############################################

## interpolation function
t = 0
w_0 = weights_1
w_target = weights_2
w_hat = w_0
mu = 0.5
counter = 0

distanceVec = list()
lossVec = list()

while dist(w_hat, w_target) > 0:

	_dist = dist(w_hat, w_target)
	print('L2:', _dist)
	counter += 1
	print('counter:', counter)

	t += stepsize
	print('stepsize:', t)

	w_hat_new = list()
	for i in range(len(w_hat)):
		w_hat_new.append(np.array((1-t)*w_0[i] + t * w_target[i]))

	## compute loss
	loss = getLoss(w_hat_new)
	print('loss:', loss)

	w_hat = w_hat_new
	np.save(w_save_dir + 'w_hat_' + str(counter) + '.npy', w_hat)

	lossVec.append(loss)
	distanceVec.append(_dist)

	np.save(save_dir + 'loss.npy', lossVec)
	np.save(save_dir + 'distance.npy', distanceVec)

print('save losses now')

np.save(save_dir + 'loss.npy', lossVec)
np.save(save_dir + 'distance.npy', distanceVec)

end_time = time.time()
np.save(save_dir + 'time.npy', end_time-start_time)

print('terminated after {0} iterations'.format(counter)	)