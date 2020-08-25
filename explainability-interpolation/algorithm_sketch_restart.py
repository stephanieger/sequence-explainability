## packages to import
import numpy as np
import argparse
import sys
import keras.backend as K
import tensorflow as tf
import random
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
random.seed(0)
import os
import time



## arguments
parser = argparse.ArgumentParser()
parser.add_argument("data", help="data set", type=str)
parser.add_argument("model_folder_1", help="where saved model is", type=str)
parser.add_argument("h5_1", help="best weights", type=str)
parser.add_argument("h5_2", help="best weights", type=str)
parser.add_argument("eps",  help="loss epsilon", type=float)  # 0.01	x
parser.add_argument("weight_eps", help='weight epsilon', type=float)  # 0.01
parser.add_argument("stepsize",  help="steps for path", type=float)  # 0.01
parser.add_argument("model", help='model - hier or avg', type=str)
parser.add_argument("model_str", help = 'for file saving', type=str)
parser.add_argument("w_1", help="numpy array of stuck weights", type=str)
parser.add_argument("--con_eps", help="constraint size", default=0.5, type=float)
parser.add_argument("--max_out", help="max outer iteration", default=20, type=int)
parser.add_argument("--max_in", help="max inner iteration", default=2000, type=int)
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
max_iter = args.max_out
conv = 1e-5
alpha = 0.5
tau = 0.8
num_grad = 1
learning_rate = 5e-2
lam_learning_rate = 5e-1
max_inner_iter = args.max_in
min_alpha = 1e-5 #1e-7
constraint_eps = args.con_eps
constraint_dist = 5
stepsize = args.stepsize
weight_eps = args.weight_eps
loss_eps = args.eps

# hyperparemeters to save:
hyp_to_save = ['_lambda', 'max_iter', 'max_inner_iter', 'conv', 'alpha', 'tau', 'num_grad', 'learning_rate',
			   'lam_learning_rate', 'min_alpha', 'constraint_dist', 'constraint_eps', 'stepsize', 'weight_eps',
			   'loss_eps']

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

## compute loss for weight sets w_1 and w_2 and get max loss
seq2seqModel = Seq2seqModel()
seq2seqInterp = Seq2seqModelInterp()
model = seq2seqModel.createModel(Config)
model_1 = seq2seqModel.createModel(Config)
model_2 = seq2seqModel.createModel(Config)
opt_model = seq2seqModel.createModel(Config)
grad_model = seq2seqModel.createModel(Config)

## load in model weights and determine max loss
weights_1 = np.load(args.w_1)
model_1.load_weights(args.h5_1)
model_2.load_weights(args.h5_2)

# weights_1 = model_1.get_weights()
weights_2 = model_2.get_weights()
tmp_weights_1 = model_1.get_weights()

np.save(save_dir + 'weights_1.npy', weights_1)
np.save(save_dir + 'weights_2.npy', weights_2)

loss_1 = getLoss(tmp_weights_1)
loss_2 = getLoss(weights_2)

max_loss = max(loss_1, loss_2) + args.eps
model_1.set_weights(weights_1)
print('max loss:', max_loss)

## define distance functions
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

def distL1(w_hat, w_target):
	w_hat_all = []
	w_target_all = []
	for i in range(len(w_hat)):
		w_hat_all += [w_hat[i].flatten()]
		w_target_all += [w_target[i].flatten()]
	w_hat_all = np.concatenate(w_hat_all)
	w_target_all = np.concatenate(w_target_all)
	d = np.linalg.norm(w_target_all - w_hat_all, ord=1)

	return d

def distLInf(w_hat, w_target):
	w_hat_all = []
	w_target_all = []
	for i in range(len(w_hat)):
		w_hat_all += [w_hat[i].flatten()]
		w_target_all += [w_target[i].flatten()]
	w_hat_all = np.concatenate(w_hat_all)
	w_target_all = np.concatenate(w_target_all)
	d = np.linalg.norm(w_target_all - w_hat_all, ord=np.inf)

	return d

## define gradient function
def getGrad(w_hat, idx, grad_model=grad_model):
	inputs, Y_train = data_loader(idx)

	if len(Y_train) > Config.BATCH_SIZE:
		Y_train = Y_train[:Config.BATCH_SIZE]
		for _i in range(len(inputs)):
			inputs[_i] = inputs[_i][:Config.BATCH_SIZE]

	grad_model.set_weights(w_hat)
	tmp = []
	for layer in grad_model.layers:
		if len(layer.weights) > 0:
			tmp.append(layer.weights)
	weights = []
	for sublist in tmp:
		for item in sublist:
			weights.append(item)

	gradients = model.optimizer.get_gradients(grad_model.total_loss, weights)

	model_inputs = [x for x in grad_model.inputs]
	model_inputs += [grad_model.sample_weights[0]]
	model_inputs += [grad_model.targets[0]]
	model_inputs += [K.learning_phase()]

	inputs_ = [x for x in inputs]
	inputs_ += [[1]]
	inputs_ += [Y_train]
	inputs_ += [1]

	grad_fn = K.function(inputs=model_inputs, outputs=gradients)
	grad = grad_fn(inputs_)

	return grad

#################################################### BEGIN OPTIMIZATION SECTION ########################################
## Define loss function
def optLoss(w_target, w_hat, _dir, grad=None):

	num = 0
	den1 = 0
	den2 = 0
	for i in range(len(grad)):
		w_t = w_target[i]
		if len(w_t.shape) == 1:
			w_h = w_hat[i]
			_d = K.expand_dims(_dir[i], axis=-1)
			den1 += tf.reduce_sum(tf.square(w_t - w_h))
			den2 += tf.reduce_sum(tf.square(_dir[i]))
			num += K.dot(K.transpose(tf.expand_dims(w_t - w_h, axis=-1)), _d)
		else:
			w_h = K.flatten(w_hat[i])
			w_t = K.flatten(w_t)
			_d = K.expand_dims(K.flatten(_dir[i]), axis=-1)
			den1 += tf.reduce_sum(tf.square(w_t - w_h))
			den2 += tf.reduce_sum(tf.square(K.flatten(_dir[i])))
			num += K.dot(K.transpose(tf.expand_dims(w_t - w_h, axis=-1)), _d)
	den = tf.sqrt(den2 + tf.keras.backend.epsilon())*tf.sqrt(den1+ tf.keras.backend.epsilon())
	_div = tf.divide(num, den + tf.keras.backend.epsilon())
	loss = tf.negative(_div)
	return loss


## define constraint
def constraint(_dir, grad):
	c = 0
	d_all = []
	for i in range(len(grad)):
		if len(grad[i].shape) == 1:
			c += np.inner(grad[i], _dir[i])
			d_all += [_dir[i]]
		else:
			_d =_dir[i].flatten()
			_g = grad[i].flatten()
			c += np.inner(_g, _d)
			d_all += [_d]
	d_norm = np.linalg.norm(np.concatenate(d_all))
	return c/d_norm

def normalize_dir(_dir):
	dir_all = []
	for i in range(len(_dir)):
		dir_all += [_dir[i].flatten()]
	dir_all = np.concatenate(dir_all)
	n = np.linalg.norm(dir_all)
	norm_dir = []
	for _d in _dir:
		norm_dir += [_d / n]
	return norm_dir

## get list of weights
opt_weights = normalize_dir(weights_1)
opt_model.set_weights(opt_weights)

tmp = list()
for layer in opt_model.layers:
	if len(layer.weights) > 0:
		tmp.append(layer.weights)
weights = []
for sublist in tmp:
	for item in sublist:
		weights.append(item)

w_target_tf = [tf.placeholder("float", shape=(weights_1[i].shape)) for i in range(len(weights_1))]
w_hat_tf = [tf.placeholder("float", shape=(weights_1[i].shape)) for i in range(len(weights_1))]
grad_tf = [tf.placeholder("float", shape=(weights_1[i].shape)) for i in range(len(weights_1))]
_lambda_tf = tf.placeholder("float", None)

opt = tf.train.AdamOptimizer()
loss_model = optLoss(w_target_tf, w_hat_tf, weights, grad=grad_tf)

# symbolic gradient
gradient_op = opt.compute_gradients(loss_model, weights)
grads_and_vars = [(tf.add(gv[0], _lambda_tf * grad_tf[i]), gv[1]) for i, gv in enumerate(gradient_op)]
train_op = opt.apply_gradients(grads_and_vars)
num_weights = len(weights_1)

## define optimization routine
def opt(w_target, w_hat, opt_model, con_eps, _lambda=_lambda, learning_rate = learning_rate,
		lam_learning_rate=lam_learning_rate, learning_decay=.95):

	opt_model.set_weights(w_hat)
	_idx = np.random.randint(0, TRAIN_BATCHES)
	grad = getGrad(w_hat, _idx)
	loss_old = 10000
	loss_new = np.squeeze(K.eval(optLoss(w_target, w_hat, weights, grad=grad)))

	iter = 0

	feed_dictionary = {}
	for i in range(num_weights):
		feed_dictionary[w_target_tf[i]] = w_target[i]
	for i in range(num_weights):
		feed_dictionary[w_hat_tf[i]] = w_hat[i]
	for i in range(num_weights):
		feed_dictionary[grad_tf[i]] = grad[i]
	feed_dictionary[_lambda_tf] = _lambda

	print('loss:', loss_new)
	grad_list = list()
	grad_list.append(grad)
	w_hat_list = list()
	lam_list = list()
	loss_list = list()
	con_list = list()

	while abs(loss_old - loss_new) > conv:

		if iter == max_iter:
			break

		feed_dictionary[_lambda_tf] = _lambda
		_idx = np.random.randint(0, TRAIN_BATCHES)
		grad = getGrad(w_hat, _idx)
		for i in range(num_weights):
			feed_dictionary[grad_tf[i]] = grad[i]
		for _ in range(max_inner_iter):
			sess.run(train_op, feed_dict=feed_dictionary)

		# # decrease the learning rate
		loss_old = np.copy(loss_new)

		_w_hat_new = opt_model.get_weights()
		_lambda = _lambda + lam_learning_rate * (constraint(_w_hat_new, grad)+con_eps)
		if _lambda < 0:
			_lambda = 0
		con = constraint(_w_hat_new, grad)
		print('constraint:', con)

		loss_new = np.squeeze(K.eval(optLoss(w_target, w_hat, weights, grad=grad)))
		print(loss_new)

		lam_list.append(_lambda)
		loss_list.append(loss_new)
		con_list.append(con)

		iter += 1
		lam_learning_rate *= learning_decay

	if con_list[-1] > 0:
		while con > 0:
			feed_dictionary[_lambda_tf] = _lambda
			for _ in range(max_inner_iter):
				sess.run(train_op, feed_dict=feed_dictionary)

			# # decrease the learning rate
			_w_hat_new = opt_model.get_weights()
			_lambda = _lambda + lam_learning_rate * (constraint(_w_hat_new, grad) + con_eps)

			if _lambda < 0:
				_lambda = 0
			con = constraint(_w_hat_new, grad)
			con_list.append(con)

	grad_op = sess.run(grads_and_vars, feed_dict=feed_dictionary)
	opt_grad = [gv[0] for gv in grad_op]
	opt_grad_norm = gradNorm(opt_grad)
	print('opt objective loss:', loss_new)
	print('num iter:', iter)
	print('constraint:', constraint(_w_hat_new, grad))
	print('lambda:', _lambda)
	direction = _w_hat_new
	np.save(save_dir + 'w_hat_list.npy', w_hat_list)

	return direction, lam_list, loss_list, con_list, opt_grad_norm

def optGrad(w_hat, w_target, opt_model, _lambda=_lambda):

	opt_model.set_weights(w_hat)
	grad = getGrad(w_hat, idx[0])

	feed_dictionary = {}
	for i in range(num_weights):
		feed_dictionary[w_target_tf[i]] = w_target[i]
	for i in range(num_weights):
		feed_dictionary[w_hat_tf[i]] = w_hat[i]
	for i in range(num_weights):
		feed_dictionary[grad_tf[i]] = grad[i]
	feed_dictionary[_lambda_tf] = _lambda
	grad_op = sess.run(gradient_op, feed_dict=feed_dictionary)
	grads = [gv[0] for gv in grad_op]

	return grads

## define function needed for line search algorithm
def linesearch(_dir, grad):
	p = 0
	for i in range(len(grad)):
		if len(grad[i].shape) == 1:
			p += np.inner(_dir[i], grad[i])
		else:
			_d = _dir[i].flatten()
			_g = grad[i].flatten()
			p += np.inner(_d, _g)
	return p

def gradNorm(_dir):
	dir_all = []
	for i in range(len(_dir)):
		dir_all += [_dir[i].flatten()]
	dir_all = np.concatenate(dir_all)
	n = np.linalg.norm(dir_all)
	return n

## interpolation function

t = 0
w_0 = weights_1
w_target = weights_2
w_hat = w_0

mu = 0.
counter = 0

distanceVec = list()
distanceInitVec = list()
distanceVecL1 = list()
distanceVecLInf = list()
lossVec = list()
gradNormVec = list()
stepSizeVec = list()
w_path = []

while dist(w_hat, w_target) > weight_eps:

	if dist(w_hat, w_target) < constraint_dist:
		constraint_eps = 0.

	print('L2:', dist(w_hat, w_target))
	print('L1:', distL1(w_hat, w_target))
	print('LInf:', distLInf(w_hat, w_target))
	counter += 1
	print('counter:', counter)

	# _stepsize = min(_t, stepsize)
	t += stepsize
	print('stepsize:', t)
	# w_proposed = w_0 + t * w_target


	w_proposed = list()
	for i in range(len(w_hat)):
		w_proposed.append(np.array((1-t)*w_0[i] + t * w_target[i]))

	## compute loss
	loss = getLoss(w_proposed)
	print('loss:', loss)
	if loss <= max_loss:
		# accept step
		w_hat = w_proposed
		np.save(w_save_dir + 'w_hat_' + str(counter) + '.npy', w_hat)

		loss_tmp = getLoss(w_proposed)
		dist_tmp = dist(w_proposed, w_target)

	else:
		## enter optimization routine
		print('opt routine')

		_dir, lam_list, loss_list, con_list, opt_grad = opt(w_target, w_hat, opt_model, constraint_eps, _lambda=_lambda,
															learning_rate=learning_rate,
															lam_learning_rate=lam_learning_rate)
		_dir = normalize_dir(_dir)

		## take step in proposed direction
		_alpha = alpha
		w_proposed = list()
		for i in range(len(w_hat)):
			w_proposed.append(np.array(w_hat[i] + _alpha * _dir[i]))

		l_tmp = getLoss(w_proposed)
		idx_ = random.sample(idx, num_grad)
		l_accept = getLoss(w_hat)
		print('w_hat loss:', l_accept)
		print('pick alpha')
		linesearchLoss = list()
		linesearchAlpha = list()

		linesearchAlpha.append(_alpha)
		linesearchLoss.append(l_tmp)

		tmp_c = 0
		while l_tmp > max_loss:
			tmp_c += 1
			_alpha = tau * _alpha
			w_proposed = list()
			for i in range(len(w_hat)):
				w_proposed.append(np.array(w_hat[i] + _alpha * _dir[i]))
			l_tmp = getLoss(w_proposed)
			linesearchAlpha.append(_alpha)
			linesearchLoss.append(l_tmp)
			if _alpha < min_alpha:
				_alpha = min_alpha
				break
		print('final:')
		# print('mu*t', _alpha * t)
		# print(l_accept - l_tmp)
		print('accepted alpha:', _alpha)
		print('loss:', getLoss(w_proposed))
		print('distance:', dist(w_proposed, w_target))
		loss_tmp = getLoss(w_proposed)
		dist_tmp = dist(w_proposed, w_target)

		# save grad norm and step size
		stepSizeVec.append(_alpha)
		gradNormVec.append(opt_grad)

		# accept step
		w_hat = w_proposed
		w_0 = w_hat
		t = 0
		np.save(w_save_dir + 'w_hat_' + str(counter) + '.npy', w_hat)
		np.save(w_save_dir + 'lam_list_' + str(counter) + '.npy', lam_list)
		np.save(w_save_dir + 'loss_list_' + str(counter) + '.npy', loss_list)
		np.save(w_save_dir + 'con_list_' + str(counter) + '.npy', con_list)
		np.save(w_save_dir + 'linesearch_alpha_' + str(counter) + '.npy', linesearchAlpha)
		np.save(w_save_dir + 'linesearch_loss_' + str(counter) + '.npy', linesearchLoss)
		np.save(w_save_dir + 'direction_vec_' + str(counter) + '.npy', _dir)

		end_time = time.time()
		np.save(save_dir + 'time.npy', end_time - start_time)

	lossVec.append(loss_tmp)
	distanceVec.append(dist_tmp)
	distanceInitVec.append(dist(w_hat, w_0))
	distanceVecL1.append(distL1(w_proposed, w_target))
	distanceVecLInf.append((distLInf(w_proposed, w_target)))

	np.save(save_dir + 'loss.npy', lossVec)
	np.save(save_dir + 'distance.npy', distanceVec)
	np.save(save_dir + 'l1_distance.npy', distanceVecL1)
	np.save(save_dir + 'linf_distance.npy', distanceVecLInf)
	np.save(save_dir + 'init_distance.npy', distanceInitVec)
	np.save(save_dir + 'w_path.npy', w_path)
	np.save(save_dir + 'linesearch_step.npy', stepSizeVec)
	np.save(save_dir + 'gradnorm_vec.npy', gradNormVec)
print('save losses now')

np.save(save_dir + 'loss.npy', lossVec)
np.save(save_dir + 'distance.npy', distanceVec)
np.save(save_dir + 'l1_distance.npy', distanceVecL1)
np.save(save_dir + 'linf_distance.npy', distanceVecLInf)
np.save(save_dir + 'init_distance.npy', distanceInitVec)
np.save(save_dir + 'w_path.npy', w_path)
np.save(save_dir + 'linesearch_step.npy', stepSizeVec)
np.save(save_dir + 'gradnorm_vec.npy', gradNormVec)

print('terminated after {0} iterations'.format(counter)	)
