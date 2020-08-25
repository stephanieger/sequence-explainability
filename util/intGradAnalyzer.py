import numpy as np
import matplotlib
matplotlib.use('Agg')
import keras.backend as K
from scipy.sparse import load_npz
import os
import math

class AvgGrad:

	def getGrad(self, _model, inputs, Y_train):
		grad = K.gradients(_model.total_loss, _model.inputs[0])

		# model inputs
		model_inputs = [x for x in _model.inputs]
		model_inputs += [_model.targets[0]]
		model_inputs += [_model.sample_weights[0]]
		model_inputs += [K.learning_phase()]

		inputs_ = [x for x in inputs]
		inputs_ += [Y_train]
		inputs_ += [[1]]
		inputs_ += [1]

		grad_fn = K.function(inputs=model_inputs, outputs=grad)
		grad = grad_fn(inputs_)

		return grad

	def grad_and_prob_func(self, Config, _model):
		def grad_and_prob(x, y):
			inputs = [x, np.zeros((len(y), 1, Config.NUM_DECODER_TOKENS))]
			grad = self.getGrad(_model, inputs, y)
			prob = _model.predict(inputs)
			return prob, grad

		return grad_and_prob

	def integrated_gradients(
			self,
			inp,
			target_label_index,
			predictions_and_gradients,
			baseline,
			steps=50):

		# inp = np.squeeze(inp, axis=0)
		# print(inp.shape)
		# if baseline is None:
		# 	baseline = 0 * inp
		# assert (baseline.shape == inp.shape)

		# Scale input and compute gradients. do this for each element in the batch
		scaled_inputs = [np.array([baseline + (float(i) / steps) * (inp[k] - baseline) for i in range(0, steps + 1)])
						 for k in range(len(inp))]
		scaled_inputs = np.array(scaled_inputs)
		# need to reshape to predict gradients/predictions
		scaled_shape = scaled_inputs.shape
		scaled_inputs = np.reshape(scaled_inputs, (scaled_shape[0]*scaled_shape[1], scaled_shape[2], scaled_shape[3]))
		#     print(scaled_inputs.shape)
		#     print(target_label_index.shape)
		predictions, grads = predictions_and_gradients(scaled_inputs,
													   target_label_index)  # shapes: <steps+1>, <steps+1, inp.shape>
		# reshape to original shape
		grads = np.reshape(grads, scaled_shape)
		print(grads.shape)

		# Use trapezoidal rule to approximate the integral.
		# See Section 4 of the following paper for an accuracy comparison between
		# left, right, and trapezoidal IG approximations:
		# "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
		# https://arxiv.org/abs/1908.06214
		# grads = grads[0]
		grads = np.array([(grads[i][:-1] + grads[i][1:]) / 2.0 for i in range(len(grads))])
		print(grads.shape)
		avg_grads = np.average(grads, axis=1)
		integrated_gradients = (inp - baseline) * avg_grads  # shape: <inp.shape>
		integrated_gradients = np.sum(integrated_gradients, axis=-1)
		return integrated_gradients, predictions

	def runGrad(self, Config, event2vec_lstm_model, folder, val_sets, start, steps):

		save_folder = folder + 'int-grad/'
		if not os.path.exists(save_folder):
			os.makedirs(save_folder)

		baseline_vec = np.load(Config.DATA_FOLDER + 'unk_vec.npy')
		# baseline_vec = np.expand_dims(baseline_vec, axis=0)
		baseline_vec= np.expand_dims(baseline_vec, axis=0)

		print(baseline_vec.shape)

		batchsize = 50 #Config.BATCH_SIZE

		# val_sets = Config.DATASETS
		vs_dict = Config.DATASETS_DICT

		# get gradients
		vs = val_sets
		x_base = 'Config.X_' + vs.upper()
		y_base = 'Config.Y_' + vs.upper()

		grad_prob = self.grad_and_prob_func(Config, event2vec_lstm_model)

		for i in range(start, vs_dict[vs]):
			print(i)
			X_val = np.load(eval(x_base) + str(i) + '.npy')
			Y_val = np.load(eval(y_base) + str(i) + '.npy')

			baseline = np.repeat(baseline_vec, X_val.shape[1], 0)


			# saver for gradient data
			gradients = []

			if len(Y_val) < batchsize:
				D = X_val
				L = Y_val
				L = np.repeat(L, steps + 1, 0)
				print(D.shape)
				int_grad, pred = self.integrated_gradients(D, L, grad_prob, baseline)
				gradients += [int_grad]
			else:
				for j in range(int(math.ceil(len(Y_val) / float(batchsize)))):
					if len(Y_val[j * batchsize:]) < batchsize:
						D = X_val[j * batchsize:]
						L = Y_val[j * batchsize:]
					else:
						D = X_val[j * batchsize: (j + 1) * batchsize]
						L = Y_val[j * batchsize: (j + 1) * batchsize]
					# D = np.expand_dims(X_val[j], axis=0)
					# L = np.expand_dims(Y_val[j], axis=0)
					L = np.repeat(L, steps + 1, 0)
					print(D.shape)
					int_grad, pred = self.integrated_gradients(D, L, grad_prob, baseline)
					gradients += [int_grad]

			gradients = np.concatenate(gradients)
			np.save(save_folder + vs + '_grad_batch_' + str(i) + '.npy', gradients)

