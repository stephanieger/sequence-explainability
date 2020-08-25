import numpy as np
import matplotlib
matplotlib.use('Agg')
import keras.backend as K
from scipy.sparse import load_npz
import os

class AvgGrad:
	def runGrad(self, Config, event2vec_lstm_model, folder, val_sets, start):

		save_folder = folder + 'grad/'
		if not os.path.exists(save_folder):
			os.makedirs(save_folder)

		batchsize = Config.BATCH_SIZE

		# val_sets = Config.DATASETS
		vs_dict = Config.DATASETS_DICT

		# get gradients
		vs = val_sets
		x_base = 'Config.X_' + vs.upper()
		y_base = 'Config.Y_' + vs.upper()

		# saver for gradient data
		gradients = []

		for i in range(start, vs_dict[vs]):
			print(i)
			X_val = np.load(eval(x_base) + str(i) + '.npy')
			Y_val = np.load(eval(y_base) + str(i) + '.npy')

			decoder_input_data = np.zeros((len(Y_val), 1,
										   Config.NUM_DECODER_TOKENS))
			inputs = [X_val, decoder_input_data]

			# initialize empty list to store gradients from a single seq_len
			seq_gradients = []

			# build gradient function
			grad = K.gradients(event2vec_lstm_model.total_loss, event2vec_lstm_model.inputs[0])

			# model inputs
			model_inputs = [x for x in event2vec_lstm_model.inputs]
			model_inputs += [event2vec_lstm_model.targets[0]]
			model_inputs += [event2vec_lstm_model.sample_weights[0]]
			model_inputs += [K.learning_phase()]

			grad_fn = K.function(inputs=model_inputs, outputs=grad)

			# need to deal with batchsize manually for gradient calculation
			if len(Y_val) > batchsize:
				# get number of batches
				num_batches = len(Y_val) // batchsize

				for j in range(num_batches):
					# get inputs
					inputs_ = [x[j * batchsize: (j + 1) * batchsize] for x in inputs]
					inputs_[0] = inputs_[0]
					inputs_ += [Y_val[j * batchsize: (j + 1) * batchsize]]
					inputs_ += [[1]]
					inputs_ += [0]

					# compute gradients
					tmp = grad_fn(inputs_)[0]
					seq_gradients += [tmp]

				if len(Y_val) % batchsize != 0:
					# get inputs
					inputs_ = [x[num_batches * batchsize:] for x in inputs]
					inputs_[0] = inputs_[0]
					inputs_ += [Y_val[num_batches * batchsize:]]
					inputs_ += [[1]]
					inputs_ += [0]

					# compute gradients
					tmp = grad_fn(inputs_)[0]
					seq_gradients += [tmp]

			else:
				inputs_ = [x for x in inputs]
				inputs_[0] = inputs_[0]
				inputs_ += [Y_val]
				inputs_ += [[1]]
				inputs_ += [0]
				tmp = grad_fn(inputs_)[0]
				seq_gradients += [tmp]

			seq_gradients = np.concatenate(seq_gradients)
			np.save(save_folder + vs + '_grad_batch_' + str(i) + '.npy', seq_gradients)
			# 	gradients += [seq_gradients]
			#
			# unnorm_gradients = np.array(gradients)
			# np.save(folder + vs + '_grad_weights.npy', unnorm_gradients)
			#
			# # normalize gradients
			# gradients = []
			# for i in range(len(unnorm_gradients)):
			# 	tmp = np.linalg.norm(unnorm_gradients[i], axis=1)
			# 	grad_norm = unnorm_gradients[i] / tmp[:, np.newaxis]
			# 	gradients += [grad_norm]
			#
			# gradients = np.array(gradients)
			# np.save(folder + vs + '_norm_grad_weights.npy', gradients)


class HierGrad:

	def runGrad(self, Config, event2vec_lstm_model, folder, val_sets, start):

		save_folder = folder + 'grad/'
		if not os.path.exists(save_folder):
			os.makedirs(save_folder)

		batchsize = Config.BATCH_SIZE

		# val_sets = Config.DATASETS
		vs_dict = Config.DATASETS_DICT

		# get gradients
		vs = val_sets

		x_base = 'Config.X_' + vs.upper()
		y_base = 'Config.Y_' + vs.upper()
		f_base = 'Config.F_' + vs.upper()
		e_base = 'Config.E_' + vs.upper()

		# # saver for gradient data
		# gradients = []

		for i in range(start, vs_dict[vs]):
			print(i)
			X_val = np.load(eval(x_base) + str(i) + '.npy')
			Y_val = np.load(eval(y_base) + str(i) + '.npy')
			F_val = np.load(eval(f_base) + str(i) + '.npy')
			E_val = np.load(eval(e_base) + str(i) + '.npy')


			decoder_input_data = np.zeros((len(Y_val), 1,
										   Config.NUM_DECODER_TOKENS))
			inputs = [X_val, F_val, E_val, decoder_input_data]

			# initialize empty list to store gradients from a single seq_len
			seq_gradients = []

			# build gradient function
			grad = K.gradients(event2vec_lstm_model.total_loss, event2vec_lstm_model.inputs[0])

			# model inputs
			model_inputs = [x for x in event2vec_lstm_model.inputs]
			model_inputs += [event2vec_lstm_model.targets[0]]
			model_inputs += [event2vec_lstm_model.sample_weights[0]]
			model_inputs += [K.learning_phase()]

			grad_fn = K.function(inputs=model_inputs, outputs=grad)

			# need to deal with batchsize manually for gradient calculation
			if len(Y_val) > batchsize:
				# get number of batches
				num_batches = len(Y_val) // batchsize

				for j in range(num_batches):
					# get inputs
					inputs_ = [x[j * batchsize: (j + 1) * batchsize] for x in inputs]
					inputs_[0] = inputs_[0]
					inputs_ += [Y_val[j * batchsize: (j + 1) * batchsize]]
					inputs_ += [[1]]
					inputs_ += [0]

					# compute gradients
					tmp = grad_fn(inputs_)[0]
					seq_gradients += [tmp]

				if len(Y_val) % batchsize != 0:
					# get inputs
					inputs_ = [x[num_batches * batchsize:] for x in inputs]
					inputs_[0] = inputs_[0]
					inputs_ += [Y_val[num_batches * batchsize:]]
					inputs_ += [[1]]
					inputs_ += [0]

					# compute gradients
					tmp = grad_fn(inputs_)[0]
					seq_gradients += [tmp]

			else:
				inputs_ = [x for x in inputs]
				inputs_[0] = inputs_[0]
				inputs_ += [Y_val]
				inputs_ += [[1]]
				inputs_ += [0]
				tmp = grad_fn(inputs_)[0]
				seq_gradients += [tmp]

			seq_gradients = np.concatenate(seq_gradients)
			np.save(save_folder + vs + '_grad_batch_' + str(i) + '.npy', seq_gradients)
		# 	gradients += [seq_gradients]
		#
		# unnorm_gradients = np.array(gradients)
		# np.save(folder + vs + '_grad_weights.npy', unnorm_gradients)
		#
		# # normalize gradients
		# gradients = []
		# for i in range(len(unnorm_gradients)):
		# 	tmp = np.linalg.norm(unnorm_gradients[i], axis=1)
		# 	grad_norm = unnorm_gradients[i] / tmp[:, np.newaxis]
		# 	gradients += [grad_norm]
		#
		# gradients = np.array(gradients)
		# np.save(folder + vs + '_norm_grad_weights.npy', gradients)

