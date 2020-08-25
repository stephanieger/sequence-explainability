import time
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
import matplotlib
import datetime
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import importlib
from scipy.sparse import load_npz


class ChurnAnalyzer:
	######################################################### RUN SEQ2SEQ CLASS MODEL ##########################################################

	def runSeq2Seq(self, Config, event2vec_lstm_model, folder):
		# calculate total training time
		start_time = time.time()
		string_time = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d%H%M%S')

		# want to monitor loss and accuracy for both training and validation data
		val_acc = []
		train_acc = []
		val_loss = []
		train_loss = []
		best_acc = 0
		best_f1 = 0.  # use this to determine when to terminate training
		tol = 1e-2  # use this to determine when to terminate training
		count = 0
		max_count = 1000  # use this to determine when to terminate the model
		best_train_loss = 100

		val_sets = Config.DATASETS
		vs_dict = Config.DATASETS_DICT
		TRAIN_BATCHES = Config.TRAIN_BATCHES

		vs_f1 = [[] for _ in range(len(val_sets))]
		vs_bin_f1 = [[] for _ in range(len(val_sets))]
		best_vs_f1 = [0. for _ in range(len(val_sets))]

		for epochs in range(Config.EPOCHS):
			idx = np.array(range(TRAIN_BATCHES))
			# shuffle data
			np.random.shuffle(idx)

			# need to store temporary accuracy and loss because the loss and accuracy are averaged over each batch
			# our batch sizes are uneven so we take the weighted average across the batches
			train_acc_tmp = 0
			train_loss_tmp = 0
			val_acc_tmp = 0
			val_loss_tmp = 0
			val_num_tmp = 0
			train_num_tmp = 0

			print('epochs:', epochs)
			for i in idx:
				# X_train = load_npz(Config.X_TRAIN + str(i) + '.npz')
				X_train = np.load(Config.X_TRAIN + str(i) + '.npy')
				Y_train = np.load(Config.Y_TRAIN + str(i) + '.npy')
				F_train = np.load(Config.F_TRAIN + str(i) + '.npy')
				E_train = np.load(Config.E_TRAIN + str(i) + '.npy')

				decoder_input_data = np.zeros((len(Y_train), 1, Config.NUM_DECODER_TOKENS))
				inputs = [X_train, F_train, E_train, decoder_input_data]

				hist = event2vec_lstm_model.fit(x=inputs, y=Y_train, epochs=1, batch_size=Config.BATCH_SIZE, verbose=0)

				train_acc_tmp += hist.history['acc'][0] * len(Y_train)
				train_loss_tmp += hist.history['loss'][0] * len(Y_train)
				train_num_tmp += len(Y_train)

			train_acc += [train_acc_tmp / train_num_tmp]
			train_loss += [train_loss_tmp / train_num_tmp]

			if train_loss[-1] < best_train_loss:
				best_train_loss = train_loss[-1]
				event2vec_lstm_model.save_weights(folder + 'train_loss_best_weights.h5')

			# get performance on all val sets
			for v_idx, vs in enumerate(val_sets):
				x_base = 'Config.X_' + vs.upper()
				y_base = 'Config.Y_' + vs.upper()
				f_base = 'Config.F_' + vs.upper()
				e_base = 'Config.E_' + vs.upper()

				# calculate accuracy on the balanced validation set
				y_pred = []
				y_val = []

				for i in range(vs_dict[vs]):
					# X_val = load_npz(eval(x_base) + str(i) + '.npz')
					X_val = np.load(eval(x_base) + str(i) + '.npy')
					Y_val = np.load(eval(y_base) + str(i) + '.npy')
					F_val = np.load(eval(f_base) + str(i) + '.npy')
					E_val = np.load(eval(e_base) + str(i) + '.npy')

					decoder_input_data = np.zeros((len(Y_val), 1, Config.NUM_DECODER_TOKENS))
					inputs = [X_val, F_val, E_val, decoder_input_data]

					if vs == 'val':
						hist = event2vec_lstm_model.evaluate(x=inputs, y=Y_val, batch_size=Config.BATCH_SIZE, verbose=0)

					y_pred += [event2vec_lstm_model.predict(inputs, batch_size=Config.BATCH_SIZE)]
					y_val += [Y_val]

					if vs == 'val':
						val_acc_tmp += hist[1] * len(Y_val)
						val_loss_tmp += hist[0] * len(Y_val)
						val_num_tmp += len(Y_val)

				if vs == 'val':
					val_acc += [val_acc_tmp / val_num_tmp]
					val_loss += [val_loss_tmp / val_num_tmp]
					if val_acc[-1] > best_acc:
						best_acc = val_acc[-1]
						event2vec_lstm_model.save_weights(folder + '_val3_best_epoch.h5')
					print(val_acc[-1])

				y_val = np.concatenate(y_val, axis=0)
				y_pred = np.concatenate(y_pred, axis=0)

				val_str = vs

				np.save(folder + val_str + '_pred' + str(epochs) + '.npy', y_pred)
				np.save(folder + val_str + '_true' + '.npy', y_val)

				prd = np.argmax(y_pred, axis=-1)
				act = np.argmax(y_val, axis=-1)
				vs_f1[v_idx] += [f1_score(act, prd)]

				if f1_score(act, prd) > best_vs_f1[v_idx]:
					best_vs_f1[v_idx] = f1_score(act, prd)
					event2vec_lstm_model.save_weights(folder + val_str + '_f1_best_epoch_' + str(epochs) + '.h5')

			# dynamic stopping condition
			if vs_f1[1][-1] > best_f1:
				if np.abs((best_f1 - vs_f1[1][-1])) < tol:
					count = 0
				else:
					count += 1
					if count == max_count:
						break
				best_f1 = vs_f1[1][-1]
			else:
				count += 1
				if count == max_count:
					break

			val_f1 = vs_f1[0]
			test_f1 = vs_f1[1]

			if (epochs + 1) % 5 == 0:
				np.save(folder + 'val_f1.npy', val_f1)
				np.save(folder + 'test_f1.npy', test_f1)
				np.save(folder + 'train_acc.npy', train_acc)
				np.save(folder + 'train_loss.npy', train_loss)

		print(__file__)
		print("training time --- %s minutes / epoch ---" % ((time.time() - start_time) / 60 / Config.EPOCHS))

		# plot validation accuracy for each epoch
		np.save(folder + 'val_f1.npy', val_f1)
		np.save(folder + 'test_f1.npy', test_f1)
		np.save(folder + 'train_acc.npy', train_acc)
		np.save(folder + 'train_loss.npy', train_loss)

