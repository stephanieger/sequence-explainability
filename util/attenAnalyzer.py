import numpy as np
import keras
import tensorflow as tf
from scipy.sparse import load_npz


########################### ATTENTION FOR AVERAGING MODEL #####################
class AttenAvg:
	def runAtten(self, Config, atten_model, folder):

		# get number of batches
		VAL_BATCHES = Config.VAL_BATCHES
		TEST_BATCHES = Config.TEST_BATCHES

		val_sets = Config.DATASETS
		vs_dict = Config.DATASETS_DICT

		# get attention
		for v_idx, vs in enumerate(val_sets):

			print(vs)
			x_base = 'Config.X_' + vs.upper()
			y_base = 'Config.Y_' + vs.upper()

			# saver for attention data
			atten = []

			for i in range(vs_dict[vs]):
				# X_val = load_npz(eval(x_base) + str(i) + '.npz')
				X_val = np.load(eval(x_base) + str(i) + '.npy')
				Y_val = np.load(eval(y_base) + str(i) + '.npy')

				decoder_input_data = np.zeros((len(Y_val), 1,
											   Config.NUM_DECODER_TOKENS))
				inputs = [X_val, decoder_input_data]

				# predict attention outputs
				(_out, _atten) = atten_model.predict(inputs, batch_size=Config.BATCH_SIZE)
				atten += [_atten]

			# save attention as an array
			atten = np.array(atten)
			np.save(folder + vs + '_atten_weights.npy', atten)


########################### ATTENTION FOR HIERARCHICAL MODEL #####################

class AttenHier:
	def runAtten(self, Config, atten_model, folder):

		# get number of batches
		VAL_BATCHES = Config.VAL_BATCHES
		TEST_BATCHES = Config.TEST_BATCHES

		val_sets = Config.DATASETS
		vs_dict = Config.DATASETS_DICT

		# get attention
		for v_idx, vs in enumerate(val_sets):

			print(vs)
			x_base = 'Config.X_' + vs.upper()
			f_base = 'Config.F_' + vs.upper()
			e_base = 'Config.E_' + vs.upper()

			# saver for attention data
			atten = []

			for i in range(vs_dict[vs]):
				# X_val = load_npz(eval(x_base) + str(i) + '.npz')
				X_val = np.load(eval(x_base) + str(i) + '.npy')
				F_val = np.load(eval(f_base) + str(i) + '.npy')
				E_val = np.load(eval(e_base) + str(i) + '.npy')

				decoder_input_data = np.zeros((len(F_val), 1,
											   Config.NUM_DECODER_TOKENS))
				inputs = [X_val, F_val, E_val, decoder_input_data]

				# predict attention outputs
				(_out, _atten) = atten_model.predict(inputs, batch_size=Config.BATCH_SIZE)
				atten += [_atten]

			# save attention as an array
			atten = np.array(atten)
			np.save(folder + vs + '_atten_weights.npy', atten)