from seq2seq.seq2seqHier import Seq2seqModel
import importlib
from util.config import *
import numpy as np
import argparse
import sys
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument("data", help="data set", type=str)
parser.add_argument("model_folder_1", help="where saved model is", type=str)
parser.add_argument("h5_1", help="best weights", type=str)
parser.add_argument("model_folder_2", help="where saved model is", type=str)
parser.add_argument("h5_2", help="best weights", type=str)
args = parser.parse_args()

# get config
sys.path.append(args.model_folder_1)
from config import *

# get config file
if args.data == 'Amazon':
	Config = AmazonConfig()
elif args.data == 'Sentiment':
	Config = SentimentConfig()

## for prediction
val_sets = Config.DATASETS
vs_dict = Config.DATASETS_DICT
TRAIN_BATCHES = Config.TRAIN_BATCHES

## deal with weights
seq2seqModel = Seq2seqModel()
event2vec_lstm_model_1 = seq2seqModel.createModel(Config)
event2vec_lstm_model_1.load_weights(args.h5_1)

event2vec_lstm_model_2 = seq2seqModel.createModel(Config)
event2vec_lstm_model_2.load_weights(args.h5_2)
print(event2vec_lstm_model_2.summary())

weights_1 = [event2vec_lstm_model_1.get_weights()]
weights_2 = [event2vec_lstm_model_2.get_weights()]

tau = np.array(range(11))/10.
vs_f1 = [[] for _ in range(len(val_sets))]  # val and then test
val_loss = [[] for _ in range(len(val_sets))]
train_loss = []

for t in tau:
	print('tau:', t)
	weights = list()
	for i in range(len(weights_1[0])):
		weights.append(np.array(t*weights_1[0][i] + (1-t)*weights_2[0][i]))

	model = seq2seqModel.createModel(Config)
	model.set_weights(weights)

	train_loss_tmp = 0
	train_num_tmp = 0

	for i in range(TRAIN_BATCHES):
		# X_train = load_npz(Config.X_TRAIN + str(i) + '.npz')
		X_train = np.load(Config.X_TRAIN + str(i) + '.npy')
		Y_train = np.load(Config.Y_TRAIN + str(i) + '.npy')
		F_train = np.load(Config.F_TRAIN + str(i) + '.npy')
		E_train = np.load(Config.E_TRAIN + str(i) + '.npy')

		decoder_input_data = np.zeros((len(Y_train), 1, Config.NUM_DECODER_TOKENS))
		inputs = [X_train, F_train, E_train, decoder_input_data]

		hist = model.evaluate(x=inputs, y=Y_train, batch_size=Config.BATCH_SIZE, verbose=0)

		train_loss_tmp += hist[0] * len(Y_train)
		train_num_tmp += len(Y_train)

	train_loss += [train_loss_tmp / train_num_tmp]

	for v_idx, vs in enumerate(val_sets):
		x_base = 'Config.X_' + vs.upper()
		y_base = 'Config.Y_' + vs.upper()
		f_base = 'Config.F_' + vs.upper()
		e_base = 'Config.E_' + vs.upper()

		# calculate accuracy on the balanced validation set
		y_pred = []
		y_val = []

		val_loss_tmp = 0
		val_num_tmp = 0

		for i in range(vs_dict[vs]):
			# X_val = load_npz(eval(x_base) + str(i) + '.npz')
			X_val = np.load(eval(x_base) + str(i) + '.npy')
			Y_val = np.load(eval(y_base) + str(i) + '.npy')
			F_val = np.load(eval(f_base) + str(i) + '.npy')
			E_val = np.load(eval(e_base) + str(i) + '.npy')

			decoder_input_data = np.zeros((len(Y_val), 1, Config.NUM_DECODER_TOKENS))
			inputs = [X_val, F_val, E_val, decoder_input_data]

			if vs == 'val':
				hist = model.evaluate(x=inputs, y=Y_val, batch_size=Config.BATCH_SIZE, verbose=0)

			# val_loss_tmp += hist.history['loss'][0] * len(Y_val)
			val_num_tmp += len(Y_val)

			y_pred += [model.predict(inputs, batch_size=Config.BATCH_SIZE)]
			y_val += [Y_val]

		y_val = np.concatenate(y_val, axis=0)
		y_pred = np.concatenate(y_pred, axis=0)

		prd = np.argmax(y_pred, axis=-1)
		act = np.argmax(y_val, axis=-1)
		vs_f1[v_idx] += [f1_score(act, prd)]
		val_loss[v_idx] += [val_loss_tmp / val_num_tmp]

val_f1 = vs_f1[0]
test_f1 = vs_f1[1]
np.save(args.model_folder_1 + 'interp_val_f1.npy', val_f1)
np.save(args.model_folder_1 + 'interp_test_f1.npy', test_f1)
np.save(args.model_folder_1 + 'interp_val_loss.npy', val_loss[0])
np.save(args.model_folder_1 + 'interp_test_loss.npy', val_loss[1])
np.save(args.model_folder_1 + 'interp_train_loss.npy', train_loss)

np.save(args.model_folder_2 + 'interp_val_f1.npy', val_f1)
np.save(args.model_folder_2 + 'interp_test_f1.npy', test_f1)
np.save(args.model_folder_2 + 'interp_val_loss.npy', val_loss[0])
np.save(args.model_folder_2 + 'interp_test_loss.npy', val_loss[1])
np.save(args.model_folder_2 + 'interp_train_loss.npy', train_loss)