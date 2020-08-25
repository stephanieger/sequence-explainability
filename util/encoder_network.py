from keras.models import Model
from keras.layers import *
import tensorflow as tf


def create_encoder(Config):
	# load in the config file
	# sys.path.append(config_folder)
	# from config import Config

	number_of_event_types = Config.NUMBER_OF_EVENT_TYPES
	output_dim = Config.DATA_DIM

	encoder_inputs = Input(shape=(None, Config.DATA_DIM,))

	try:
		if Config.FC:
			x = Dense(Config.FC_DIM)(encoder_inputs)
		else:
			x = encoder_inputs
	except:
		x = encoder_inputs

	# normalize the embedding
	x = BatchNormalization(axis=-1)(x)

	# return sequence for attention
	ret_seq = True

	x, state_h, state_c = LSTM(Config.L1_DIM, dropout=0.0, recurrent_dropout=0.0, return_sequences=ret_seq, return_state=True)(x)
	x = BatchNormalization(axis=-1)(x)

	if Config.L2_DIM != None:
		x = BatchNormalization(axis=-1)(x)
		x, state_h, state_c = LSTM(Config.L2_DIM, dropout=0.2, recurrent_dropout=0.2, \
				return_sequences=ret_seq, return_state=True)(x)
	if Config.L3_DIM != None:
		x = BatchNormalization(axis=-1)(x)
		encoder_out, state_h, state_c = LSTM(Config.L3_DIM, \
			dropout=0.2, recurrent_dropout=0.2, \
			return_sequences=ret_seq, return_state=True)(x)

	if Config.L3_DIM != None:
		enc_dim = Config.L3_DIM
	elif Config.L2_DIM != None:
		enc_dim = Config.L2_DIM
	else:
		enc_dim = Config.L1_DIM

	if Config.L3_DIM == None:
		encoder_out = x
	elif Config.L2_DIM == None:
		encoder_out = x
	inputs = [encoder_inputs]

	enc_model = Model(inputs=inputs, outputs=[encoder_out, state_h, state_c])
	print(enc_model.summary())

	return enc_model, enc_dim

