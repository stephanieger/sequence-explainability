from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import importlib
from util.encoder_network import create_encoder
from util.HierAttentionLSTM import HierarchicalAttentionDecoder
import sys


class Seq2seqModel():

	def createModel(self, Config):

		num_decoder_tokens = Config.NUM_DECODER_TOKENS

		# get the encoder network
		encoder_net, enc_dim = create_encoder(Config)
		encoder_inputs = Input(shape=(None, Config.DATA_DIM))
		hier_event_inputs = Input(shape=(None, Config.NUMBER_OF_EVENT_TYPES,))
		hier_inputs = Input(shape=(Config.NUMBER_OF_EVENT_TYPES, ))

		inputs = [encoder_inputs]
		encoder_out, state_h, state_c = encoder_net(inputs)

		# concatenate hidden states into input
		states = [state_h, state_c]

		# build decoder model
		# these are all layers for the decoder model
		decoder_inputs = Input(shape=(None, num_decoder_tokens))
		decoder_lstm = HierarchicalAttentionDecoder(enc_dim, return_sequences=True, return_state=True)
		# if Config.ATTENTION_FLAG:
		# 	decoder_lstm = HierarchicalAttentionDecoder(enc_dim, return_sequences=True, return_state=True)
		# else:
		# 	decoder_lstm = LSTM(enc_dim, return_sequences=True, return_state=True)
		decoder_dense = Dense(num_decoder_tokens, activation='softmax')

		# start building the model
		inputs = decoder_inputs

		atten, state_h, state_c = decoder_lstm(inputs, initial_state=states,
											   constants=[encoder_out, hier_event_inputs, hier_inputs])

		# if Config.ATTENTION_FLAG:
		# 	atten, state_h, state_c = decoder_lstm(inputs, initial_state=states,
		# 										   constants=[encoder_out, hier_event_inputs, hier_inputs])
		# else:
		# 	atten, state_h, state_c = decoder_lstm(inputs, initial_state=states)
		outputs = Lambda(lambda x: K.expand_dims(x, axis=1))(state_h)
		outputs = decoder_dense(outputs)

		# concatenate outputs

		inputs = [encoder_inputs, hier_event_inputs, hier_inputs, decoder_inputs]
		event2vec_lstm_model = Model(inputs=inputs, outputs=outputs)
		print(event2vec_lstm_model.summary())

		opt = None
		# define optimizer
		if Config.OPTIMIZER == 'adagrad':
			opt = optimizers.Adagrad(clipvalue=10.0, lr=Config.OPT_LR)
		elif Config.OPTIMIZER == 'adadelta':
			opt = optimizers.Adadelta(clipvalue=10.0, lr=Config.OPT_LR)
		elif Config.OPTIMIZER == 'rmsprop':
			opt = optimizers.RMSprop(clipvalue=10.0, lr=Config.OPT_LR)
		else:
			raise ValueError('invalid argument for optimizer type')

		event2vec_lstm_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

		return event2vec_lstm_model


class Seq2seqAtten():

	def createModel(self, Config):

		num_decoder_tokens = Config.NUM_DECODER_TOKENS

		# get the encoder network
		encoder_net, enc_dim = create_encoder(Config)
		encoder_inputs = Input(shape=(None, Config.DATA_DIM))
		hier_event_inputs = Input(shape=(None, Config.NUMBER_OF_EVENT_TYPES,))
		hier_inputs = Input(shape=(Config.NUMBER_OF_EVENT_TYPES,))

		inputs = [encoder_inputs]
		encoder_out, state_h, state_c = encoder_net(inputs)

		# concatenate hidden states into input
		states = [state_h, state_c]

		# build decoder model
		# these are all layers for the decoder model
		decoder_inputs = Input(shape=(None, num_decoder_tokens))
		if Config.ATTENTION_FLAG:
			decoder_lstm = HierarchicalAttentionDecoder(enc_dim, return_sequences=True, return_state=True)
		else:
			decoder_lstm = LSTM(enc_dim, return_sequences=True, return_state=True)
		decoder_dense = Dense(num_decoder_tokens, activation='softmax')

		# start building the model
		inputs = decoder_inputs

		atten, state_h, state_c = decoder_lstm(inputs, initial_state=states,
											   constants=[encoder_out, hier_event_inputs, hier_inputs])
		outputs = Lambda(lambda x: K.expand_dims(x, axis=1))(state_h)
		outputs = decoder_dense(outputs)

		inputs = [encoder_inputs, hier_event_inputs, hier_inputs, decoder_inputs]
		event2vec_lstm_model = Model(inputs=inputs, outputs=[outputs, atten])
		print(event2vec_lstm_model.summary())

		opt = None
		# define optimizer
		if Config.OPTIMIZER == 'adagrad':
			opt = optimizers.Adagrad(clipvalue=10.0, lr=Config.OPT_LR)
		elif Config.OPTIMIZER == 'adadelta':
			opt = optimizers.Adadelta(clipvalue=10.0, lr=Config.OPT_LR)
		elif Config.OPTIMIZER == 'rmsprop':
			opt = optimizers.RMSprop(clipvalue=10.0, lr=Config.OPT_LR)
		else:
			raise ValueError('invalid argument for optimizer type')

		event2vec_lstm_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

		return event2vec_lstm_model


