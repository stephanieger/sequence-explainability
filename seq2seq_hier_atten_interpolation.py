from seq2seq.seq2seqHier import Seq2seqAtten
from util.attenAnalyzer import AttenHier
import importlib
# from util.config import *
import numpy as np
import argparse
import sys
import glob
import os
# to call a specific config file run as python3 seq2seq_avg.py util.myconfig
# where the config file is at util/myconfig.py


parser = argparse.ArgumentParser()
parser.add_argument("data", help="data set", type=str)
parser.add_argument("save_folder", help="where to save model", type=str)
parser.add_argument("model_folder", help ='where saved model is to get correct config', type=str)
parser.add_argument("weights_folder", help="where interpolation weights are saved", type=str)
args = parser.parse_args()

num_weights = len(glob.glob(args.weights_folder + 'w_hat_*'))
# get config
sys.path.append(args.model_folder)

# create save directory
if not os.path.exists(args.save_folder):
	os.makedirs(args.save_folder)

from config import *

# get config file
if args.data == 'Amazon':
	Config = AmazonConfig()
elif args.data == 'Sentiment':
	Config = SentimentConfig()

seq2seqModel = Seq2seqAtten()
event2vec_lstm_model = seq2seqModel.createModel(Config)
# event2vec_lstm_model.load_weights(args.h5)
print(event2vec_lstm_model.summary())

for n in range(num_weights):
	weights = np.load(args.weights_folder + 'w_hat_' + str(n+1) + '.npy')
	save_dir = args.save_folder + 'interp-step-' + str(n) + '/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	event2vec_lstm_model.set_weights(weights)
	attenAnalyzer = AttenHier()
	attenAnalyzer.runAtten(Config, event2vec_lstm_model, save_dir)

print(__file__)

