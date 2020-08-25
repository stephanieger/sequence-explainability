from seq2seq.seq2seqAvg import Seq2seqAtten
from util.attenAnalyzer import AttenAvg
import importlib
from util.config import *
import numpy as np
import argparse
import sys

# to call a specific config file run as python3 seq2seq_avg.py util.myconfig
# where the config file is at util/myconfig.py


parser = argparse.ArgumentParser()
parser.add_argument("data", help="data set", type=str)
parser.add_argument("model_folder", help="where saved model is", type=str)
parser.add_argument("h5", help="best weights", type=str)
args = parser.parse_args()

# get config
sys.path.append(args.model_folder)
from config import *

# get config file
if args.data == 'Amazon':
    Config = AmazonConfig()
elif args.data == 'Sentiment':
    Config = SentimentConfig()

seq2seqModel = Seq2seqAtten()
event2vec_lstm_model = seq2seqModel.createModel(Config)
event2vec_lstm_model.load_weights(args.h5)
print(event2vec_lstm_model.summary())

attenAnalyzer = AttenAvg()
attenAnalyzer.runAtten(Config, event2vec_lstm_model, args.model_folder)
print(__file__)

