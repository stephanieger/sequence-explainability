from seq2seq.seq2seqAvg import Seq2seqModel
from util.GradAnalyzer import AvgGrad
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
parser.add_argument("val_set", help="set to predict on", type=str)
parser.add_argument("-s", "--start", default=0, type=int)
args = parser.parse_args()

# get config
sys.path.append(args.model_folder)
from config import *

# get config file
if args.data == 'Amazon':
    Config = AmazonConfig()
elif args.data == 'Sentiment':
    Config = SentimentConfig()

seq2seqModel = Seq2seqModel()
event2vec_lstm_model = seq2seqModel.createModel(Config)
event2vec_lstm_model.load_weights(args.h5)
print(event2vec_lstm_model.summary())

attenAnalyzer = AvgGrad()
attenAnalyzer.runGrad(Config, event2vec_lstm_model, args.model_folder, args.val_set, args.start)
print(__file__)

