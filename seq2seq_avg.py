from seq2seq.seq2seqAvg import Seq2seqModel
from util.churnAvg import ChurnAnalyzer
import random as rn
import tensorflow as tf
from util.config import *
import numpy as np
import argparse
from util.folder_name import get_folder_name

# to call a specific config file run as python3 seq2seq_avg.py util.myconfig
# where the config file is at util/myconfig.py

parser = argparse.ArgumentParser()
parser.add_argument("data", help="data set ")
args = parser.parse_args()

# get config file
if args.data == 'Amazon':
    Config = AmazonConfig()
elif args.data == 'Sentiment':
    Config = SentimentConfig()

if Config.SEEDED_RUN:
    seed = Config.SEED_VALUE
    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)


seq2seqModel = Seq2seqModel()
event2vec_lstm_model = seq2seqModel.createModel(Config)
print(event2vec_lstm_model.summary())

# Prep output folder
file_name = str(__file__).split('.')[0] # get file name for saving figures
folder = get_folder_name(file_name, Config, 'util.config')

churnAnalyzer = ChurnAnalyzer()
churnAnalyzer.runSeq2Seq(Config, event2vec_lstm_model, folder)
print(__file__)

