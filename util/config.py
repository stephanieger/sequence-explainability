class SentimentConfig:
	# folders
	DATA_FOLDER = '../IMDB/process-data-bert/batch/'
	OUTPUT_FOLDER = '../IMDB/bert-outputs/'

	# number of batches
	TRAIN_BATCHES = 105
	VAL_BATCHES = 94
	TEST_BATCHES = 99

	# embedding weights
	EMBEDDING_WEIGHTS = DATA_FOLDER + 'embedding_weights.npy'

	# data
	X_TRAIN = DATA_FOLDER + 'x_train_'
	X_VAL = DATA_FOLDER + 'x_val_'
	X_TEST = DATA_FOLDER + 'x_test_'

	# one-hot features
	F_TRAIN = DATA_FOLDER + 'f_train_'
	F_VAL = DATA_FOLDER + 'f_val_'
	F_TEST = DATA_FOLDER + 'f_test_'

	# event mask
	E_TRAIN = DATA_FOLDER + 'e_train_'
	E_VAL = DATA_FOLDER + 'e_val_'
	E_TEST = DATA_FOLDER + 'e_test_'

	# labels
	Y_TRAIN = DATA_FOLDER + 'y_train_'
	Y_VAL = DATA_FOLDER + 'y_val_'
	Y_TEST = DATA_FOLDER + 'y_test_'

	# type data
	T_TRAIN = DATA_FOLDER + 't_train_'
	T_VAL = DATA_FOLDER + 't_val_'
	T_TEST = DATA_FOLDER + 't_test_'

	# parameters for data
	TIMESTEPS = 128
	DATA_DIM = 1024
	WORD2VEC_DIM = 1024
	NUM_CLASSES = 1
	NUMBER_OF_EVENT_TYPES = 2538
	NUM_DECODER_TOKENS = 2

	# model building parameters
	HIDDEN_NEURONS = 100
	DENSE_HIDDEN_NEURONS = 8
	NUM_LAYERS = 3
	L1_DIM = 128
	L2_DIM = 64
	L3_DIM = 16 #None

	# model training parameters
	BATCH_SIZE = 512
	EPOCHS = 300
	OPT_LR = 1e-3
	DROPOUT_KEEP_PROB = 0.8
	OPTIMIZER = 'adagrad'

	# fully connected layer
	FC = True
	FC_DIM = 100
	EMBEDDING_TRAINABLE = True

	# attention related config
	ATTN_N = 10
	DATASETS = ["val", "test"]
	ALL_DATASETS = ["train", "val", "test"]
	ALL_DATA_DICT = {"train": TRAIN_BATCHES, "val": VAL_BATCHES, "test": TEST_BATCHES}
	DATASETS_DICT = {"val": VAL_BATCHES, "test": TEST_BATCHES}

	# seeding for model runs
	SEEDED_RUN = True
	SEED_VALUE = 0

	# improved gradient
	MU = 0.5
	T = 1
	EPS = 1e-10
	SUBSET = 10
	SUBSEED = 0

	ATTENTION_FLAG = True

class AmazonConfig:
	# folders
	DATA_FOLDER = '../amazon-polarity/process-data-bert/batch/'
	OUTPUT_FOLDER = '../amazon-polarity/bert-outputs/'

	# number of batches
	TRAIN_BATCHES = 100
	VAL_BATCHES = 97
	TEST_BATCHES = 102

	# embedding weights
	EMBEDDING_WEIGHTS = DATA_FOLDER + 'embedding_weights.npy'

	# data
	X_TRAIN = DATA_FOLDER + 'x_train_'
	X_VAL = DATA_FOLDER + 'x_val_'
	X_TEST = DATA_FOLDER + 'x_test_'

	# one-hot features
	F_TRAIN = DATA_FOLDER + 'f_train_'
	F_VAL = DATA_FOLDER + 'f_val_'
	F_TEST = DATA_FOLDER + 'f_test_'

	# event mask
	E_TRAIN = DATA_FOLDER + 'e_train_'
	E_VAL = DATA_FOLDER + 'e_val_'
	E_TEST = DATA_FOLDER + 'e_test_'

	# labels
	Y_TRAIN = DATA_FOLDER + 'y_train_'
	Y_VAL = DATA_FOLDER + 'y_val_'
	Y_TEST = DATA_FOLDER + 'y_test_'

	# type data
	T_TRAIN = DATA_FOLDER + 't_train_'
	T_VAL = DATA_FOLDER + 't_val_'
	T_TEST = DATA_FOLDER + 't_test_'

	# parameters for data
	TIMESTEPS = 128
	DATA_DIM = 1024
	WORD2VEC_DIM = 1024
	NUM_CLASSES = 1
	NUMBER_OF_EVENT_TYPES = 2602 #1044
	NUM_DECODER_TOKENS = 2

	# model building parameters
	HIDDEN_NEURONS = 16
	DENSE_HIDDEN_NEURONS = 8
	NUM_LAYERS = 2
	L1_DIM = 128
	L2_DIM = 64
	L3_DIM = None

	# fully connected layer
	FC = True
	FC_DIM = 100
	EMBEDDING_TRAINABLE = True

	# model training parameters
	BATCH_SIZE = 512
	EPOCHS = 600
	OPT_LR = 1e-3
	DROPOUT_KEEP_PROB = 0.8
	OPTIMIZER = 'adagrad'

	# attention related config
	ATTN_N = 10
	DATASETS = ["val", "test"]
	DATASETS_DICT = {"val": VAL_BATCHES, "test": TEST_BATCHES}
	ALL_DATASETS = ["train", "val", "test"]
	ALL_DATA_DICT = {"train": TRAIN_BATCHES, "val": VAL_BATCHES, "test": TEST_BATCHES}

	# seeding for model runs
	SEEDED_RUN = True
	SEED_VALUE = 0

	# improved gradient
	MU = 0.5
	T = 1
	EPS = 1e-10
	SUBSET = 10
	SUBSEED = 0

	ATTENTION_FLAG = True
