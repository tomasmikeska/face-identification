import os
from utils import relative_path

# Hyperparams
EMBEDDING_SIZE = 128
MIN_FACES_PER_PERSON = 2
NUM_CHANNELS = 1
BATCH_SIZE = 512
TRIPLET_LOSS_ALPHA = 0.5
MAX_PERSON_IMGS_IN_BATCH = 50
MAX_ANCHOR_POS_COUNT_IN_BATCH = 10
SEMIHARD_MARGIN = 0.3
TEST_BATCH_SIZE = 256
TEST_PERIOD = 5 # every 'x' batches
SAVE_PERIOD = 20 # every 'x' batches

# Paths
MODEL_SAVE_PATH = os.environ.get('MODEL_SAVE_PATH', relative_path('../model/'))
LATEST_MODEL_PATH = relative_path('../model/facenet_squeezenet_weights.h5')
DATASET_COMPRESSED_PATH = os.environ.get('DATASET', relative_path('../data/dataset.npy'))
LOCAL_TRAIN_DIR = relative_path('../data/train/')
LOCAL_TEST_DIR = relative_path('../data/test/')
