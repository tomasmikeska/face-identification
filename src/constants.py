import os
from utils import relative_path


# Hyperparams
EMBEDDING_SIZE       = 256
MIN_FACES_PER_PERSON = 130  # Min faces per person in training data
DEV_FACES_PER_PERSON = 5    # Number of images per person in dev data
BATCH_SIZE           = 128
EPOCHS               = 20
EPOCH_PARTITION      = 1. / 5.  # Epoch subset to use in training for more updates (callbacks, val data eval)
TARGET_IMG_WIDTH     = 96
TARGET_IMG_HEIGHT    = 112
MIN_IMG_WIDTH        = TARGET_IMG_WIDTH   # no image upscale allowed
MIN_IMG_HEIGHT       = TARGET_IMG_HEIGHT  # no image upscale allowed
INPUT_SHAPE          = (TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3)

# Paths
MODEL_SAVE_PATH      = os.environ.get('MODEL_SAVE_PATH', relative_path('../model/'))
VGG_TRAIN_PATH       = os.environ.get('VGG_DATASET', relative_path('../data/VGGFace2/')) + '/train/'
VGG_TEST_PATH        = os.environ.get('VGG_DATASET', relative_path('../data/VGGFace2/')) + '/test/'
VGG_BB_TEST_MAP      = os.environ.get('BB_TEST', relative_path('../data/bb_landmark/loose_bb_test.csv'))
VGG_BB_TRAIN_MAP     = os.environ.get('BB_TRAIN', relative_path('../data/bb_landmark/loose_bb_train.csv'))
LFW_PATH             = os.environ.get('LFW_DATASET', relative_path('../data/lfw/'))
LFW_PAIRS_PATH       = os.environ.get('LFW_PAIRS', relative_path('../data/lfw_pairs.txt'))
TB_LOGS              = os.environ.get('TB_LOGS', relative_path('../tb_logs/'))
