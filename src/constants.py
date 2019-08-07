import os
from utils import relative_path


# Hyperparams
ARCFACE_M            = 0.5
ARCFACE_S            = 64.
CENTERLOSS_ALPHA     = 0.008
CENTERLOSS_LAMBDA    = 0.5
EMBEDDING_SIZE       = 256
MIN_FACES_PER_PERSON = 10  # Min num of samples per class - or class is removed
MAX_FACES_PER_PERSON = 150  # Max num of samples per class - additional samples are removed
MIN_FACES_UNSAMPLE   = 10  # All classes with lower num of samples are upscaled to this num of samples
DEV_FACES_PER_PERSON = 2  # Number of images per person in dev data
BATCH_SIZE           = 256
EPOCHS               = 30
TARGET_IMG_WIDTH     = 96
TARGET_IMG_HEIGHT    = 112
MIN_IMG_WIDTH        = TARGET_IMG_WIDTH   # no image upscale allowed
MIN_IMG_HEIGHT       = TARGET_IMG_HEIGHT  # no image upscale allowed
INPUT_SHAPE          = (TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3)

# Paths
MODEL_SAVE_PATH      = os.environ.get('MODEL_SAVE_PATH', relative_path('../model/'))
VGG_TRAIN_PATH       = os.environ.get('VGG_DATASET',     relative_path('../data/VGGFace2/')) + '/train/'
VGG_TEST_PATH        = os.environ.get('VGG_DATASET',     relative_path('../data/VGGFace2/')) + '/test/'
VGG_BB_TRAIN_MAP     = os.environ.get('BB_TRAIN',        relative_path('../data/vggface_bb_landmark/loose_bb_train.csv'))
VGG_BB_TEST_MAP      = os.environ.get('BB_TEST',         relative_path('../data/vggface_bb_landmark/loose_bb_test.csv'))
CASIA_PATH           = os.environ.get('CASIA_DATASET',   relative_path('../data/CASIA-WebFace/'))
CASIA_BB_MAP         = os.environ.get('CASIA_BB',        relative_path('../data/casia_landmark.csv'))
LFW_PATH             = os.environ.get('LFW_DATASET',     relative_path('../data/lfw/'))
LFW_BB_MAP           = os.environ.get('LFW_BB',          relative_path('../data/lfw_landmark.csv'))
LFW_PAIRS_PATH       = os.environ.get('LFW_PAIRS',       relative_path('../data/lfw_pairs.txt'))
