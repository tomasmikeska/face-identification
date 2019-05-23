import os
import numpy as np
import squeezenet
import facenet
import gc
from itertools import combinations
from sklearn.utils import shuffle
from lfw_dataset import load_dataset
from triplet_generator import get_offline_triplet_generator, get_online_triplet_generator, get_combined_triplet_generator
from facenet import triplet_loss
from utils import relative_path, file_exists
from constants import MODEL_SAVE_PATH, LATEST_MODEL_PATH, SAVE_PERIOD, TEST_BATCH_SIZE, TEST_PERIOD


def load_model(input_shape):
    base_model = squeezenet.create_model(input_shape)
    if file_exists(LATEST_MODEL_PATH):
        base_model.load_weights(LATEST_MODEL_PATH)
    facenet_model = facenet.create_model(base_model, input_shape)
    facenet_model.compile(optimizer='adam',
                          loss=triplet_loss)
    return facenet_model, base_model


def save_model(model):
    print('Saving model...')
    model.save_weights(MODEL_SAVE_PATH + 'facenet_squeezenet_weights.h5')


def calc_accuracy(distances):
    return np.mean(distances[:, 0] < distances[:, 1])


def train(siamese_model, base_model, X, y):
    test_generator = get_offline_triplet_generator(X, y, triplet_count=TEST_BATCH_SIZE)
    acc_list = []
    for i, batch in enumerate(get_combined_triplet_generator(X, y, siamese_model.predict)):
        print('Training batch %s' % (i + 1))
        siamese_model.train_on_batch(batch, np.zeros(len(batch[0])))

        if i % TEST_PERIOD == 0:
            distances = siamese_model.predict(next(test_generator))
            acc_list.append(calc_accuracy(distances))
            print('Acc on test: %s' % np.mean(acc_list[-5:]))
        if i % SAVE_PERIOD == 0:
            save_model(base_model)

if __name__ == '__main__':
    # Train
    X, y = load_dataset()
    input_shape = X.shape[1:]
    facenet_model, base_model = load_model(input_shape)
    train(facenet_model, base_model, X, y)
    save_model(base_model)
