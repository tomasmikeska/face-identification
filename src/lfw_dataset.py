import os
import numpy as np
import random
import pickle
from sklearn.datasets import fetch_lfw_people
from utils import relative_path, file_exists
from constants import NUM_CHANNELS, MIN_FACES_PER_PERSON, DATASET_COMPRESSED_PATH


def fetch_dataset():
    lfw_people = fetch_lfw_people(min_faces_per_person=MIN_FACES_PER_PERSON, resize=0.8)
    images = lfw_people.images.reshape(lfw_people.images.shape + (NUM_CHANNELS,))
    images = images / 255.0
    return images, lfw_people.target


def load_dataset():
    if file_exists(DATASET_COMPRESSED_PATH):
        with open(DATASET_COMPRESSED_PATH, 'rb') as file:
            return pickle.load(file)
    else:
        return fetch_dataset()


if __name__ == '__main__':
    dataset = fetch_dataset()
    with open(DATASET_COMPRESSED_PATH, 'wb+') as file:
        pickle.dump(dataset, file)
    print('Dataset saved to %s' % (DATASET_COMPRESSED_PATH))
