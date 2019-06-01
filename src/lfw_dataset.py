import os
import numpy as np
import random
import pickle
from sklearn.datasets import fetch_lfw_people
from utils import relative_path, file_exists
from constants import MIN_FACES_PER_PERSON, DATASET_COMPRESSED_PATH


def person_image_mapping(images, targets):
    person_images = { person_id: [] for person_id in set(targets) }

    for i in range(0, images.shape[0]):
        person_id = targets[i]
        person_images[person_id].append(images[i])

    return person_images


def fetch_dataset():
    lfw_people = fetch_lfw_people(min_faces_per_person=MIN_FACES_PER_PERSON, resize=1.0)
    images = lfw_people.images / 255.0
    images = images.reshape(images.shape + (1,))
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
