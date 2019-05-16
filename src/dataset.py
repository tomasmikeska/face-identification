import os
import numpy as np
import random
from sklearn.datasets import fetch_lfw_people
from utils import relative_path, file_exists


NUM_CHANNELS = 1
MIN_FACES_PER_PERSON = 21
MIN_FACES_IN_BATCH = 7
DATASET_COMPRESSED_PATH = os.environ.get('DATASET', relative_path('../data/dataset.npy'))


def normalize_scale(x):
    '''Scale grayscale image <0,255> ints to <0,1> floats'''
    return x / 255.0


def transform_img(img):
    return normalize_scale(img)


def person_image_mapping(images, targets, person_list):
    person_images = { person: [] for person in person_list }

    for i in range(0, images.shape[0]):
        transformed_img = transform_img(images[i])
        person = person_list[targets[i]]
        person_images[person].append(transformed_img)

    return person_images


def load_batches(person_images, input_shape):
    batches = []

    for i in range(0, MIN_FACES_PER_PERSON // MIN_FACES_IN_BATCH):
        X_batch = []
        y_batch = []

        for person in person_images.keys():
            random.shuffle(person_images[person])
            for i in range(0, MIN_FACES_IN_BATCH):
                X_batch.append(person_images[person][i].reshape(input_shape))
                y_batch.append(person)

        batches.append({ 'X': np.array(X_batch), 'y': y_batch })

    return batches


def fetch_dataset():
    lfw_people = fetch_lfw_people(min_faces_per_person=MIN_FACES_PER_PERSON, resize=0.8)
    targets = lfw_people.target
    person_list = lfw_people.target_names.tolist()
    _, image_height, image_width = lfw_people.images.shape
    input_shape = (image_height, image_width, NUM_CHANNELS)

    person_image_dict = person_image_mapping(lfw_people.images, targets, person_list)
    batches = load_batches(person_image_dict, input_shape)
    return batches, person_list, input_shape


def load_dataset():
    if file_exists(DATASET_COMPRESSED_PATH):
        return np.load(DATASET_COMPRESSED_PATH, allow_pickle=True)
    else:
        return fetch_dataset()


if __name__ == '__main__':
    dataset = fetch_dataset()
    np.save(DATASET_COMPRESSED_PATH, dataset)
    print('Dataset saved to %s' % (DATASET_COMPRESSED_PATH))
