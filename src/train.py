import os
import numpy as np
import squeezenet
import facenet
import gc
from itertools import combinations
from sklearn.utils import shuffle
from dataset import load_dataset
from facenet import triplet_loss, accuracy
from utils import relative_path


EPOCHS = 24
EMBEDDING_SIZE = 128
MODEL_SAVE_PATH = os.environ.get('MODEL_SAVE_PATH', relative_path('../model/'))
MAX_TRIPLET_LIST_SIZE = 500
SEMIHARD_MARGIN = 0.2


def load_model(input_shape):
    base_model = squeezenet.create_model(input_shape, EMBEDDING_SIZE)
    base_model.load_weights(relative_path('facenet_squeezenet_weights.h5')) # TODO: remove incremental learning
    facenet_model = facenet.create_model(base_model, input_shape, EMBEDDING_SIZE)
    facenet_model.compile(optimizer='adam',
                          loss=triplet_loss,
                          metrics=[accuracy])
    return facenet_model, base_model


def is_semihard(anchor, pos, neg):
    return np.linalg.norm(anchor - pos) + SEMIHARD_MARGIN > np.linalg.norm(anchor - neg)


def get_semihards(batch, embeddings, anchor_i, pos_i, person_target):
    semihards = []
    for i, person in enumerate(batch['y']):
        if person != person_target and is_semihard(embeddings[anchor_i], embeddings[pos_i], embeddings[i]):
            semihards.append(batch['X'][i])
    return np.array(semihards)


def get_positives(batch, person_target):
    pos_indices = []
    for i, person in enumerate(batch['y']):
        if person == person_target:
            pos_indices.append(i)
    return np.array(pos_indices)


def get_triplets(batch, embeddings, person_list):
    triplets = [[], [], []]
    for person in person_list:
        pos_indices = get_positives(batch, person)
        pos_tuples = list(combinations(pos_indices, 2))
        for anchor_i, pos_i in pos_tuples:
            semihards = get_semihards(batch, embeddings, anchor_i, pos_i, person)
            for semihard in semihards:
                triplets[0].append(batch['X'][anchor_i])
                triplets[1].append(batch['X'][pos_i])
                triplets[2].append(semihard)
    return [np.array(triplets[0]), np.array(triplets[1]), np.array(triplets[2])]


def train(siamese_model, base_model, batches, person_list):
    for epoch in range(EPOCHS):
        print('-- EPOCH: %s' % (epoch + 1))
        for i, batch in enumerate(batches):
            print('Training batch %s out of %s' % (i + 1, len(batches)))
            embeddings = base_model.predict(batch['X'])
            triplets = get_triplets(batch, embeddings, person_list)
            print('Triplets size: %s' % len(triplets[0]))
            anchor, pos, neg = shuffle(triplets[0], triplets[1], triplets[2])

            for i in range(0, len(triplets[0]), MAX_TRIPLET_LIST_SIZE):
                anchor_sub = anchor[i:i+MAX_TRIPLET_LIST_SIZE]
                pos_sub = pos[i:i+MAX_TRIPLET_LIST_SIZE]
                neg_sub = neg[i:i+MAX_TRIPLET_LIST_SIZE]
                _, acc = siamese_model.train_on_batch([anchor_sub, pos_sub, neg_sub], np.zeros(len(anchor_sub)))
                print('Accuracy on triplet sublist: %s' % acc)


def test():
    pass


if __name__ == '__main__':
    # Train
    batches, person_list, input_shape = load_dataset()
    facenet_model, base_model = load_model(input_shape)
    train(facenet_model, base_model, batches, person_list)
    test()
    base_model.save_weights(MODEL_SAVE_PATH + 'facenet_squeezenet_weights.h5')
