import numpy as np
import itertools
from itertools import cycle, combinations
import random
from sklearn.utils import shuffle
from tensorflow.keras.layers import Input, Lambda, concatenate, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow.keras.backend as K
from constants import EMBEDDING_SIZE, TRIPLET_LOSS_ALPHA


def euclidean_distance(embeddings):
    x, y = embeddings
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def triplet_loss(_unused, stacked_dists):
    alpha     = K.constant(TRIPLET_LOSS_ALPHA)
    pos_dist  = stacked_dists[:,0,0]
    neg_dist  = stacked_dists[:,1,0]
    tert_dist = stacked_dists[:,2,0]

    return K.mean(K.maximum(K.constant(0),
                            pos_dist - 0.5*(neg_dist + tert_dist) + alpha))


def accuracy(_unused, stacked_dists):
    '''Compute acc as percentage of triplets that satisfy pos_dist < neg_dist'''
    return K.mean(stacked_dists[:,0,0] < stacked_dists[:,1,0])


def create_facenet_model(base_model, input_shape):
    # Add embedding top
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(EMBEDDING_SIZE, activation='sigmoid')(x)
    base_model = Model(base_model.input, x)

    input_anchor   = Input(shape=input_shape, name='input_anchor')
    input_positive = Input(shape=input_shape, name='input_positive')
    input_negative = Input(shape=input_shape, name='input_negative')

    normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')

    anchor   = normalize(base_model(input_anchor))
    positive = normalize(base_model(input_positive))
    negative = normalize(base_model(input_negative))

    # Compute L2 norms in model graph instead of in triplet loss
    # to be able to use them in metrics
    pos_dist  = Lambda(euclidean_distance)([anchor, positive])
    neg_dist  = Lambda(euclidean_distance)([anchor, negative])
    tert_dist = Lambda(euclidean_distance)([positive, negative])

    stacked_dists = Lambda(lambda vects: K.stack(vects, axis=1),
                           output_shape=(None, EMBEDDING_SIZE),
                           name='stacked_embeddings')([pos_dist, neg_dist, tert_dist])

    return Model([input_anchor, input_positive, input_negative], stacked_dists, name='siamese_net'), base_model
