import numpy as np
from keras.layers import BatchNormalization, Dropout, Dense, Input
from keras.models import Model
import tensorflow as tf
import keras.backend as K
from keras import regularizers
from keras.applications.densenet import DenseNet121
from layers.arcface import ArcFace
from constants import ARCFACE_M, ARCFACE_S


def load_model(input_shape, n_classes, embedding_size):
    '''Create face identification model

    Args:
        input_shape (tuple): input image shape, only one of 2 model inputs (it takes targets as well)
        n_classes (int): Number of identities
        embedding_size (int): Size of embedding vector
    '''
    img_input = Input(shape=input_shape)
    targets   = Input(shape=(n_classes,))
    # Base feature extractor model
    base_model = DenseNet121(include_top=False,
                             weights=None,
                             input_tensor=img_input,
                             pooling='avg')
    x = base_model.output
    # Embeddings - BN+Dropout+FC+BN
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(embedding_size,
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(1e-4),
              use_bias=False)(x)
    x = BatchNormalization(name='embeddings', scale=False)(x)
    # ArcFace softmax layer
    out = ArcFace(n_classes,
                  m=ARCFACE_M,
                  s=ARCFACE_S,
                  regularizer=regularizers.l2(1e-4))([x, targets])
    # Create model
    model = Model([img_input, targets], out)

    return model


def preprocess_input(x):
    '''Scale RGB image to [-1, 1] range'''
    return x / 127.5 - 1.
