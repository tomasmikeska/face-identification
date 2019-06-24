import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D, Lambda, Input, Flatten
from keras.models import Model
from keras.applications.xception import Xception
import tensorflow as tf
import keras.backend as K
from center_loss import CenterLossLayer
from metrics import dominant_label_metric, confidence_metric


LAMBDA = 0.03
ALPHA  = 0.5


def load_model(input_shape, n_classes, embedding_size):
    targets = Input(shape=(n_classes,))

    base_model = Xception(include_top=False,
                          weights=None,
                          input_shape=input_shape,
                          pooling='avg')
    x = base_model.output
    emb_out = Dense(embedding_size, name='emb_out')(x)

    softmax_out = Dense(n_classes,
                        activation='softmax',
                        name='softmax_out')(emb_out)
    # L2 normalization
    l2_normalized = Lambda(lambda x: x / K.sqrt(K.sum(x * x, axis=1, keepdims=True)))(emb_out)

    center_loss = CenterLossLayer(n_classes=n_classes,
                                  embedding_size=embedding_size,
                                  alpha=LAMBDA,
                                  name='centerloss_out')([l2_normalized, targets])

    model = Model(inputs=[base_model.input, targets],
                  outputs=[softmax_out, center_loss])
    model.compile(optimizer='nadam',
                  loss=['categorical_crossentropy', lambda y_true, y_pred: y_pred],
                  loss_weights=[1.0, LAMBDA],  # L = L_softmax + λ * L_center
                  metrics=['accuracy', dominant_label_metric(), confidence_metric()])

    return model


def preprocess_input(x):
    return x / 255.0
