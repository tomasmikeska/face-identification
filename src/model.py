import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D, Lambda, Input, Flatten
from keras.models import Model
from keras.applications.xception import Xception
import tensorflow as tf
import keras.backend as K
from keras.engine.topology import Layer
from metrics import dominant_label_metric, confidence_metric


LAMBDA = 0.03
ALPHA  = 0.5


class CenterLossLayer(Layer):

    def __init__(self, n_classes=10, embedding_size=2, alpha=ALPHA, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.emb_size = embedding_size
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.n_classes, self.emb_size),
                                       initializer='uniform',
                                       trainable=False)
        super().build(input_shape)

    def call(self, inputs, mask=None):
        x, targets = inputs
        # center_count = 1 + Σ_m_i δ(y_i = j)
        center_count = K.sum(K.transpose(targets), axis=1, keepdims=True) + 1
        # Δc_j = (Σ_m_i δ(y_i = j) * (c_j - x_i)) / center_count
        delta_centers = K.dot(K.transpose(targets), K.dot(targets, self.centers) - x) / center_count
        # c_j = c_j - α * Δc_j
        updated_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, updated_centers), x)
        # L = (1/2) * Σ_m_i (x_i - c_yi)^2
        self.loss = x - K.dot(targets, self.centers)
        self.loss = K.sum(self.loss * self.loss, axis=1, keepdims=True)
        return self.loss

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.loss)


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
