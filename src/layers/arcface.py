import math
import keras.backend as K
from keras.layers import Layer
from keras import regularizers
import tensorflow as tf


class ArcFace(Layer):
    def __init__(self, n_classes, s=64.0, m=0.5, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='he_normal',
                                 trainable=True,
                                 regularizer=self.regularizer)
        super(ArcFace, self).build(input_shape[0])

    def call(self, inputs):
        x, y = inputs
        # l2 normalize to unit hypersphere
        x = K.l2_normalize(x, axis=1)
        W = K.l2_normalize(self.W, axis=0)
        # cos theta
        logits = K.dot(x, W)
        # target logits
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = K.cos(theta + self.m)
        # arcface logits
        logits = (1 - y) * logits + y * target_logits
        # feature re-scale
        logits *= self.s
        return K.softmax(logits)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.n_classes)
