import keras.backend as K
from keras.engine.topology import Layer


class CenterLossLayer(Layer):

    def __init__(self, n_classes, embedding_size, alpha, **kwargs):
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
