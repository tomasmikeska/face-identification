import keras.backend as K
from keras.layers import Layer


class BatchRenormalization(Layer):

    def __init__(self,
                 axis=-1,
                 r_max=3.,
                 d_max=5.,
                 alpha=0.01,
                 epsilon=1e-3,
                 t_delta=0.1,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 moving_mean_initializer='zeros',
                 moving_var_initializer='zeros',
                 **kwargs):
        super(BatchRenormalization, self).__init__(**kwargs)
        # Hyperparameters
        self.axis        = axis
        self.r_max_final = r_max
        self.d_max_final = d_max
        self.alpha       = alpha
        self.epsilon     = epsilon
        self.t_delta     = t_delta
        # Initializers
        self.gamma_initializer       = gamma_initializer
        self.beta_initializer        = beta_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_var_initializer  = moving_var_initializer

    def build(self, input_shape):
        dim = input_shape[self.axis]

        self.r_max = K.variable(1, name='{}_r_max'.format(self.name))
        self.d_max = K.variable(0, name='{}_d_max'.format(self.name))
        self.t     = K.variable(0, name='{}_t'.format(self.name))
        self.gamma = self.add_weight(shape=(dim,),
                                     initializer=self.gamma_initializer,
                                     name='{}_gamma'.format(self.name),
                                     trainable=True)
        self.beta  = self.add_weight(shape=(dim,),
                                     initializer=self.beta_initializer,
                                     name='{}_beta'.format(self.name),
                                     trainable=True)
        self.mean  = self.add_weight(shape=(dim,),
                                     initializer=self.moving_mean_initializer,
                                     name='{}_running_mean'.format(self.name),
                                     trainable=False)
        self.var   = self.add_weight(shape=(dim,),
                                     initializer=self.moving_var_initializer,
                                     name='{}_running_var'.format(self.name),
                                     trainable=False)
        super(BatchRenormalization, self).build(input_shape)

    def call(self, x, training=None):
        if training:
            batch_mean = K.mean(x, axis=self.axis, keepdims=True)
            batch_var  = K.var(x, axis=self.axis, keepdims=True)
            r = K.stop_gradient(K.clip(batch_var / (self.var + self.epsilon), 1. / self.r_max, self.r_max))
            d = K.stop_gradient(K.clip((batch_mean - self.mean) / (var + self.epsilon), -self.d_max, self.d_max))
            x_hat  = (x - batch_mean) * r / batch_var - d
            # Running average of mean and variance
            mean_update = self.mean + self.alpha * (batch_mean - self.mean)
            var_update  = self.var  + self.alpha * (batch_var  - self.var)
            self.add_update((self.mean, mean_update), x)
            self.add_update((self.var, var_update), x)
            # Update r and d params
            r_val = self.r_max_final / (1 + (self.r_max_final - 1) * K.exp(-self.t))
            d_val = self.d_max_final / (1 + ((self.d_max_final / 1e-3) - 1) * K.exp(-(2 * self.t)))
            self.add_update([K.update(self.r_max, r_val),
                             K.update(self.d_max, d_val),
                             K.update(self.t, self.t + self.t_delta)], x)
            # y_i = γ * (x_i - μ) + β
            output = self.gamma * x_hat + self.beta
        else: # Inference
            # y_i = (x_i - μ_b) * r / σ_b + d
            output = self.gamma * (x - self.mean) / (self.var + self.epsilon) + self.beta

        return output
