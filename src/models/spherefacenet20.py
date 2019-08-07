from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Add, PReLU, Flatten
from keras.initializers import TruncatedNormal
from keras.models import Model
from keras import regularizers


def ResBlock(n, filters, input_tensor):
    x = Conv2D(filters,
               kernel_size=3,
               strides=2,
               padding='same',
               kernel_initializer='glorot_uniform',
               use_bias=True,
               bias_initializer='zeros')(input_tensor)
    x = PReLU(alpha_initializer='zeros')(x)

    for _ in range(n):
        block = Conv2D(filters,
                       kernel_size=(3, 3),
                       padding='same',
                       kernel_initializer=TruncatedNormal(stddev=0.1))(x)
        block = PReLU(alpha_initializer='zeros')(block)
        block = Conv2D(filters,
                       kernel_size=(3, 3),
                       padding='same',
                       kernel_initializer=TruncatedNormal(stddev=0.1))(x)
        block = PReLU(alpha_initializer='zeros')(block)

        x = Add()([block, x])

    return x


def SphereFaceNet20(embedding_size, input_tensor):
    x = ResBlock(n=1, filters=64,  input_tensor=input_tensor)
    x = ResBlock(n=2, filters=128, input_tensor=x)
    x = ResBlock(n=4, filters=256, input_tensor=x)
    x = ResBlock(n=1, filters=512, input_tensor=x)
    x = Flatten()(x)

    x = Dense(embedding_size,
              name='embeddings',
              use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(1e-4))(x)

    return x
