from keras.layers import Dense, Conv2D, MaxPool2D, BatchNormalization, PReLU, DepthwiseConv2D, GlobalAveragePooling2D
from keras.models import Model


def get_effnet_post(inputs):
    x = PReLU()(inputs)
    x = BatchNormalization()(x)
    return x


def get_effnet_block(inputs, channels_in, channels_out):
    x = Conv2D(channels_in,
               kernel_size=(1, 1),
               padding='same',
               use_bias=False)(inputs)
    x = get_effnet_post(x)

    x = DepthwiseConv2D(kernel_size=(1, 3), padding='same', use_bias=False)(x)
    x = get_effnet_post(x)
    # Separable pooling
    x = MaxPool2D(pool_size=(2, 1),
                  strides=(2, 1))(x)

    x = DepthwiseConv2D(kernel_size=(3, 1),
                        padding='same',
                        use_bias=False)(x)
    x = get_effnet_post(x)

    x = Conv2D(channels_out,
               kernel_size=(2, 1),
               strides=(1, 2),
               padding='same',
               use_bias=False)(x)
    x = get_effnet_post(x)

    return x


def Effnet(inputs, embedding_size):
    x = get_effnet_block(inputs, 32, 64)
    x = get_effnet_block(x, 64, 128)
    x = get_effnet_block(x, 128, 256)

    x = GlobalAveragePooling2D()(x)
    x = Dense(embedding_size)(x)
    x = PReLU()(x)

    model = Model(inputs=inputs, outputs=x, name='EffNet')

    return model
