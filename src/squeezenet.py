from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Lambda, concatenate, Input, BatchNormalization, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from constants import EMBEDDING_SIZE


def fire(x, squeeze=16, expand=64):
    x = Conv2D(squeeze, (1,1), padding='valid', activation='relu')(x)
    left = Conv2D(expand, (1,1), padding='valid', activation='relu')(x)
    right = Conv2D(expand, (3,3), padding='same', activation='relu')(x)
    return concatenate([left, right], axis=3)


def create_model(input_shape):
    img_input = Input(input_shape)
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='valid')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = fire(x, squeeze=16, expand=16)
    x = fire(x, squeeze=16, expand=16)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = fire(x, squeeze=32, expand=32)
    x = fire(x, squeeze=32, expand=32)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = fire(x, squeeze=48, expand=48)
    x = fire(x, squeeze=48, expand=48)
    x = fire(x, squeeze=64, expand=64)
    x = fire(x, squeeze=64, expand=64)
    x = Dropout(0.2)(x)
    x = Conv2D(512, (1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(EMBEDDING_SIZE, activation='sigmoid')(x)

    return Model(img_input, out)
