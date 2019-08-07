from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dense, Dropout, Lambda, Flatten, Add
from keras.models import Model
import keras.backend as K


def mfm_linear(units):
    def fn(x):
        x1 = Dense(units)(x)
        x2 = Dense(units)(x)
        return Lambda(lambda x: K.max(x, axis=0))([x1, x2])
    return fn


def mfm_conv(filters, kernel_size=3, **kwargs):
    def fn(x):
        x1 = Conv2D(filters // 2, kernel_size, **kwargs)(x)
        x2 = Conv2D(filters // 2, kernel_size, **kwargs)(x)
        return Lambda(lambda x: K.max(x, axis=0))([x1, x2])
    return fn


def group(in_channels, out_channels, kernel_size=3):
    def fn(x):
        x = mfm_conv(in_channels, kernel_size=1)(x)
        x = mfm_conv(out_channels, kernel_size=kernel_size, padding='same')(x)
        return x
    return fn


def resblock(channels):
    def fn(x):
        out = mfm_conv(channels, kernel_size=3, padding='same')(x)
        out = mfm_conv(channels, kernel_size=3, padding='same')(out)
        return Add()([out, x])
    return fn


def LightCNN_29(input_tensor):
    x = mfm_conv(48, kernel_size=5)(input_tensor)
    x = Add()([MaxPooling2D()(x), AveragePooling2D()(x)])

    x = resblock(48)(x)
    x = group(in_channels=48, out_channels=96)(x)
    x = Add()([MaxPooling2D()(x), AveragePooling2D()(x)])

    x = resblock(96)(x)
    x = resblock(96)(x)
    x = group(in_channels=96, out_channels=192)(x)
    x = Add()([MaxPooling2D()(x), AveragePooling2D()(x)])

    x = resblock(192)(x)
    x = resblock(192)(x)
    x = resblock(192)(x)
    x = group(in_channels=192, out_channels=128)(x)

    x = resblock(128)(x)
    x = resblock(128)(x)
    x = resblock(128)(x)
    x = resblock(128)(x)
    x = group(in_channels=128, out_channels=128)(x)
    x = Add()([MaxPooling2D()(x), AveragePooling2D()(x)])

    out = Flatten()(x)

    return out


def LightCNN_9(embedding_size, input_tensor):
    x = mfm_conv(48, kernel_size=5)(input_tensor)
    x = MaxPooling2D()(x)
    x = group(in_channels=48, out_channels=96)(x)
    x = MaxPooling2D()(x)
    x = group(in_channels=96, out_channels=192)(x)
    x = MaxPooling2D()(x)
    x = group(in_channels=192, out_channels=128)(x)
    x = group(in_channels=128, out_channels=128)(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = mfm_linear(embedding_size)(x)

    return x
