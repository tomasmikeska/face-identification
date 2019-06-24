import math
import keras.backend as K


def dominant_label_metric():
    '''Measures percentage of most common output label'''
    def dominant_label_prob(y_label, y_pred):
        onehot = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_pred.shape[1])
        p = K.mean(onehot, axis=0, keepdims=True)[0]
        return K.max(p)
    return dominant_label_prob


def confidence_metric():
    '''Measures avg output confidence'''
    def confidence(y_label, y_pred):
        return K.mean(K.max(y_pred, axis=1, keepdims=True))
    return confidence
