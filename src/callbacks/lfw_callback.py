from keras.callbacks import Callback
from keras.models import Model
from validate_lfw import get_lfw_accuracy


class LFWCallback(Callback):

    def on_epoch_end(self, epoch, logs):
        print('Validating on LFW')
        prod_model = Model(self.model.inputs[0], self.model.get_layer(name='embeddings').output)
        args = {
            'flip_images':     False,
            'subtract_mean':   True,
            'cosine_distance': True,
            'l2_normalize':    True,
            'subset':          False
        }
        acc, acc_std, best_threshold = get_lfw_accuracy(prod_model, **args)
        logs['lfw_acc'] = acc
        logs['lfw_acc_std'] = acc_std
        logs['lfw_threshold'] = best_threshold
        print('acc={:.4f} std={:.4f} threshold={:.4f}'.format(acc, acc_std, best_threshold))
