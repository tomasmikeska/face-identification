import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import to_categorical
from model import load_model
from dataset import get_image_mapping, train_data_generator, get_dev_data
from dataset import get_number_of_indetities, get_train_size
from utils import relative_path
from constants import INPUT_SHAPE, BATCH_SIZE, EPOCH_PARTITION, EMBEDDING_SIZE
from constants import EPOCHS, VGG_BB_TRAIN_MAP, MODEL_SAVE_PATH, TB_LOGS


def train(model, image_mapping_df):
    train_seq = train_data_generator(image_mapping_df, BATCH_SIZE)
    train_size = get_train_size(image_mapping_df)
    dev_data = get_dev_data(image_mapping_df)

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH + 'xception_centerloss_weights_{epoch:02d}-{val_softmax_out_acc:.3f}.h5',
                        monitor='val_softmax_out_acc',
                        save_weights_only=True,
                        verbose=1),
        TensorBoard(log_dir=TB_LOGS,
                    histogram_freq=1)
    ]

    model.fit_generator(train_seq,
                        steps_per_epoch=int(train_size * EPOCH_PARTITION) // BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=dev_data,
                        callbacks=callbacks)


if __name__ == '__main__':
    image_mapping_df = get_image_mapping(VGG_BB_TRAIN_MAP)
    n_identities = get_number_of_indetities(image_mapping_df)
    print('n_identities: %s' % n_identities)
    print('n_images: %s' % len(image_mapping_df))
    # Train
    model = load_model(INPUT_SHAPE, n_identities, EMBEDDING_SIZE)
    train(model, image_mapping_df)
