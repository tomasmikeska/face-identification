import numpy as np
import squeezenet
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from lfw_dataset import load_dataset, person_image_mapping
from utils import relative_path, file_exists
from constants import MODEL_SAVE_PATH


BATCH_SIZE = 128


data_generator = ImageDataGenerator(rotation_range=15,
                                    shear_range=0.01,
                                    horizontal_flip=True)


def load_model(n_classes, input_shape):
    model = squeezenet.create_model(n_classes, input_shape)
    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model


def get_dataset_split(X, y):
    person_image = person_image_mapping(X, y)
    X_train = []
    X_test  = []
    y_train = []
    y_test  = []

    for person, images in person_image.items():
        split = int(round(len(images) * 0.1))
        X_test  = X_test  + images[:split]
        y_test  = y_test  + [person for _ in images[:split]]
        X_train = X_train + images[split:]
        y_train = y_train + [person for _ in images[split:]]

    return np.array(X_train), np.array(X_test), to_categorical(y_train), to_categorical(y_test)


def train(model, X, y):
    X_train, X_test, y_train, y_test = get_dataset_split(X, y)
    X_train, y_train = shuffle(X_train, y_train)
    data_generator.fit(X_train)
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH + 'squeezenet_softmax_weights.h5',
                        monitor='val_loss',
                        verbose=1,
                        save_weights_only=True,
                        save_best_only=True,
                        period=3),
        # EarlyStopping(monitor='val_loss', min_delta=0.001, patience=30, verbose=0),
        TensorBoard(log_dir=relative_path('../tb_logs'), histogram_freq=1)
    ]
    model.fit_generator(data_generator.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=len(X_train) / BATCH_SIZE,
                        epochs=200,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks)


if __name__ == '__main__':
    # Train
    X, y = load_dataset()
    n_classes = len(set(y))
    input_shape = X.shape[1:]
    model = load_model(n_classes, input_shape)
    train(model, X, y)
