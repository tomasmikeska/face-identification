import os
import numpy as np
from comet_ml import Experiment
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from model import load_model
from metrics import dominant_label_metric, confidence_metric
from datasets.casia_webface import get_image_mapping, CASIASequence, get_dev_data, get_number_of_indetities
from callbacks.lfw_callback import LFWCallback
from utils import relative_path
from constants import INPUT_SHAPE, BATCH_SIZE, EMBEDDING_SIZE, \
                      EPOCHS, CASIA_BB_MAP, MODEL_SAVE_PATH, \
                      MIN_FACES_PER_PERSON, MAX_FACES_PER_PERSON, MIN_FACES_UNSAMPLE, \
                      ARCFACE_M, ARCFACE_S


def train(model, image_mapping_df):
    train_seq    = CASIASequence(image_mapping_df, BATCH_SIZE)
    train_labels = train_seq.get_labels()
    dev_data     = get_dev_data(image_mapping_df)

    model.compile(optimizer=optimizers.SGD(lr=0.1, clipvalue=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', dominant_label_metric(), confidence_metric()])
    model.summary()

    callbacks = [
        TerminateOnNaN(),
        ModelCheckpoint(
            MODEL_SAVE_PATH + 'arcface_densenet121_{epoch:02d}-{val_acc:.3f}.h5',
            monitor='val_acc',
            save_weights_only=True,
            verbose=1),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3,
            min_delta=0.01,
            cooldown=3,
            min_lr=0.001,
            verbose=1),
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=5),
        LFWCallback()
    ]

    model.fit_generator(
        train_seq,
        epochs=EPOCHS,
        validation_data=dev_data,
        class_weight=compute_class_weight('balanced', np.unique(train_labels), train_labels),
        use_multiprocessing=True,
        workers=6,
        callbacks=callbacks)


if __name__ == '__main__':
    # Read csv landmark mapping
    image_mapping_df = get_image_mapping(CASIA_BB_MAP)
    n_identities = get_number_of_indetities(image_mapping_df)
    # CometML experiment
    if os.getenv('COMET_API_KEY'):
        experiment = Experiment(api_key=os.getenv('COMET_API_KEY'),
                                project_name=os.getenv('COMET_PROJECTNAME'),
                                workspace=os.getenv('COMET_WORKSPACE'))
        experiment.log_parameters({
            'batch_size':           BATCH_SIZE,
            'embedding_size':       EMBEDDING_SIZE,
            'n_identities':         n_identities,
            'input_shape':          INPUT_SHAPE,
            'min_faces_per_person': MIN_FACES_PER_PERSON,
            'max_faces_per_person': MAX_FACES_PER_PERSON,
            'min_faces_upsample':   MIN_FACES_UNSAMPLE,
            'arcface_m':            ARCFACE_M,
            'arcface_s':            ARCFACE_S
        })
    # Train
    model = load_model(INPUT_SHAPE, n_identities, EMBEDDING_SIZE)
    train(model, image_mapping_df)
