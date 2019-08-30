import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle, islice
from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm
from keras.utils import to_categorical, Sequence
from keras.preprocessing.image import ImageDataGenerator
from extract_face import extract_face, augment_face
from model import preprocess_input
from constants import CASIA_PATH, INPUT_SHAPE, DEV_FACES_PER_PERSON, INPUT_SHAPE
from constants import MIN_FACES_PER_PERSON, MAX_FACES_PER_PERSON, MIN_FACES_UNSAMPLE


def read_image(row, augment=True):
    img_np = extract_face(CASIA_PATH + '/' + row.FILE, row)
    if img_np is None:
        return None
    if augment:
        img_np = augment_face(img_np)
    return preprocess_input(img_np)


def get_image_mapping(landmarks_path):
    df = pd.read_csv(landmarks_path, sep='\t| ', engine='python') # PATH, ID, ...landmarks
    # Filter identities with less than minimum num of faces
    df['COUNT'] = df.groupby('ID')['ID'].transform('count')
    df = df[df['COUNT'] >= MIN_FACES_PER_PERSON]
    # Tag first images as dev hold-out set
    df['DEV'] = False
    dev_indices = df.groupby('ID').head(DEV_FACES_PER_PERSON).index
    df.loc[dev_indices, 'DEV'] = True
    return df


def get_identities(image_mapping_df):
    return image_mapping_df['ID'].unique()


def get_number_of_indetities(image_mapping_df):
    return len(get_identities(image_mapping_df))


def get_train_items(image_mapping_df):
    items = []

    for identity in image_mapping_df[image_mapping_df['DEV'] == False]['ID'].unique():
        id_rows = []
        for i, row in image_mapping_df[image_mapping_df['ID'] == identity].iterrows():
            id_rows.append(row)
        sample_count = MIN_FACES_UNSAMPLE if len(id_rows) < MIN_FACES_UNSAMPLE else min(MAX_FACES_PER_PERSON, len(id_rows))
        for row in islice(cycle(id_rows), sample_count):
            items.append(row)

    return items


class CASIASequence(Sequence):
    def __init__(self, image_mapping_df, batch_size):
        self.batch_size   = batch_size
        self.items        = shuffle(get_train_items(image_mapping_df))
        self.identities   = get_identities(image_mapping_df)
        self.n_identities = len(self.identities)

    def get_labels(self):
        y = []
        for row in self.items:
            y.append(np.argmax(self.identities == row.ID))
        return np.array(y)

    def __len__(self):
        return len(self.items) // self.batch_size

    def __getitem__(self, idx):
        X = []
        y = []

        for row in self.items[idx * self.batch_size:(idx + 1) * self.batch_size]:
            img_np = read_image(row)

            if img_np is not None:
                X.append(img_np)
                y.append(np.argmax(self.identities == row.ID))

        y_categorical = to_categorical(y, num_classes=self.n_identities)

        return [np.array(X), y_categorical], y_categorical


def get_dev_data(image_mapping_df):
    identities   = get_identities(image_mapping_df)
    n_identities = len(identities)
    X = []
    y = []

    for i, row in image_mapping_df[image_mapping_df['DEV'] == True].iterrows():
        img_np = read_image(row, augment=False)

        if img_np is not None:
            X.append(img_np)
            y.append(np.argmax(identities == row['ID']))

    y_categorical = to_categorical(y, num_classes=n_identities)

    return [np.array(X), y_categorical], y_categorical
