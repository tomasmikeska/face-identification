import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.utils import shuffle
from tqdm import tqdm
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from model import preprocess_input
from constants import MIN_IMG_WIDTH, MIN_IMG_HEIGHT, TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT, MIN_FACES_PER_PERSON
from constants import VGG_TRAIN_PATH, VGG_TEST_PATH, INPUT_SHAPE, DEV_FACES_PER_PERSON, INPUT_SHAPE
from utils import file_listing, dir_listing, last_component, get_file_name, relative_path, mkdir


def extract_face(img_np, bb):
    x, y, width, height = bb.X, bb.Y, bb.W, bb.H

    if width / height > TARGET_IMG_WIDTH / TARGET_IMG_HEIGHT:
        fin_width = width
        fin_height = int(fin_width * (TARGET_IMG_HEIGHT / float(TARGET_IMG_WIDTH)))
        fin_x = x
        fin_y = y - (fin_height - height) / 2
    else:
        fin_height = height
        fin_width = int(fin_height * (TARGET_IMG_WIDTH / float(TARGET_IMG_HEIGHT)))
        fin_x = x - (fin_width - width) / 2
        fin_y = y

    img_pil = Image.fromarray(img_np)
    img_pil = img_pil.crop((fin_x, fin_y, fin_x + fin_width, y + fin_height))
    img_pil = img_pil.resize((TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT), Image.ANTIALIAS)
    return np.array(img_pil)


def read_image(path, bb, image_generator=None):
    img_np = plt.imread(path)
    img_np = extract_face(img_np, bb)
    if image_generator:
        img_np = image_generator.random_transform(img_np)
    return preprocess_input(img_np)


def get_image_mapping(bb_path):
    df = pd.read_csv(bb_path)
    # Filter small faces
    df = df[df.W >= MIN_IMG_WIDTH]
    df = df[df.H >= MIN_IMG_HEIGHT]
    # Split NAME_ID column to NAME and ID
    df['NAME'], df['ID'] = df['NAME_ID'].str.split('/', 1).str
    # Filter identities with less than minimum num of faces
    df['COUNT'] = df.groupby('NAME')['NAME'].transform('count')
    df = df[df['COUNT'] > MIN_FACES_PER_PERSON]
    # Tag first images as dev hold-out set
    df['DEV'] = False
    dev_indices = df.groupby('NAME').head(DEV_FACES_PER_PERSON).index
    df.loc[dev_indices, 'DEV'] = True
    return df


def get_identities(image_mapping_df):
    return image_mapping_df['NAME'].unique()


def get_number_of_indetities(image_mapping_df):
    return len(get_identities(image_mapping_df))


def get_train_size(image_mapping_df):
    return len(image_mapping_df[image_mapping_df['DEV'] == False])


def train_data_generator(image_mapping_df, batch_size):
    image_generator = ImageDataGenerator(horizontal_flip=True)
    identities      = get_identities(image_mapping_df)
    n_identities    = len(identities)
    train_items     = image_mapping_df[image_mapping_df['DEV'] == False]

    while True:
        items = train_items.sample(n=batch_size, weights=(1 / image_mapping_df['COUNT']))
        X = []
        y = []

        for i, row in items.iterrows():
            file = VGG_TRAIN_PATH + '/' + row['NAME_ID'] + '.jpg'
            X.append(read_image(file, row, image_generator))
            y.append(np.argmax(identities == row['NAME']))

        y_categorical = to_categorical(y, num_classes=n_identities)

        yield [np.array(X), y_categorical], [y_categorical, np.random.rand(len(X), 1)]


def get_dev_data(image_mapping_df):
    identities   = get_identities(image_mapping_df)
    n_identities = len(identities)
    X = []
    y = []

    for i, row in image_mapping_df[image_mapping_df['DEV'] == True].iterrows():
        file = VGG_TRAIN_PATH + '/' + row['NAME_ID'] + '.jpg'
        X.append(read_image(file, row))
        y.append(np.argmax(identities == row['NAME']))

    X, y = shuffle(X, y)
    y_categorical = to_categorical(y, num_classes=n_identities)
    
    return [np.array(X), y_categorical], [y_categorical, np.random.rand(len(X), 1)]
