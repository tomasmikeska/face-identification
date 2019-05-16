import numpy as np
import os
import cv2
from skimage.transform import resize
from scipy.spatial import distance
from utils import file_listing, dir_listing, last_component
from utils import k_nearest, most_common
from squeezenet import create_model
from constants import MODEL_SAVE_PATH, LOCAL_TRAIN_DIR, LOCAL_TEST_DIR


input_shape = (100, 75, 1)


model = create_model(input_shape)
model.load_weights(MODEL_SAVE_PATH + 'facenet_squeezenet_weights.h5')


def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return resize(img, (input_shape[0], input_shape[1]), mode='reflect')


def calc_embedding(filepath):
    img = load_image(filepath)
    return model.predict(img.reshape((1,) + input_shape))


def predict(embeddings, target_img_path):
    distances = []
    target_emb = calc_embedding(target_img_path)

    for dir_path in dir_listing(LOCAL_TRAIN_DIR):
        for img_path in file_listing(dir_path, 'png'):
            dist = distance.euclidean(target_emb, embeddings[img_path])
            distances.append((dist, last_component(dir_path)))

    nearest_buckets = k_nearest(3, distances)
    return most_common(nearest_buckets)


def calc_all_embeddings():
    # Calculate embeddings
    embeddings = {}
    for dir_path in dir_listing(LOCAL_TRAIN_DIR):
        for filepath in file_listing(dir_path, 'png'):
            embeddings[filepath] = calc_embedding(filepath)
    return embeddings


if __name__ == '__main__':
    embeddings = calc_all_embeddings()
    # Calculate accuracy
    total = 0
    correct = 0

    for dir_path in dir_listing(LOCAL_TEST_DIR):
        for img_path in file_listing(dir_path, 'png'):
            total += 1
            prediction = predict(embeddings, img_path)
            if prediction == last_component(dir_path):
                correct += 1
            else:
                print('Wrong: %s' % img_path)

    print('Got %s correct out of %s' % (correct, total))
    print('-> %s percent accuracy' % ((correct / total) * 100))
