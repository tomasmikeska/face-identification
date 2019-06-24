import argparse
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
from sklearn.svm import LinearSVC
from PIL import Image
from keras.models import load_model
from keras.utils import to_categorical
from model import preprocess_input
from utils import file_listing
from constants import INPUT_SHAPE, LFW_PATH, LFW_PAIRS_PATH
from constants import MIN_IMG_WIDTH, MIN_IMG_HEIGHT, TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT


def center_dist(face_bb, img_np):
    '''Calculates distances of face bounding box from center of image

    Args:
        face_bb (object): MTCNN bounding box
        img_np (numpy array): Image RGB data
    '''
    face_x, face_y, face_w, face_h = face_bb['box']
    img_h, img_w, _ = img_np.shape
    return (face_y + face_h / 2. - img_h / 2.)**2 + (face_x + face_w / 2. - img_w / 2.)**2


def extract_face(img_np, face_detector):
    '''Detect face, crop it and resize to target width and height'''
    faces = face_detector.detect_faces(img_np)

    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: center_dist(x, img_np))
        x, y, width, height = faces[0]['box']

        # Deltas - distances to match target bounding box
        hor_delta = int(max(0, height - width) / 2)
        ver_delta = int(max(0, width - height) / 2)

        img_pil = Image.fromarray(img_np)
        img_pil = img_pil.crop((x - hor_delta, y - ver_delta, x + width + hor_delta, y + height + ver_delta))
        img_pil = img_pil.resize((TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT), Image.ANTIALIAS)

        return np.array(img_pil)


def read_image(path, face_detector):
    img_np = plt.imread(path)
    face = extract_face(img_np, face_detector)
    if face is not None:
        img_np = face.reshape(INPUT_SHAPE)
        return preprocess_input(img_np)


def calculate_distances(model, pairs, face_detector):
    pair_dists = []
    pair_labels = []

    for img_l, img_r, is_same_person in tqdm(pairs):
        img_l_np = read_image(img_l, face_detector)
        img_r_np = read_image(img_r, face_detector)

        if img_l_np is None or img_r_np is None:
            continue

        emb_l = model.predict(img_l_np.reshape((1,) + INPUT_SHAPE))
        emb_r = model.predict(img_r_np.reshape((1,) + INPUT_SHAPE))
        dist  = np.linalg.norm(emb_l - emb_r)

        pair_dists.append(dist)
        pair_labels.append(1 if is_same_person else 0)

    return pair_dists, pair_labels


def evaluate(model, pairs, face_detector):
    pair_dists, pair_labels = calculate_distances(model, pairs, face_detector)
    pair_dists = np.reshape(pair_dists, (len(pair_dists), 1))
    # Train binary classifier to get threshold value and separate classes
    clf = LinearSVC()
    clf.fit(pair_dists, pair_labels)
    acc = clf.score(pair_dists, pair_labels)
    return acc


def read_pairs(pairs_meta_path):
    '''Read LFW pairs in [(img_left_np, img_right_np, is_same_person)] format'''
    pairs = []
    with open(pairs_meta_path, 'r') as f:
        for line in f.readlines():
            line_stripped = line.strip().split()

            if len(line_stripped) == 3:  # Same person line - <name> <img_l_id> <img_r_id>
                name  = line_stripped[0]
                img_l = int(line_stripped[1]) - 1
                img_r = int(line_stripped[2]) - 1
                imgs  = file_listing(LFW_PATH + name, 'jpg')
                pairs.append((imgs[img_l], imgs[img_r], True))
            elif len(line_stripped) == 4:  # Different people line - <name_l> <img_l_id> <name_r> <img_r_id>
                name_l = line_stripped[0]
                img_l  = int(line_stripped[1]) - 1
                name_r = line_stripped[2]
                img_r  = int(line_stripped[3]) - 1
                imgs_l = file_listing(LFW_PATH + name_l, 'jpg')[img_l]
                imgs_r = file_listing(LFW_PATH + name_r, 'jpg')[img_r]
                pairs.append((imgs_l, imgs_r, name_l == name_r))

    return pairs


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Validate trained model on LFW dataset')
    parser.add_argument('--model', type=str, help='Converted model path')
    args = parser.parse_args()
    # Evaluate
    face_detector = MTCNN()  # Model used for detectign face bounding boxes
    model = load_model(args.model)  # Trained face verification model
    pairs = read_pairs(LFW_PAIRS_PATH)  # LFW validation pairs
    acc = evaluate(model, pairs, face_detector)
    print('Accuracy: %s' % acc)
