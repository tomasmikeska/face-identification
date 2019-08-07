import matplotlib.pyplot as plt
import argparse
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import Callback
from extract_face import extract_face
from model import preprocess_input
from utils import file_listing
from constants import INPUT_SHAPE, LFW_PATH, LFW_PAIRS_PATH, LFW_BB_MAP


N_FOLDS = 10


def read_lines(path):
    lines = []
    with open(path, 'r') as f:
        for line in f.readlines():
            lines.append(line)
    return lines


def read_pair(line):
    line_stripped = line.strip().split()

    if len(line_stripped) == 3:  # Same person line - <name> <img_l_id> <img_r_id>
        name  = line_stripped[0]
        img_l = int(line_stripped[1]) - 1
        img_r = int(line_stripped[2]) - 1
        imgs  = file_listing(LFW_PATH + '/' + name, 'jpg')
        return (imgs[img_l], imgs[img_r], True)
    elif len(line_stripped) == 4:  # Different people line - <name_l> <img_l_id> <name_r> <img_r_id>
        name_l = line_stripped[0]
        img_l  = int(line_stripped[1]) - 1
        name_r = line_stripped[2]
        img_r  = int(line_stripped[3]) - 1
        imgs_l = file_listing(LFW_PATH + '/' + name_l, 'jpg')[img_l]
        imgs_r = file_listing(LFW_PATH + '/' + name_r, 'jpg')[img_r]
        return (imgs_l, imgs_r, name_l == name_r)


def read_pairs(pairs_meta_path):
    '''Read LFW pairs in [(img_left_np, img_right_np, is_same_person)] format'''
    lines = read_lines(pairs_meta_path)[1:]
    return list(map(read_pair, lines))


def read_image(row):
    img_np = extract_face(LFW_PATH + '/' + row.FILE, row)
    return preprocess_input(img_np)


def read_images(imgs):
    df = pd.read_csv(LFW_BB_MAP, sep='\t| ', engine='python')
    X = []
    for img_path in imgs:
        file = '/'.join(img_path.split('/')[-2:])
        row = df[df['FILE'] == file].iloc[0]
        img_np = read_image(row)
        X.append(img_np)
    return np.array(X)


def get_unique_imgs(pairs):
    imgs = []
    for img_l, img_r, _ in pairs:
        imgs = imgs + [img_l, img_r]
    return list(set(imgs))


def get_embeddings(model, img_paths, **kwargs):
    images_np = read_images(img_paths)
    embeddings = model.predict(images_np)

    if kwargs['flip_images']:
        flip_embeddings = model.predict(np.flip(images_np, axis=2))
        embeddings = np.hstack([embeddings, flip_embeddings])

    if kwargs['subtract_mean']:
        embeddings = embeddings - np.mean(embeddings, axis=1, keepdims=True)

    if kwargs['l2_normalize']:
        embeddings = embeddings / np.sqrt(np.sum(embeddings * embeddings, axis=1, keepdims=True))

    return embeddings


def distance(a, b, cosine=False):
    if cosine:  # d = arccos( A • B / (||A|| • ||B||) ) / π
        norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-5
        similarity = a.dot(b.T) / norm
        return np.arccos(similarity) / math.pi
    else:  # d = ||A - B||
        return np.linalg.norm(a - b)


def calculate_distances(model, pairs, **kwargs):
    unique_images = get_unique_imgs(pairs)
    embeddings = get_embeddings(model, unique_images, **kwargs)
    distances  = []

    for img_l, img_r, is_same_person in pairs:
        emb_l = embeddings[unique_images.index(img_l)]
        emb_r = embeddings[unique_images.index(img_r)]
        dist  = distance(emb_l, emb_r, cosine=kwargs.get('cosine_distance'))
        distances.append((dist, is_same_person))

    return distances


def k_fold(n, n_folds):
    folds = []
    base  = list(range(n))
    step  = n // n_folds

    for fold_i in range(0, n, step):
        test  = base[fold_i:fold_i+step]
        train = list(set(base)-set(test))
        folds.append([train,test])

    return folds


def eval_acc(threshold, distances):
    pos = [1 if (d <= threshold and is_same) or (d > threshold and not is_same) else 0 for d, is_same in distances]
    return np.mean(pos)


def find_best_threshold(thresholds, distances):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, distances)
        if accuracy > best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def select_indexes(values, indexes):
    subset = []
    for i in indexes:
        subset.append(values[i])
    return subset


def get_lfw_accuracy(model, **kwargs):
    print('Inferencing embeddings')
    pairs = read_pairs(LFW_PAIRS_PATH)
    if kwargs['subset']:
        pairs = pairs[:600]
    distances = calculate_distances(model, pairs, **kwargs)

    print('Calculating 10-fold accuracy')
    accuracy = []
    best_thresholds = []
    folds = k_fold(n=len(pairs), n_folds=N_FOLDS)
    thresholds = np.arange(-1.0, 2.0, 0.01)

    for train, test in folds:
        best_threshold = find_best_threshold(thresholds, select_indexes(distances, train))
        accuracy.append(eval_acc(best_threshold, select_indexes(distances, test)))
        best_thresholds.append(best_threshold)

    return np.mean(accuracy), np.std(accuracy), np.mean(best_thresholds)


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Validate trained model on LFW dataset')
    parser.add_argument('--model-path', type=str, help='Converted model path')
    parser.add_argument('--flip-images',
                        action='store_true',
                        help='Calculate embedding also on horizontally flipped images')
    parser.add_argument('--subtract-mean',
                        action='store_true',
                        help='Subtract embedding mean before calculating the distance')
    parser.add_argument('--cosine-distance',
                        action='store_true',
                        help='Use cosine distance instead of euclidean distance')
    parser.add_argument('--l2-normalize',
                        action='store_true',
                        help='L2 normalize all embeddings along row axis')
    parser.add_argument('--subset',
                        action='store_true',
                        help='Enable quick validation on first 600 pairs')
    args = parser.parse_args()
    # Evaluate
    print('Loading the model')
    model = load_model(args.model_path)  # Trained face verification model
    acc, acc_std, best_threshold = get_lfw_accuracy(model, **vars(args))

    print('LFW acc={:.4f} std={:.4f} threshold={:.4f}'.format(acc, acc_std, best_threshold))
