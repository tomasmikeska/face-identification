import numpy as np
import random
from itertools import cycle
from constants import BATCH_SIZE, MAX_PERSON_IMGS_IN_BATCH, MAX_ANCHOR_POS_COUNT_IN_BATCH, SEMIHARD_MARGIN


def is_semihard(anchor, pos, neg):
    return np.linalg.norm(anchor - pos) + SEMIHARD_MARGIN > np.linalg.norm(anchor - neg)


def person_image_mapping(images, targets):
    person_images = { person_id: [] for person_id in set(targets) }

    for i in range(0, images.shape[0]):
        person_id = targets[i]
        person_images[person_id].append(images[i])

    return person_images


def get_offline_triplet_generator(X, y, triplet_count=BATCH_SIZE):
    person_image = person_image_mapping(X, y)

    while True:
        batch_anchor = []
        batch_pos    = []
        batch_neg    = []

        for _ in range(0, triplet_count):
            anchor, neg = random.sample(list(person_image.keys()), 2)
            anchor_img, pos_img = random.sample(person_image[anchor], 2)
            neg_img = random.choice(person_image[neg])
            batch_anchor.append(anchor_img)
            batch_pos.append(pos_img)
            batch_neg.append(neg_img)

        yield [np.array(batch_anchor), np.array(batch_pos), np.array(batch_neg)]


def get_online_triplet_generator(X, y, calc_distances, triplet_count=BATCH_SIZE):
    offline_gen = get_offline_triplet_generator(X, y, triplet_count=int(triplet_count*1.25))
    batch = np.empty((3, 0,) + X.shape[1:])

    while True:
        sub_batch = next(offline_gen)
        dists = calc_distances(sub_batch)
        semihard_indices = np.where(dists[:, 0, 0] + SEMIHARD_MARGIN > dists[:, 1, 0])[0]
        batch = np.append(batch, np.array(sub_batch)[:, semihard_indices], axis=1)

        if batch.shape[1] >= triplet_count:
            yield [batch[0], batch[1], batch[2]]
            batch = np.empty((3, 0,) + X.shape[1:])


def get_combined_triplet_generator(X, y, calc_distances):
    offline_gen = get_offline_triplet_generator(X, y, triplet_count=(BATCH_SIZE // 2))
    online_gen  = get_online_triplet_generator(X, y, calc_distances, triplet_count=(BATCH_SIZE // 2))

    while True:
        offline_batch = next(offline_gen)
        online_batch  = next(online_gen)
        batch = list(map(lambda t: np.concatenate(t, axis=0), zip(offline_batch, online_batch)))
        yield batch
