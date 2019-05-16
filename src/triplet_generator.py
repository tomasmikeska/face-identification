import numpy as np
import random
from itertools import cycle
from constants import BATCH_SIZE, MAX_PERSON_IMGS_IN_BATCH, MAX_ANCHOR_POS_COUNT_IN_BATCH, SEMIHARD_MARGIN, TEST_BATCH_SIZE


def is_semihard(anchor, pos, neg):
    return np.linalg.norm(anchor - pos) + SEMIHARD_MARGIN > np.linalg.norm(anchor - neg)


def person_image_mapping(images, targets):
    person_images = { person_id: [] for person_id in set(targets) }

    for i in range(0, images.shape[0]):
        person_id = targets[i]
        person_images[person_id].append(images[i])

    return person_images


def get_test_triplet_generator(X, y):
    person_image = person_image_mapping(X, y)

    while True:
        batch_anchor = []
        batch_pos    = []
        batch_neg    = []

        for _ in range(0, TEST_BATCH_SIZE):
            anchor, neg = random.sample(list(person_image.keys()), 2)
            anchor_img, pos_img = random.sample(person_image[anchor], 2)
            neg_img = random.choice(person_image[neg])
            batch_anchor.append(anchor_img)
            batch_pos.append(pos_img)
            batch_neg.append(neg_img)

        yield [np.array(batch_anchor), np.array(batch_pos), np.array(batch_neg)]


def get_train_triplet_generator(X, y, calc_embedding):
    person_image = person_image_mapping(X, y)
    batch_anchor = []
    batch_pos    = []
    batch_neg    = []

    for person in cycle(person_image.keys()):
        pos_tuples = zip(person_image[person], person_image[person][1:])
        person_count = 0

        for anchor_img, pos_img in pos_tuples:
            other_people_imgs = [ imgs for other_person, imgs in person_image.items() if person != other_person ]
            other_people_imgs = [ img for imgs in other_people_imgs for img in imgs ]
            random.shuffle(other_people_imgs)
            anchor_emb = calc_embedding(anchor_img.reshape((1,) + anchor_img.shape))
            anchor_pos_count = 0

            for neg_img in other_people_imgs:
                pos_emb = calc_embedding(pos_img.reshape((1,) + pos_img.shape))
                neg_emb = calc_embedding(neg_img.reshape((1,) + neg_img.shape))

                if is_semihard(anchor_emb, pos_emb, neg_emb):
                    batch_anchor.append(anchor_img)
                    batch_pos.append(pos_img)
                    batch_neg.append(neg_img)
                    person_count += 1
                    anchor_pos_count += 1

                if len(batch_anchor) == BATCH_SIZE:
                    yield [np.array(batch_anchor), np.array(batch_pos), np.array(batch_neg)]
                    anchor_emb = calc_embedding(anchor_img.reshape((1,) + anchor_img.shape))
                    batch_anchor = []
                    batch_pos    = []
                    batch_neg    = []

                if anchor_pos_count >= MAX_ANCHOR_POS_COUNT_IN_BATCH:
                    break

                if person_count >= MAX_PERSON_IMGS_IN_BATCH:
                    break

            if person_count >= MAX_PERSON_IMGS_IN_BATCH:
                    break
