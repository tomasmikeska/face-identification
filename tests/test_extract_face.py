import pytest
import random
import numpy as np
from collections import namedtuple
from constants import TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT
import extract_face


@pytest.fixture()
def landmarks(scope='module'):
    class Landmarks(object):
        pass
    l = Landmarks()
    l.LEFT_EYE_X           = extract_face.REF_LEFT_EYE[0]
    l.RIGHT_EYE_X          = extract_face.REF_RIGHT_EYE[0]
    l.NOSE_X               = extract_face.REF_NOSE[0]
    l.LEFT_MOUTH_CORNER_X  = extract_face.REF_LEFT_MOUTH_CORNER[0]
    l.RIGHT_MOUTH_CORNER_X = extract_face.REF_RIGHT_MOUTH_CORNER[0]
    l.LEFT_EYE_Y           = extract_face.REF_LEFT_EYE[1]
    l.RIGHT_EYE_Y          = extract_face.REF_RIGHT_EYE[1]
    l.NOSE_Y               = extract_face.REF_NOSE[1]
    l.LEFT_MOUTH_CORNER_Y  = extract_face.REF_LEFT_MOUTH_CORNER[1]
    l.RIGHT_MOUTH_CORNER_Y = extract_face.REF_RIGHT_MOUTH_CORNER[1]
    return l


def test_transform_face(landmarks):
    oversized_img    = np.random.normal(0, 1, (500, 300, 3))
    target_size_img  = np.random.normal(0, 1, (TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3))

    oversized_transformed   = extract_face.transform_face(oversized_img, landmarks)
    target_size_transformed = extract_face.transform_face(target_size_img, landmarks)

    assert oversized_transformed.shape == (TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3)
    assert np.all(target_size_transformed == target_size_img)


def test_horizontal_flip():
    img_np = np.array([[[1, 1, 1], [1, 2, 1], [1, 3, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    target = np.array([[[1, 3, 1], [1, 2, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    assert np.all(extract_face.flip_image_horizontally(img_np) == target)
