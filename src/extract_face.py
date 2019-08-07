import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import SimilarityTransform
from constants import TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT


#  -- Face extraction --


# Target transformation facial landmarks
# that are used for similarity transformation
#
# Reference point      = (   x   ,    y   )
REF_LEFT_EYE           = (30.2946, 51.6963)
REF_RIGHT_EYE          = (65.5318, 51.5014)
REF_NOSE               = (48.0252, 71.7366)
REF_LEFT_MOUTH_CORNER  = (33.5493, 92.3655)
REF_RIGHT_MOUTH_CORNER = (62.7299, 92.2041)


def alignment(src_img, landmarks):
    ref_pts = [REF_LEFT_EYE, REF_RIGHT_EYE, REF_NOSE, REF_LEFT_MOUTH_CORNER, REF_RIGHT_MOUTH_CORNER]
    crop_size = (TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT)

    s = np.array(ref_pts).astype(np.float32)
    r = np.array(landmarks).astype(np.float32)

    tfm = SimilarityTransform()
    tfm.estimate(r, s)
    M = tfm.params[0:2, :]

    face_img = cv2.warpAffine(src_img, M, crop_size)

    return face_img


def transform_face(img_np, landmarks):
    left_eye           = [landmarks.LEFT_EYE_X,           landmarks.LEFT_EYE_Y]
    right_eye          = [landmarks.RIGHT_EYE_X,          landmarks.RIGHT_EYE_Y]
    nose               = [landmarks.NOSE_X,               landmarks.NOSE_Y]
    left_mouth_corner  = [landmarks.LEFT_MOUTH_CORNER_X,  landmarks.LEFT_MOUTH_CORNER_Y]
    right_mouth_corner = [landmarks.RIGHT_MOUTH_CORNER_X, landmarks.RIGHT_MOUTH_CORNER_Y]
    src_pts            = [left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner]
    return alignment(img_np, src_pts)


def extract_face(path, landmarks):
    '''Extract face from image path using similarity transformation with provided landmarks

    Args:
        path (string): Path to image to read
        landmarks (array): Array of face landmark points, i.e. [x, y] array, in order:
                           [left eye, right eye, nose, left mouth corner, right mouth corner]

    Returns:
        Numpy array containing extracted face image
    '''
    img_np = plt.imread(path)
    if img_np.ndim != 3:
        return None
    return transform_face(img_np, landmarks)


# -- Data augmentation --


def flip_image_horizontally(img_np):
    return np.flip(img_np, 1)


def augment_face(img_np):
    if random.random() > 0.5: # Flip images vertically
        img_np = flip_image_horizontally(img_np)
    return img_np
