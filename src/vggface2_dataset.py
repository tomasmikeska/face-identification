import cv2
import os
import math
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image
from tqdm import tqdm
from constants import VGG_TRAIN_PATH
from utils import file_listing, dir_listing, last_component, relative_path, mkdir


TARGET_PATH = relative_path('../data/VGGFace2_funneled/train/')
MIN_IMG_SIZE = 70
TARGET_IMG_SIZE = 100
CROP_MARGIN = 10


# def extract_face(img_path, target_path, detector):
#     img_np = cv2.imread(img_path)
#     faces = detector.detect_faces(img_np)
#
#     if len(faces) > 0:
#         x, y, width, height = faces[0]['box']
#
#         if width > MIN_IMG_SIZE and height > MIN_IMG_SIZE:
#             left_eye = faces[0]['keypoints'].get('left_eye')
#             right_eye = faces[0]['keypoints'].get('right_eye')
#
#             # Filter only front face images (both eyes visible, no glasses)
#             if left_eye == None or right_eye == None:
#                 return
#
#             # 2d facial alignment using eyes coords - atan((y1 - y2)/(x1 - x2))
#             degs = math.degrees(math.atan((left_eye[1] - right_eye[1]) / (left_eye[0] - right_eye[0])))
#             hor_delta = int(max(0, height - width) / 2) + CROP_MARGIN
#             ver_delta = int(max(0, width - height) / 2) + CROP_MARGIN
#
#             img_pil = Image.fromarray(img_np)
#             img_pil = img_pil.rotate(degs)
#             img_pil = img_pil.crop((x-hor_delta, y-ver_delta, x+width+hor_delta, y+height+ver_delta))
#             img_pil = img_pil.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE), Image.ANTIALIAS).convert('LA')
#             img_pil.convert('RGB').save(target_path, 'JPEG')

def extract_face(img_path, target_path, detector):
    img_np = cv2.imread(img_path)
    faces = detector.detectMultiScale(
        img_np,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_IMG_SIZE, MIN_IMG_SIZE),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0:
        x, y, width, height = faces[0]

        hor_delta = int(max(0, height - width) / 2) + CROP_MARGIN
        ver_delta = int(max(0, width - height) / 2) + CROP_MARGIN

        img_pil = Image.fromarray(img_np)
        img_pil = img_pil.crop((x-hor_delta, y-ver_delta, x+width+hor_delta, y+height+ver_delta))
        img_pil = img_pil.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE), Image.ANTIALIAS).convert('LA')
        return img_pil.convert('RGB').save(target_path, 'JPEG')


def get_files(source_root, target_root):
    files = []
    for dir_name in dir_listing(source_root):
        for file_path in file_listing(dir_name, 'jpg'):
            target_dir = target_root + '/' + last_component(dir_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            target_path = target_dir + '/' + last_component(file_path)
            files.append((file_path, target_path))
    return files


if __name__ == '__main__':
    # detector = MTCNN()
    detector = cv2.CascadeClassifier(relative_path('../model/haarcascade/haarcascade_frontalface_default.xml'))
    files = get_files(VGG_TRAIN_PATH, TARGET_PATH)
    files = files[624138:]

    for file_path, target_path in tqdm(files):
        try:
            extract_face(file_path, target_path, detector)
        except Exception:
            pass # skip image
