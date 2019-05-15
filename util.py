import cv2
import os
import numpy as np
import time
from PIL import Image


def load_orgimgs(path='./BSDS200/'):
    if os.path.isdir(path) is None:
        exit('{} dir is not found.'.format(path))

    imgs_name = os.listdir(path)
    imgs = []
    for img_name in imgs_name:
        img_path = os.path.join(path, img_name)
        img_bin = cv2.imread(img_path)

        imgs.append(img_bin)

    return imgs


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    return new_image


def get_crop_datasets(path='./BSDS200/', width=180, height=180, times=2):
    img_set = load_orgimgs(path=path)
    crop = []
    for img in img_set:
        crop_widths = np.random.randint(0, img.shape[1] - width, times)
        crop_heights = np.random.randint(0, img.shape[0] - height, times)

        for crop_width, crop_height in zip(crop_widths, crop_heights):
            crop_img = img[crop_height: crop_height + height, crop_width: crop_width + width]
            crop.append(crop_img)

    return crop


def resize_imgset(imgs, width, height):
    resized_imgs = []
    for img in imgs:
        resized = cv2.resize(img, (width, height))
        resized_imgs.append(resized)

    return resized_imgs


def generate_resized_dataset(dataset_path, width, height):
    pass


def noised_RVIN(img, p=0.1):
    noised_img = img.copy()
    noised_p = np.random.uniform(0, 1, (img.shape[0], img.shape[1], img.shape[2]))
    noised_positions = np.where(noised_p < p)
    noise_values = np.random.randint(0, 255, len(noised_positions[0]))

    noised_img[noised_positions] = noise_values

    return noised_img


if __name__ == '__main__':
    start = time.time()
    imgs = load_orgimgs()
    for img in imgs:
        noised = noised_RVIN(img, 0.2)

    end = time.time()

    print('{} times noising time: {}'.format(len(imgs), end - start))
    # cv2.imshow('org', imgs[0])
    # cv2.imshow('noised', noised)

    # cv2.waitKey(0)
