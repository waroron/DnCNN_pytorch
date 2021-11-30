import cv2
import os
import numpy as np
import time
from PIL import Image, ImageChops, ImageStat
import math
import torch
import torch.nn as nn
import pandas as pd
import zipfile
import io
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def save(self, dir, name):
        if not os.path.isdir(dir):
            print('make dir {}'.format(dir))
            os.mkdir(dir)

        path = os.path.join(dir, name)
        torch.save(self.state_dict(), path)
        print('save model {}'.format(path))


def load_orgimgs(path='./BSDS200/', shape=None):
    if os.path.isdir(path) is None:
        exit('{} dir is not found.'.format(path))

    imgs_name = os.listdir(path)
    imgs = []

    for img_name in imgs_name:
        img_path = os.path.join(path, img_name)
        img_bin = Image.open(img_path)
        # img_bin.show()
        if shape:
            img_bin = img_bin.resize(shape)

        imgs.append(img_bin)

    return imgs


def load_orgimgs_from_zip(file_path):
    imgs = []
    with zipfile.ZipFile(file_path, 'r') as zip_file:
        infos = zip_file.infolist()

        for info in infos:
            file_bin = zip_file.read(info.filename)
            img = Image.open(io.BytesIO(file_bin))
            # img.show()
            imgs.append(img)
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


def get_mse(img1, img2):
    diff_img = ImageChops.difference(img1, img2)
    stat = ImageStat.Stat(diff_img)
    mse = sum(stat.sum2) / len(stat.count) / stat.count[0]
    return mse


def get_psnr(img1, img2):
    mse = get_mse(img1, img2)
    return 10 * math.log10(255 ** 2 / mse)


def get_crop_datasets(path='./BSDS200/', width=180, height=180, times=2):
    img_set = load_orgimgs(path=path)
    crop = []
    for img in img_set:
        crop_widths = np.random.randint(0, img.width - width, times)
        crop_heights = np.random.randint(0, img.height - height, times)

        for crop_width, crop_height in zip(crop_widths, crop_heights):
            np_img = np.array(img)
            crop_img = np_img[crop_height: crop_height + height, crop_width: crop_width + width]
            crop.append(Image.fromarray(crop_img))

    return crop


def resize_imgset(imgs, width, height):
    resized_imgs = []
    for img in imgs:
        resized = img.resize((width, height))
        resized_imgs.append(resized)

    return resized_imgs


def generate_resized_dataset(dataset_path, width, height):
    pass


def noised_RVIN(img, noised_p):
    np_img = np.array(img)
    noised_imgs = []
    org_imgs = []

    for p in noised_p:
        noised_img = np_img.copy()
        noised_p = np.random.uniform(0, 1, noised_img.shape)
        noised_positions = np.where(noised_p < p)
        noise_values = np.random.randint(0, 255, len(noised_positions[0]))

        noised_img[noised_positions] = noise_values
        noised_img = Image.fromarray(noised_img)

        noised_imgs.append(noised_img)
        org_imgs.append(img)

    return noised_imgs, org_imgs


def append_csv_from_dict(dir, csv_path, dict_data):
    if not os.path.isdir(dir):
        print('make {} dir'.format(dir))
        os.mkdir(dir)
    csv_path = os.path.join(dir, csv_path)
    columns = ['num_epoch', 'training_loss', 'test_loss', 'test_psnr']
    append_col = []

    for column in columns:
        append_col.append(dict_data[column])

    df = pd.DataFrame([append_col], columns=columns)
    try:
        if os.path.isfile(csv_path):
            existed_csv = pd.read_csv(csv_path, encoding='utf_8_sig').iloc[:, 1:]
            existed_csv = pd.concat([existed_csv, df])
            existed_csv.to_csv(csv_path, encoding='utf_8_sig')
            return True
        df.to_csv(csv_path, encoding='utf_8_sig')
    except:
        return False


def save_torch_model(dir, name, model):
    if not os.path.isdir(dir):
        print('make dir {}'.format(dir))
        os.mkdir(dir)

    torch.save(model.state_dict(), os.path.join(dir, name))


def generate_denoising_testset(org_dir, save_dir, noise_p, shape):
    orgimgs_set = load_orgimgs(path=org_dir, shape=shape)
    save_org_dir = os.path.join(save_dir, 'org')

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        print('make dir {}'.format(save_dir))

    if not os.path.isdir(save_org_dir):
        os.mkdir(save_org_dir)
        print('make dir {}'.format(save_org_dir))

    for p in noise_p:
        score_list = []
        columns = ['MSE', 'PSNR']
        index = []
        save_noise_dir = os.path.join(save_dir, 'p_{}'.format(p))

        if not os.path.isdir(save_noise_dir):
            os.mkdir(save_noise_dir)
            print('make dir {}'.format(save_noise_dir))

        for num, org in enumerate(orgimgs_set):
                org_path = os.path.join(save_org_dir, 'img_{}.png'.format(num))
                noise_path = os.path.join(save_noise_dir, 'img_{}.png'.format(num))
                index.append('img_{}'.format(num))

                noised_img, _ = noised_RVIN(org, noised_p=[p])

                mse = get_mse(org, noised_img[0])
                psnr = get_psnr(org, noised_img[0])

                score_list.append([mse, psnr])

                if not os.path.isfile(org_path):
                    org.save(org_path)
                noised_img[0].save(noise_path)

                print('save {}'.format(org_path))
        csv_path = os.path.join(save_noise_dir, 'evaluation_score.csv')
        df = pd.DataFrame(score_list, index=index, columns=columns)
        df.to_csv(csv_path)
        print('{} saved'.format(csv_path))


def generate_SISR_testset(org_dir, save_dir, upscale_list, shape):
    orgimgs_set = load_orgimgs(path=org_dir, shape=shape)
    save_org_dir = os.path.join(save_dir, 'org')

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        print('make dir {}'.format(save_dir))

    if not os.path.isdir(save_org_dir):
        os.mkdir(save_org_dir)
        print('make dir {}'.format(save_org_dir))

    for scale in upscale_list:
        score_list = []
        columns = ['MSE', 'PSNR']
        index = []
        save_noise_dir = os.path.join(save_dir, 'scale_{}'.format(scale))

        if not os.path.isdir(save_noise_dir):
            os.mkdir(save_noise_dir)
            print('make dir {}'.format(save_noise_dir))

        for num, org in enumerate(orgimgs_set):
                org_path = os.path.join(save_org_dir, 'img_{}.png'.format(num))
                sr_path = os.path.join(save_noise_dir, 'img_{}.png'.format(num))
                index.append('img_{}'.format(num))

                tmp = org.resize((int(shape[0] / scale), int(shape[1] / scale)))
                sr_img = tmp.resize(shape, Image.BICUBIC)

                mse = get_mse(org, sr_img)
                psnr = get_psnr(org, sr_img)

                score_list.append([mse, psnr])

                if not os.path.isfile(org_path):
                    org.save(org_path)
                sr_img.save(sr_path)

                print('save {}'.format(org_path))
        csv_path = os.path.join(save_noise_dir, 'evaluation_score.csv')
        df = pd.DataFrame(score_list, index=index, columns=columns)
        df.to_csv(csv_path)
        print('{} saved'.format(csv_path))


if __name__ == '__main__':
    # start = time.time()
    # load_orgimgs()
    # end = time.time()
    # print('not zip time: {}'.format(end - start))
    #
    # start = time.time()
    # load_orgimgs_from_zip('BSDS200.zip')
    # end = time.time()
    # print('zip time: {}'.format(end - start))
    # open_zip('Urban100_test.zip')
    # generate_SISR_testset('Urban100', 'Urban100_test', [2, 4, 8], (180, 180))
    generate_denoising_testset('Set14_2', 'Set14_test', [0.05, 0.1, 0.2, 0.3, 0.5], [180, 180])
    # start = time.time()
    # imgs = load_orgimgs()
    # for img in imgs:
    #     noised = noised_RVIN(img, 0.2)
    #
    # noised.show()
    # end = time.time()
    #
    # print('{} times noising time: {}'.format(len(imgs), end - start))
    # cv2.imshow('org', imgs[0])
    # cv2.imshow('noised', noised)

    # cv2.waitKey(0)
