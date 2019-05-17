from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from util import get_crop_datasets, noised_RVIN, cv2pil, pil2cv
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


class DnCNN(nn.Module):
    """
    DnCNN
    (1) Conv + ReLU
        for the first layer, 64 filters of size 3 * 3 * c and ReLU

    (2) Conv + BN + ReLU
        for hidden layers, 64 filters of size 3 * 3 * 64 and batch normalization is added between Conv and ReLU

    (3) Conv
        for the last layer, c filters of size 3 * 3 * 64

    Experimental settings in the original paper (DnCNN-S)
        400 images of size 180 * 180 is used for training, and corrupt the images
        with gaussian noise of 3 kinds of noise level.
        Its patch size is set as 40 * 40.
        DnCNN-S model depth is set as 17.

    """
    def __init__(self):
        super(DnCNN, self).__init__()
        self.first = nn.Conv2d(3, 64, 3, padding=1)
        self.hidden_conv = nn.ModuleList([nn.Conv2d(64, 64, 3, padding=1) for n in range(15)])
        self.hidden_bn = nn.ModuleList([nn.BatchNorm2d(64) for n in range(15)])
        self.last = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.first(x))
        for num in range(15):
            _conv = self.hidden_conv[num](x)
            _bn = self.hidden_bn[num](x)
            x = F.relu(_bn)

        x = self.last(x)
        return x

    def denoise(self, noised_img):
        """
        cv2形式の画像をとり，pillow形式に変換し，学習時と同様のtransformを適用し，
        順伝搬させる．
        その後，画像形式に沿うようにreshapeしcv2形式に変換する．
        :param noised_img:
        :return:
        """
        output = self.forward(noised_img)
        tmp = F.relu(noised_img - output) * 255.0
        trans = tmp
        return trans


class DenoisingDatasets(Dataset):
    def __init__(self, dir, data_transform=noised_RVIN):
        imgs = get_crop_datasets(path=dir)

        self.data_transform = data_transform

        noised_set = []
        for img in imgs:
            noised = noised_RVIN(img)
            noised_set.append(self.data_transform(noised))

        # convert cv2 image to Pillow img
        org_imgs = []
        for img in imgs:
            org_imgs.append(self.data_transform(img))

        self.org = org_imgs
        self.noised = noised_set

    def __len__(self):
        return len(self.org)

    def __getitem__(self, i):
        # if self.data_transform:
        #     noised_img = self.data_transform(self.df[i])
        # else:
        #     noised_img = self.df[i]
        org_img = self.org[i]
        noised_img = self.noised[i]

        return noised_img, org_img
