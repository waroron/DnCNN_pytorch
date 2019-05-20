from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from util import get_crop_datasets, noised_RVIN, cv2pil, pil2cv, MyModel, load_orgimgs
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


class DenoisingModel(MyModel):
    def __init__(self):
        super(DenoisingModel, self).__init__()

    def denoise(self, img):
        """
        get an PIL image and output its denoised PIL image
        Convert PIL to Tensor of shape [C, H, W](or [1, C, H, W]).
        and reconvert the Tensor after forward propagation.
        :param img:
        :return:
        """
        trans = transforms.ToPILImage()
        pre_operate = transforms.Compose([transforms.ToTensor()])
        tensor_img = pre_operate(img)

        # ここわんちゃん[N, C, H, W]の形式になっていない，って怒られるかも
        output = self(tensor_img)
        output_pil = trans(output)

        return output_pil


class DnCNN(DenoisingModel):
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
        get normalized and corrupted image, output denoised image of type Tensor
        It is necessary to convert Tensor to PIL with ToPILImage for enable to show the denoised image.
        :param noised_img:
        :return:
        """
        output = self.forward(noised_img)
        tmp = F.relu(noised_img - output)
        return tmp


class DenoisingDatasets(Dataset):
    def __init__(self, dir, noise_p, data_transform, shape=(180, 180), num_crop=2):
        self.num_crop = num_crop
        self.img_height = shape[0]
        self.img_width = shape[1]
        self.data_transform = data_transform
        self.noise_p = noise_p
        imgs = get_crop_datasets(path=dir, width=self.img_width, height=self.img_height, times=self.num_crop)

        noised_set = []
        org_set = []
        for img in imgs:
            noised_imgs, org_imgs = noised_RVIN(img, noised_p=self.noise_p)

            for noised, org in zip(noised_imgs, org_imgs):
                noised_set.extend(self.data_transform(noised))
                org_set.extend(self.data_transform(org))

        # convert cv2 image to Pillow img
        # org_imgs = []
        # for img in imgs:
        #     org_imgs.append(self.data_transform(img))

        self.org = org_set
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


class ImageDatasets(Dataset):
    def __init__(self, dir, data_transform, shape=None):
        imgs = load_orgimgs(dir)
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