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
import os


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
    def __init__(self, filter_size=3):
        super(DnCNN, self).__init__()
        self.depth = 15     # DnCNN-S
        same_padding = int((filter_size - 1) / 2.0)
        self.first = nn.Conv2d(3, 64, filter_size, padding=same_padding)
        self.hidden_conv = nn.ModuleList([nn.Conv2d(64, 64, filter_size, padding=same_padding) for n in range(self.depth)])
        self.hidden_bn = nn.ModuleList([nn.BatchNorm2d(64) for n in range(15)])
        self.last = nn.Conv2d(64, 3, filter_size, padding=same_padding)

    def forward(self, x):
        x = F.relu(self.first(x))
        for num in range(self.depth):
            _conv = self.hidden_conv[num](x)
            _bn = self.hidden_bn[num](x)
            x = F.relu(_bn)

        x = self.last(x)
        return x

    def init_params(self):
        nn.init.kaiming_normal_(self.first.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        for num in range(self.depth):
            nn.init.kaiming_normal_(self.hidden_conv[num].weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.hidden_bn[num].weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.last.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def loss(self, x, y):
        output = self.forward(x)
        criterion = nn.MSELoss()
        loss = criterion(output, x.sub(y))

        return loss

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


class HRLNet(DenoisingModel):
    """

    """
    def __init__(self, filter_size=3, n_inference_subnet=4, m=5):
        super(HRLNet, self).__init__()
        same_padding = int((filter_size - 1) / 2.0)

        # Feature Extraction Net
        # Conv(f1, d1, c1)
        self.first_dim = 80
        self.first = nn.Conv2d(3, self.first_dim, filter_size, padding=same_padding)

        # Inference Net
        # Conv(fn, dn, cn), m
        self.n_inference_subnet = n_inference_subnet
        self.m = m
        subnets = nn.ModuleList([])
        inferences = nn.ModuleList([])

        # 初回のinference layerの入力次元数はfeature extraction layerの出力次元数
        conv_layers, inference = self.inference_layer(f=3, d=64, m=self.m, c=self.first_dim)
        subnets.extend(conv_layers)
        inferences.append(inference)

        for _ in range(self.n_inference_subnet - 1):
            conv_layers, inference = self.inference_layer(f=3, d=64, m=self.m, c=64)
            subnets.extend(conv_layers)
            inferences.append(inference)

        # layerをただリストにするだけだとよくないみたい
        # --> Sequentialを使う or ModuleListを使う
        self.inference_subnets = subnets
        self.each_inferenced_map = inferences

        # Fusion Net
        self.fusion = nn.Conv2d(self.n_inference_subnet, 3, 1)

    def init_params(self):
        nn.init.kaiming_normal_(self.first.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        for inference_net in self.inference_subnets:
            nn.init.kaiming_normal_(inference_net.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for inference_net in self.each_inferenced_map:
            nn.init.kaiming_normal_(inference_net.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def inference_layer(self, f, d, c, m):
        same_padding = int((f - 1) / 2.0)
        conv_layers = nn.ModuleList([])

        conv = nn.Conv2d(c, d, f, padding=same_padding)
        conv_layers.append(conv)
        for _ in range(m - 1):
            conv = nn.Conv2d(d, d, f, padding=same_padding)
            conv_layers.append(conv)

        inference = nn.Conv2d(d, 3, 3, padding=same_padding)

        return conv_layers, inference

    def forward(self, x):
        # Feature extraction
        x = F.relu(self.first(x))
        inferences = []
        list_inferences = []

        # inference
        for n_s in range(self.n_inference_subnet):
            for n_m in range(self.m):
                count = n_s * self.n_inference_subnet + n_m
                # print(count)
                x = F.relu(self.inference_subnets[count](x))
                # print(x.size())
            inference = F.relu(self.each_inferenced_map[n_s](x))
            inferences.append(inference[:, 0])
            list_inferences.append(inference)

        # Concat and Fusion
        concat_inferences = torch.stack(inferences, 1)
        output = self.fusion(concat_inferences)

        return output, inferences, list_inferences

    def loss(self, x, y):
        # alpha: importance of corresponding loss functions
        alpha = 1

        output, inferences, list_inferences = self.forward(x)
        criterion = nn.MSELoss()
        loss = criterion(output, x.sub(y))

        for inference in list_inferences:
            # print(inference.size())
            loss = loss.add(criterion(inference, x.sub(y)))

        return loss

    def denoise(self, noised_img):
        """
        get normalized and corrupted image, output denoised image of type Tensor
        It is necessary to convert Tensor to PIL with ToPILImage for enable to show the denoised image.
        :param noised_img:
        :return:
        """
        output, _, __ = self.forward(noised_img)
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
                noised_set.append(self.data_transform(noised))
                org_set.append(self.data_transform(org))

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
    """
    util内のgenerate_testsetみたいのを使ってることを想定
    testset
        - /org/  : 原画像
        - /p_0.1/ : 0.1のRVINが重畳

        みたいな感じで保存されているはず
    """
    def __init__(self, dir, data_transform, noise_p, shape=None):
        self.dir = dir
        self.org_dir = os.path.join(dir, 'org')
        self.noise_p = noise_p
        self.data_transform = data_transform

        noised_set, org_set = self.load_denoising_datasets()

        self.org = org_set
        self.noised = noised_set

    def load_denoising_datasets(self):
        num_imgs = len(os.listdir(self.org_dir))
        org_set = []
        noised_set = []
        for num in range(num_imgs):
            org_path = os.path.join(self.org_dir, 'img_{}.png'.format(num))
            org_img = self.data_transform(Image.open(org_path))

            for p in self.noise_p:
                noise_path = os.path.join(self.dir, 'p_{}'.format(p), 'img_{}.png'.format(num))
                noised_img = self.data_transform(Image.open(noise_path))

                org_set.append(org_img)
                noised_set.append(noised_img)

        return noised_set, org_set

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
