from dncnn import DnCNN, DenoisingDatasets, DenoisingTestsets, HRLNet, SISRDatasets, SISRTestsets
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from util import append_csv_from_dict, save_torch_model, load_orgimgs
import os
import time
import numpy as np
from PIL import ImageChops, ImageStat
import math
from torchsummary import summary


def evaluate(model, test_set, device):
    sum_loss = 0
    for num, (image, org) in enumerate(test_set):
        image = image.to(device)
        org = org.to(device)
        loss = model.loss(image, org)
        # print(num)
        sum_loss += loss.data.cpu().numpy()

    return sum_loss / len(test_set)


def psnr(img1, img2):
    diff_img = ImageChops.difference(img1, img2)
    stat = ImageStat.Stat(diff_img)
    mse = sum(stat.sum2) / len(stat.count) / stat.count[0]
    return 10 * math.log10(255 ** 2 / mse)


def denoise_test(model, test_set, device):
    sum_loss = 0
    trans = transforms.ToPILImage()
    denoised_fig = []
    for num, (image, orgs) in enumerate(test_set):
        image = image.to(device)
        denoised = model.denoise(image).data.cpu()
        for img, org in zip(denoised, orgs):
            org = trans(org)
            img = trans(img)
            score = psnr(org, img)
            # print(score)
            sum_loss += score
            denoised_fig.append(img)
    print('Test PSNR: {:.6f}'.format(sum_loss / len(denoised_fig)))
    return denoised_fig, sum_loss / len(denoised_fig)


def train_denoising_model(model, experimental_name, base_dir, epoch, dev, training_p, test_p,
          batch_size, filter_size, save_max, load_model=None):
    """
    Training method for denoising neural net model
    :param model:
    :param experimental_name:
    :param base_dir:
    :param epoch:
    :param dev:
    :param training_p:
    :param test_p:
    :param batch_size:
    :param filter_size:
    :param save_max:
    :param load_model:
    :return:
    """
    MODEL_PATH = 'model.pth'
    DATASET = 'BSDS200/'
    TESTSET = 'Urban100_test/'
    csv_name = 'epoch_data.csv'
    dataset_dir = os.path.join(base_dir, DATASET)
    testset_dir = os.path.join(base_dir, TESTSET)
    transform = transforms.Compose(
        [transforms.ToTensor()])

    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    print(device)
    if load_model:
        # model = DnCNN(filter_size=filter_size)
        model.load_state_dict(torch.load(load_model))
        print('load model {}.'.format(load_model))
    else:
        print('Not found load model.')
        # model = HRLNet(filter_size=filter_size)
        model.init_params()
        # model = DnCNN(filter_size=filter_size)

    model = model.to(device)
    summary(model, (3, 180, 180))
    optimizer = Adam(model.parameters())
    experimental_dir = os.path.join(base_dir, experimental_name)
    
    if not os.path.isdir(experimental_dir):
        print('make dir {}'.format(experimental_dir))
        os.mkdir(experimental_dir)

    dataset = DenoisingDatasets(dir=dataset_dir, data_transform=transform, noise_p=training_p)
    test_set = DenoisingTestsets(dir=testset_dir, data_transform=transform, noise_p=test_p)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    testloader = DataLoader(
        dataset=test_set,
        batch_size=5,
        shuffle=False
    )
    print('All datasets have been loaded.')

    for num in range(epoch + 1):
        total_loss = 0
        calc_time = 0
        test_loss = 0
        start = time.time()
        for batch_idx, (image, label) in enumerate(dataloader):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            loss = model.loss(image, label)
            total_loss += float(loss.data.cpu().numpy())
            loss.backward()
            optimizer.step()

            total_loss /= len(dataloader)
            test_loss = evaluate(model, testloader, device)

        end = time.time()
        calc_time += (end - start)
        data_dict = {'num_epoch': num,
                     'training_loss': total_loss,
                     'test_loss': test_loss}
        print('Train Epoch: {} \tLoss: {:.6f} \tTest Loss: {:.6f} \tCalculation Time: {:.4f}sec'.format(
            num, total_loss, test_loss, calc_time))

        # denoising evaluation via testset
        if num % 10 == 0:
            denoised_imgs, test_psnr = denoise_test(model, testloader, device)
            psnr_dict = {'test_psnr': test_psnr}
            data_dict.update(psnr_dict)
            append_csv_from_dict(experimental_dir, csv_name, data_dict)
            epoch_dir = os.path.join(experimental_dir, str(num))
            save_torch_model(epoch_dir, MODEL_PATH, model)

            for num, denoised in enumerate(denoised_imgs):
                if num >= save_max:
                    break
                denoised_name = os.path.join(epoch_dir, 'denoised_{}.png'.format(num))
                denoised.save(denoised_name)
    epoch_dir = os.path.join(experimental_dir, str(epoch))
    model_path = os.path.join(epoch_dir, MODEL_PATH)
    return model_path


def train_SISR_model(model, experimental_name, base_dir, epoch, dev, training_scale, test_scale,
                          batch_size, filter_size, save_max, load_model=None):
    """
    Training method for denoising neural net model
    :param model:
    :param experimental_name:
    :param base_dir:
    :param epoch:
    :param dev:
    :param training_p:
    :param test_p:
    :param batch_size:
    :param filter_size:
    :param save_max:
    :param load_model:
    :return:
    """
    MODEL_PATH = 'model.pth'
    DATASET = 'BSDS200/'
    TESTSET = 'Urban100_test/'
    csv_name = 'epoch_data.csv'
    dataset_dir = os.path.join(base_dir, DATASET)
    testset_dir = os.path.join(base_dir, TESTSET)
    transform = transforms.Compose(
        [transforms.ToTensor()])

    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    print(device)
    if load_model:
        # model = DnCNN(filter_size=filter_size)
        model.load_state_dict(torch.load(load_model))
        print('load model {}.'.format(load_model))
    else:
        print('Not found load model.')
        # model = HRLNet(filter_size=filter_size)
        model.init_params()
        # model = DnCNN(filter_size=filter_size)

    model = model.to(device)
    summary(model, (3, 180, 180))
    optimizer = Adam(model.parameters())
    experimental_dir = os.path.join(base_dir, experimental_name)

    if not os.path.isdir(experimental_dir):
        print('make dir {}'.format(experimental_dir))
        os.mkdir(experimental_dir)

    dataset = SISRDatasets(dir=dataset_dir, data_transform=transform, upscale_list=training_scale)
    test_set = SISRTestsets(dir=testset_dir, data_transform=transform, upscale_list=test_scale)
    extra_test_set = load_orgimgs('SISR_TEST')
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    testloader = DataLoader(
        dataset=test_set,
        batch_size=5,
        shuffle=False
    )
    print('All datasets have been loaded.')

    for num in range(epoch + 1):
        total_loss = 0
        calc_time = 0
        test_loss = 0
        start = time.time()
        for batch_idx, (image, label) in enumerate(dataloader):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            loss = model.loss(image, label)
            total_loss += float(loss.data.cpu().numpy())
            loss.backward()
            optimizer.step()

            total_loss /= len(dataloader)
            test_loss = evaluate(model, testloader, device)

        end = time.time()
        calc_time += (end - start)
        data_dict = {'num_epoch': num,
                     'training_loss': total_loss,
                     'test_loss': test_loss}
        print('Train Epoch: {} \tLoss: {:.6f} \tTest Loss: {:.6f} \tCalculation Time: {:.4f}sec'.format(
            num, total_loss, test_loss, calc_time))

        # denoising evaluation via testset
        if num % 10 == 0:
            denoised_imgs, test_psnr = denoise_test(model, testloader, device)
            psnr_dict = {'test_psnr': test_psnr}
            data_dict.update(psnr_dict)
            append_csv_from_dict(experimental_dir, csv_name, data_dict)
            epoch_dir = os.path.join(experimental_dir, str(num))
            save_torch_model(epoch_dir, MODEL_PATH, model)

            for num, denoised in enumerate(denoised_imgs):
                if num >= save_max:
                    break
                denoised_name = os.path.join(epoch_dir, 'denoised_{}.png'.format(num))
                denoised.save(denoised_name)

            # to_tensor = transforms.ToTensor()
            # trans = transforms.ToPILImage()
            # for num, test_img in enumerate(extra_test_set):
            #     tensor_img = to_tensor(test_img)
            #     test_img = torch.reshape(tensor_img, (1, 3, test_img.height, test_img.width))
            #     image = test_img.to(device)
            #     enlarged = model.denoise(image).data.cpu()
            #     enlarged = trans(enlarged)
            #     enlarged_path = os.path.join(epoch_dir, 'enlarged_extra_{}.png'.format(num))
            #     enlarged.save(enlarged_path)

    epoch_dir = os.path.join(experimental_dir, str(epoch))
    model_path = os.path.join(epoch_dir, MODEL_PATH)
    return model_path


def train_mix_dataset(experimental_name, base_dir, epoch, times, dev, training_p, test_p, batch_size, filter_size, save_max):
    last_model_path = ''
    for num in range(times):
        if num == 0:
            last_model_path = train_denoising_model(experimental_name + 'part{}'.format(num + 1), base_dir, epoch, dev, training_p, test_p, batch_size, filter_size, save_max, load_model=None)
        else:
            last_model_path = train_denoising_model(experimental_name + 'part{}'.format(num + 1), base_dir, epoch, dev, training_p, test_p, batch_size, filter_size, save_max, load_model=last_model_path)


if __name__ == '__main__':
    model = HRLNet()
    # train_denoising_model(model, 'DnCNN-S', './', 150, 'cuda', [0.1, 0.2], [0.1], 10, 5, 30)
    train_SISR_model(model, 'HRLNet_forSISR', './', 150, 'cuda', [2, 4], [2], 10, 5, 30)
