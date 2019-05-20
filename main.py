from dncnn import DnCNN, DenoisingDatasets
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from util import append_csv_from_dict, save_torch_model
import os
import time
import numpy as np
from PIL import ImageChops, ImageStat
import math


def evaluate(model, test_set, device):
    criterion = nn.MSELoss()
    sum_loss = 0
    for num, (image, org) in enumerate(test_set):
        image = image.to(device)
        org = org.to(device)
        output = model(image)
        loss = criterion(output, image.sub(org))
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


def train(experimental_name, base_dir, epoch, dev, load_model=None):
    MODEL_PATH = 'model.pth'
    DATASET = 'BSDS200/'
    TESTSET = 'Set5/'
    csv_name = 'epoch_data.csv'
    dataset_dir = os.path.join(base_dir, DATASET)
    testset_dir = os.path.join(base_dir, TESTSET)
    transform = transforms.Compose(
        [transforms.ToTensor()])

    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    print(device)
    if load_model:
        model = DnCNN()
        model.load_state_dict(torch.load(load_model))
        model = model.to(device)
        print('load model {}.'.format(load_model))
    else:
        print('Not found load model.')
        model = DnCNN().to(device)

    optimizer = Adam(model.parameters())
    criterion = nn.MSELoss()
    experimental_dir = os.path.join(base_dir, experimental_name)
    
    if not os.path.isdir(experimental_dir):
        print('make dir {}'.format(experimental_dir))
        os.mkdir(experimental_dir)

    dataset = DenoisingDatasets(dir=dataset_dir, data_transform=transform, noise_p=[0.1])
    test_set = DenoisingDatasets(dir=testset_dir, data_transform=transform, noise_p=[0.1])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=5,
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
        for batch_idx, (image, label) in enumerate(dataloader):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            start = time.time()
            output = model(image)
            loss = criterion(output, image.sub(label))
            total_loss += loss.data.cpu().numpy()
            loss.backward()
            end = time.time()
            calc_time += (end - start)
            optimizer.step()

            total_loss /= len(dataloader)
            calc_time /= len(dataloader)
            test_loss = evaluate(model, testloader, device)
        
        data_dict = {'num_epoch': num,
                     'training_loss': total_loss,
                     'test_loss': test_loss}
        print('Train Epoch: {} \tLoss: {:.6f} \tTest Loss: {:.6f} \tCalculation Time: {:.4f}sec'.format(
            num, total_loss, test_loss, calc_time))

        if num % 10 == 0:
            denoised_imgs, test_psnr = denoise_test(model, testloader, device)
            append_csv_from_dict(experimental_dir, csv_name, data_dict)
            epoch_dir = os.path.join(experimental_dir, str(num))
            save_torch_model(epoch_dir, MODEL_PATH, model)

            for num, denoised in enumerate(denoised_imgs):
                denoised_name = os.path.join(epoch_dir, 'denoised_{}.png'.format(num))
                denoised.save(denoised_name)
    epoch_dir = os.path.join(experimental_dir, str(epoch))
    model_path = os.path.join(epoch_dir, MODEL_PATH)
    return model_path


def train_mix_dataset(experimental_name, base_dir, epoch, times, dev):
    last_model_path = ''
    for num in range(times):
        if num == 0:
            last_model_path = train(experimental_name + 'part{}'.format(num + 1), base_dir, epoch, dev)
        else:
            last_model_path = train(experimental_name + 'part{}'.format(num + 1), base_dir, epoch, dev, last_model_path)


if __name__ == '__main__':
    train('test', './', 10, 'cuda')
    pass
