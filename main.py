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
            denoised_fig.append(denoised)
        print('Test PSNR: {:.6f}'.format(sum_loss / len(denoised_fig)))
        return denoised_fig, sum_loss / len(denoised_fig)


def train(experimental_name, base_dir, epoch, dev):
    MODEL_PATH = 'model.pth'
    DATASET = 'BSDS200/'
    TESTSET = 'Set14/'
    csv_name = 'epoch_data.csv'
    dataset_dir = os.path.join(base_dir, DATASET)
    testset_dir = os.path.join(base_dir, TESTSET)
    transform = transforms.Compose(
        [transforms.ToTensor()])

    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    print(device)
    model = DnCNN().to(device)
    
    dataset = DenoisingDatasets(dir=dataset_dir, data_transform=transform)
    test_set = DenoisingDatasets(dir=dataset_dir, data_transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=True
    )

    testloader = DataLoader(
        dataset=test_set,
        batch_size=5,
        shuffle=False
    )

    optimizer = Adam(model.parameters())
    criterion = nn.MSELoss()
    experimental_dir = os.path.join(base_dir, experimental_name)
    
    if not os.path.isdir(experimental_dir):
        print('make dir {}'.format(experimental_dir))
        os.mkdir(experimental_dir)

    for num in range(epoch):
        total_loss = 0
        calc_time = 0
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


if __name__ == '__main__':
    train('test', './', 10, 'cuda')
    pass
