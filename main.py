from dncnn import DnCNN, DenoisingDatasets
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import cv2


def evaluate(model, test_set, device):
    for num, (image, org) in enumerate(test_set):
        image = image.to(device)
        org = org.to(device)
        output = model.denoise(image)[0]

        cv2.imshow('test', output[0])
        cv2.waitKey(0)

        print(output)


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(device)
    epoch = 100
    model = DnCNN().to(device)
    dataset = DenoisingDatasets(dir='BSDS200/', data_transform=transform)
    test_set = DenoisingDatasets(dir='Set14/', data_transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        shuffle=True
    )

    testloader = DataLoader(
        dataset=test_set,
        batch_size=2,
        shuffle=False
    )

    optimizer = Adam(model.parameters())
    criterion = nn.MSELoss()

    for batch_idx, (image, label) in enumerate(dataloader):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, image.sub(label))
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(image), len(dataloader.dataset),
            100. * batch_idx / len(dataloader), loss))

        evaluate(model, testloader, device)
