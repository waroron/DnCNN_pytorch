from dncnn import DnCNN, DenoisingDatasets
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision.transforms as transforms


def evaluate(model, test_set, device):
    criterion = nn.MSELoss()
    sum_loss = 0
    for num, (image, org) in enumerate(test_set):
        image = image.to(device)
        org = org.to(device)
        output = model.denoise(image)
        loss = criterion(output, image.sub(org))
        sum_loss += loss.data.cpu().numpy()

    print('test loss: {}'.format(sum_loss / len(test_set)))


if __name__ == '__main__':
    MODEL_PATH = 'model.pth'
    transform = transforms.Compose(
        [transforms.ToTensor()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    epoch = 100
    model = DnCNN().to(device)
    dataset = DenoisingDatasets(dir='BSDS200/', data_transform=transform)
    test_set = DenoisingDatasets(dir='Set5/', data_transform=transform)
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
    minibatch_time = 100

    for num in range(minibatch_time):
        total_loss = 0
        for batch_idx, (image, label) in enumerate(dataloader):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, image.sub(label))
            total_loss += loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} \tLoss: {:.6f}'.format(
            num, total_loss / len(dataloader)))
        evaluate(model, testloader, device)
        torch.save(model, MODEL_PATH)
