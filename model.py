import torch
import random
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models

IMG_SHAPE = (3, 224, 224)
NUM_EPOCHS = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ImageDataset(Dataset):
    def __init__(self, num_imgs=128):
        self.num_imgs = num_imgs
        self.data_x1 = [torch.rand(IMG_SHAPE) for i in range(num_imgs)]
        self.data_x2 = [torch.rand(IMG_SHAPE) for i in range(num_imgs)]
        self.data_y = [torch.randint(0, 2, (1, )) for i in range(num_imgs)]

    def __getitem__(self, index):
        return self.data_x1[index], self.data_x2[index], self.data_y[index]

    def __len__(self):
        return self.num_imgs


class MetricLearningNet(torch.nn.Module):
    def __init__(self):
        super(MetricLearningNet, self).__init__()
        self.net1 = models.resnet18()
        self.net2 = models.resnet18()
        self.net1.fc = torch.nn.Linear(512, 100)
        self.net2.fc = torch.nn.Linear(512, 100)
        self.last_linear = torch.nn.Linear(200, 1)
        self.squash_layer = torch.nn.Sigmoid()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.criterion = torch.nn.MSELoss()

    def forward(self, x1, x2):
        feature1 = self.net1(x1)
        feature2 = self.net2(x2)
        activation = self.last_linear(torch.cat((feature1, feature2), 1))
        distance = self.squash_layer(activation)
        return distance

    def distance(self, x1, x2):
        return self.forward(x1, x2)

    def fit(self, dataloader):
        since = time.time()
        for epoch in range(NUM_EPOCHS):
            batch = 1
            for batch_x1, batch_x2, batch_y in dataloader:
                bx1 = batch_x1.to(device)
                bx2 = batch_x2.to(device)
                by = batch_y.to(device)

                self.optimizer.zero_grad()
                by_pred = self.forward(bx1, bx2).squeeze()
                by = by.squeeze()
                loss = self.criterion(by_pred, by)

                loss.backward()
                self.optimizer.step()
                print('Epoch:{:02d}/{}, Batch:{:02d}, Loss:{:.4f}'.format(
                    epoch + 1, NUM_EPOCHS, batch, loss))
                batch += 1
        print('Training completed in {:.2f}s'.format(time.time() - since))


def main():
    dataset = ImageDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    print('data loaded...')
    mlnet = MetricLearningNet().to(device)
    print('model loaded...')
    print('start training...')
    mlnet.fit(dataloader)


if __name__ == '__main__':
    main()