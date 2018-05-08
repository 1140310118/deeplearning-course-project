import torch
from torchvision import transforms, models, datasets

NCHW = (4,3,224,224)
NUM_EPOCHS = 10

class MetricLearningNet(torch.nn.Module):

    def __init__(self):
        super(MetricLearningNet, self).__init__()
        self.net1 = models.resnet18()
        self.net2 = models.resnet18()
        self.net1.fc = torch.nn.Linear(512, 100)
        self.net2.fc = torch.nn.Linear(512, 100)
        self.distance = torch.nn.Linear(200, 1)

    def forward(self, x1, x2):
        feature1 = self.net1(x1)
        feature2 = self.net2(x2)
        distance = self.distance(torch.cat((feature1, feature2), 1))
        return distance

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mlnet = MetricLearningNet().to(device)
    x1 = torch.rand(NCHW).to(device)
    x2 = torch.rand(NCHW).to(device)
    y = torch.randint(0,2,(NCHW[0],)).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mlnet.parameters())

    for i in range(NUM_EPOCHS):
        optimizer.zero_grad()
        distance = mlnet(x1, x2).squeeze()
        loss = criterion(distance, y)
        print(i, loss)
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()