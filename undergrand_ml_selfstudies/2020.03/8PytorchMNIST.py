import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time



class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # x = F.torch.relu(self.fc1(x))
        # x = F.torch.relu(self.fc2(x))
        # x = F.torch.relu(self.fc3(x))
        x = F.torch.sigmoid(self.fc1(x))
        x = F.torch.sigmoid(self.fc2(x))
        x = F.torch.sigmoid(self.fc3(x))
        return F.log_softmax(self.fc4(x), dim=1)


def main():
    train = datasets.MNIST("", train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))

    test = datasets.MNIST("", train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))

    trainset = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)

    net = Net()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    epochs = 5

    time_start = time.time()

    for epoch in range(epochs):
        for data in trainset:
            X, y = data
            net.zero_grad()
            output = net(X.view(-1, 28**2))
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}; loss: {loss}")

    print(f"Time taken: {time.time()-time_start}")

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testset:
            X, y = data
            outputs = net(X.view(-1, 28**2))
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f"Accuracy: {100*correct / total}%")


if __name__ == "__main__":
    main()
