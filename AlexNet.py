# Setup & Library

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torchvision import  models

# Device Configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-Parameters

num_inputs = 28 * 28
num_epochs = 1
batch_size = 4
learning_rate = 0.01

# Dataset Preparation

# dataset has PIL Image of range(0,1)
# We transform them to tensor of normalize range of [-1,1]

transform = transform.Compose([
    transform.Resize((224, 224)),
    transform.ToTensor(),
    transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root="\data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root="\data", train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5  # unNormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Get some random image
example = iter(train_loader)
image, label = next(example)


# show the image
# imshow(torchvision.utils.make_grid(image))


# Model
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.Conv1 = nn.Conv2d(3,96,11,4,padding=0)
        self.pool1  = nn.MaxPool2d(kernel_size=3,stride=2)
        self.norm1 = nn.LocalResponseNorm(size=5)
        self.Conv2 = nn.Conv2d(in_channels= 96,out_channels=256,kernel_size= 5, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.norm2 = nn.LocalResponseNorm(size=5)
        self.Conv3 = nn.Conv2d(256,384,kernel_size=3,padding=1)
        self.Conv4 = nn.Conv2d(384,384,kernel_size=3,padding=1)
        self.Conv5 = nn.Conv2d(384,256,kernel_size=3,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.fc6 = nn.Linear(in_features=256*4*4, out_features=4096)
        self.fc7 = nn.Linear(4096,4096)
        self.fc8 = nn.Linear(4096,10)

    def forward(self,x):
        x = F.relu(self.Conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)
        x = F.relu(self.Conv2(x))
        x = self.pool2(x)
        x = self.norm2(x)
        x = F.relu(self.Conv3(x))
        x = F.relu(self.Conv4(x))
        x = F.relu(self.Conv5(x))
        x = self.pool3(x)
        x= torch.flatten(x,1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)

        return x


model = AlexNet()
# Training

criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)

n_total_step = len(train_loader)

for epochs in range(num_epochs):
    for i, (image, label) in enumerate(train_loader):
        # original shape = (4,3,32,32) = 4,3,1024
        # Input layer : Input_Channel : 3, Output_Channel : 6, Kernel_size :5

        image = image.to(device)
        label = label.to(device)

        # forward pass
        output = model(image)
        loss = criteria(output, label)

        # backward propagation

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epochs [{epochs + 1}/{num_epochs}],[{i + 1}/{n_total_step}],Loss:[{loss.item()}]")

print("Finished Training")



with torch.no_grad():
    n_correct = 0
    n_sample = 0

    n_class_correct = [0 for i in range(10)]
    n_class_sample = [0 for i in range(10)]

    for image, labels in test_loader:
        image = image.to(device)
        labels = labels.to(device)

        output = model(image)

        # Max Return (value, index)

        _, predicted = torch.max(output, 1)
        n_sample += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_sample[label] += 1


        accuracy = 100.0 * n_correct / n_sample
        print(f'Accuracy of the network: {accuracy} %')

        for i in range(10):
            accuracy = 100.0 * n_class_correct[i] / n_class_sample[i]
            print(f'Accuracy of {classes[i]}: {accuracy} %')
