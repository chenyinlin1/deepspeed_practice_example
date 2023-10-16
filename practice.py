import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 确定使用哪一块gpu
import torch
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import argparse
import deepspeed
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def add_argument():
    parser=argparse.ArgumentParser(description='CIFAR')
    # Train.
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
    parser.add_argument('--isDeepSpeed',action='store_true', help='是否使用deepspeed')
    args, _ = parser.parse_known_args()
    if args.isDeepSpeed:
        # Include DeepSpeed configuration arguments.   
        parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    args=parser.parse_args()
    return args

def is_main_process() -> bool:
    """判断当前进程是否是主进程,一些保存，输出的程序只在主进程上面运行"""
    return not dist.is_initialized() or dist.get_rank() == 0
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    '''1、超参数部分'''
    args = add_argument()

    '''2、模型部分'''
    model = Net()
    criterion = nn.CrossEntropyLoss()
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    if args.isDeepSpeed:
        model,*_ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters,optimizer=optimizer)
        device = torch.device('cuda',args.local_rank)#放置到当前模型所在的gpu上面。
    else:
        device = torch.device('cuda')
        model = model.to(device)

    '''3、数据集部分'''
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./DATA/CIFAR10', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./DATA/CIFAR10', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    '''处理数据集'''
    if args.isDeepSpeed:
        trainloader =  DataLoader(trainset,batch_size=batch_size,sampler=DistributedSampler(trainset, shuffle=True),
                                    num_workers=16,pin_memory=True)
    else:
        trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=16,pin_memory=True)

    progress_bar = tqdm(
        total=len(trainloader),
        desc=f'Training 1/1 epoch',
        position=0,
        leave=True,
        disable=not is_main_process(),
    )
    print('the length of trainloader',len(trainloader))#通过trainloader的数目可以检查使用deepspeed后，每张显卡上面的数据是否不一样。
    for epoch in range(100):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
            progress_bar.update(1)
    print('Finished Training')
    if is_main_process():
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
