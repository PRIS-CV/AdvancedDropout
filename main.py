'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from mlp import MLP
from variationalBayesDropout import AdvancedDropout

os.environ["CUDA_VISIBLE_DEVICES"] = "0"# use GPU 0
nb_epoch = 300# number of epochs
lr = 0.1# initial learning rate

print("OPENING " + 'results_train.csv')
results_train_file = open('results_train.csv', 'a')
results_train_file.write('epoch,train_acc,train_loss\n')
results_train_file.flush()

print("OPENING " + 'results_test.csv')
results_test_file = open('results_test.csv', 'a')
results_test_file.write('epoch,test_acc,test_loss\n')
results_test_file.flush()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='/data/xiejiyang/data', # your data path
                                        train=True, 
                                        download=True, 
                                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=256, 
                                          shuffle=True, 
                                          num_workers=4)

testset = torchvision.datasets.CIFAR10(root='/data/xiejiyang/data', # your data path
                                       train=False, 
                                       download=True, 
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=256, 
                                         shuffle=False, 
                                         num_workers=4)

# Model
print('==> Building model..')
net = models.vgg16_bn(pretrained=False)
net = nn.Sequential(*list(net.children())[:-2])# get vgg16 backbone (only all the conv layers)
class model_vgg(nn.Module):
    def __init__(self, model, node_list):
        '''
        params:
        model (nn.Sequential): backbone
        node_list (int list): n elements where the first element is the input layers' node number, 
                                the middle n-1 elements are hidden layers' node numbers, 
                                and the last element is the output layers' node number
        '''
        super(model_vgg, self).__init__() 
        self.features = model
        self.classifier = MLP(node_list)# fc layers

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

net = model_vgg(net, [512*1*1, 512, 10])# construct the model

device = torch.device("cuda")
net = net.to(device)
net.features.to(device)
net.classifier.to(device)

criterion = nn.CrossEntropyLoss()

# set optimizer
dp_params = []
res_params = []
for m in net.classifier.classifier:
    if isinstance(m, AdvancedDropout):
        dp_params.append(m.weight_h)
        dp_params.append(m.bias_h)
        dp_params.append(m.weight_mu)
        dp_params.append(m.bias_mu)
        dp_params.append(m.weight_sigma)
        dp_params.append(m.bias_sigma)
    elif isinstance(m, nn.Linear):
        res_params.append(m.weight)
        if hasattr(m,"bias"):
            res_params.append(m.bias)

optimizer = optim.SGD([{'params':net.features.parameters(), 'lr':lr},
                       {'params':res_params,'lr':lr},
                       {'params':dp_params,'lr':1e-4}], momentum=0.9, weight_decay=5e-4)
# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    idx = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        idx = batch_idx
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss +=loss.detach().cpu()
        _, predicted = torch.max(outputs,-1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().item()

    train_acc = 100.*correct/total
    train_loss = train_loss/(idx+1)
    print('Iteration %d, train_acc = %.4f, train_loss = %.4f' % (epoch, train_acc, train_loss))
    results_train_file.write('%d,%.4f,%.4f\n' % (epoch, train_acc, train_loss))
    results_train_file.flush()

    return train_acc, train_loss

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)

        loss = criterion(outputs, targets)

        test_loss += loss.detach().cpu()
        _, predicted =  torch.max(outputs,-1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().item()

    test_acc = 100.*correct/total
    test_loss = test_loss/(idx+1)
    print('Iteration %d, test_acc = %.4f, test_loss = %.4f' % (epoch, test_acc, test_loss))
    results_test_file.write('%d,%.4f,%.4f\n' % (epoch, test_acc, test_loss))
    results_test_file.flush()

    return test_acc, test_loss

# net.load_state_dict(torch.load('checkpoint.pth'))
for epoch in range(0, nb_epoch):
    print('\nEpoch: %d' % epoch)
    if epoch in [150, 225]:# learning rate drop
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
    train(epoch)
    test(epoch)
    torch.save(net.state_dict(), 'checkpoint.pth')
