import numpy as np
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        out = self.relu(x)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, in_channels=64, out_channels=64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, in_channels=64, out_channels=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, in_channels=128, out_channels=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, in_channels=256, out_channels=512, blocks=layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        identity_downsample = None
        if stride != 1 or in_channels != out_channels:
            identity_downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(in_channels, out_channels, stride=stride, identity_downsample=identity_downsample))
        in_channels = out_channels
        for a in range(1, blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

def resnet18():
    layers = [2, 2, 2, 2]
    block = BasicBlock
    model = ResNet(block, layers)
    return model

def test_architecture():
    net = resnet18()
    summary(net, input_size=(3, 224, 224))
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.size())

test_architecture()



