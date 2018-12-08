import torch.nn as nn
import torch.nn.functional as F


def make_model(args, num_class):
    return ResNet(args, num_class)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_feat, temp_feat, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, temp_feat, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(temp_feat)
        self.conv2 = nn.Conv2d(temp_feat, temp_feat, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(temp_feat)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_feat != self.expansion * temp_feat:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_feat, self.expansion * temp_feat, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion * temp_feat)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += self.shortcut(residual)
        x = F.relu(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_feat, temp_feat, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, temp_feat, kernel_size=1,  bias=True)
        self.bn1 = nn.BatchNorm2d(temp_feat)
        self.conv2 = nn.Conv2d(temp_feat, temp_feat, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(temp_feat)
        self.conv3 = nn.Conv2d(temp_feat, self.expansion*temp_feat, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.expansion*temp_feat)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_feat != self.expansion * temp_feat:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_feat, self.expansion * temp_feat, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion * temp_feat)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += self.shortcut(residual)
        x = F.relu(x)
        return x





class ResNet(nn.Module):
    def __init__(self, args, num_class):
        super(ResNet, self).__init__()
        self.args = args
        self.in_feat = 16
        self.num_class = num_class
        self.cfg = {
            '18': (BasicBlock, [2, 2, 2, 2]),
            '34': (BasicBlock, [3, 4, 6, 3]),
            '50': (Bottleneck, [3, 4, 6, 3]),
            '101': (Bottleneck, [3, 4, 23, 3]),
            '152': (Bottleneck, [3, 8, 36, 3]),
        }

        self.block, self.num_blocks = self.cfg['101']
        self.conv1 =  nn.Conv2d(3, self.in_feat, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_feat)
        self.layer1 = self.make_layer(self.block, 16, self.num_blocks[0], stride=1)
        self.layer2 = self.make_layer(self.block, 32, self.num_blocks[1], stride=2)
        self.layer3 = self.make_layer(self.block, 64, self.num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * self.block.expansion, num_class)


    def make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_feat, planes, stride))
            self.in_feat = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
