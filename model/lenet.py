
import torch.nn as nn


def make_model(args, num_class, parent=False):
    return LeNet(args, num_class)


def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=0, bias=bias)


class LeNet(nn.Module):
    def __init__(self, args, num_class):
        super(LeNet, self).__init__()
        self.num_class = num_class
        act = nn.ReLU(True)

        body = [conv(3, 6, kernel_size=5)]
        body.append(act)
        body.append(nn.MaxPool2d(2))

        body.append(conv(6, 16, kernel_size=5))
        body.append(nn.ReLU(True))
        body.append(nn.MaxPool2d(2))

        tail = []
        tail.append(nn.Linear(16*5*5, 120))
        tail.append(nn.ReLU(True))
        tail.append(nn.Linear(120, 84))
        tail.append(nn.ReLU(True))
        tail.append(nn.Linear(84, self.num_class))

        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        x = self.tail(x)
        return x
