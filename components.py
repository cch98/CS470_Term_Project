import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Conv2d(64, 128, 5, stride = 1, padding = 2)
        self.layer2 = nn.Conv2d(128, 64, 3, stride = 1, padding = 1)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x), negative_slope = 0.1)
        x = F.leaky_relu(self.layer2(x), negative_slope = 0.1)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Conv2d(64, 128, 3, stride = 1, padding = 1)
        self.layer2 = nn.Conv2d(128, 64, 5, stride = 1, padding = 2)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x), negative_slope = 0.1)
        x = F.leaky_relu(self.layer2(x), negative_slope = 0.1)
        return x


class DenceBlock(nn.Module):
    def __init__(self):
        super(DenceBlock, self).__init__()
        self.layer1 = nn.Conv2d(64, 64, 3, stride = 1, padding = 1)
        self.layer2 = nn.Conv2d(128, 64, 3, stride = 1, padding = 1)
        self.layer3 = nn.Conv2d(192, 64, 3, stride = 1, padding =1)
        self.layer4 = nn.Conv2d(256, 64, 3, stride = 1, padding = 1)


    def forward(self, x):
        x_ = x
        x = F.leaky_relu(self.layer1(x_), negative_slope = 0.1)
        x_ = torch.cat((x_, x), 1)
        x = F.leaky_relu(self.layer2(x_), negative_slope = 0.1)
        x_ = torch.cat((x_, x), 1)
        x = F.leaky_relu(self.layer3(x_), negative_slope = 0.1)
        x_ = torch.cat((x_, x), 1)
        x = F.leaky_relu(self.layer4(x_), negative_slope = 0.1)

        return x

class RRDB(nn.Module):
    def __init__(self):
        super(RRDB, self).__init__()
        self.B1 = DenceBlock()
        self.B2 = DenceBlock()
        self.B3 = DenceBlock()

    def forward(self, x):
        x0 = x
        x_ = x
        x = self.B1.forward(x_)
        x_ = x_+0.2*x
        x = self.B2.forward(x_)
        x_ = x_+0.2*x
        x = self.B3.forward(x_)
        x_ = x_+0.2*x
        x = x0+0.2*x_

        return x



class CbCrNet(nn.Module):
    def __init__(self):
        super(CbCrNet, self).__init__()
        self.layer1 = nn.Conv2d(6, 64, 3, stride = 1, padding = 1)
        self.encoder = Encoder()
        self.B1 = RRDB()
        self.B2 = RRDB()
        # self.B3 = RRDB()
        self.decoder = Decoder()
        self.layer2 = nn.Conv2d(64, 3, 3, stride = 1, padding = 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.encoder(x)
        x_ = x
        x = self.B1(x)
        x = self.B2(x)
        # x = self.B3(x)
        x = x+x_
        x = self.decoder(x)
        x = torch.tanh(self.layer2(x))

        return x

