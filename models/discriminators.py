import torch
from torch import nn
from models.nodes import *
from models.encoders import *
import global_v as glv
import torch.nn.functional as F
from numbers import Number
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
import numpy as np


class Discriminator_EM_MNIST(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1),  # -4
            nn.BatchNorm2d(32),
            LIFNode()  # nn.LeakyReLU(0.2)
        )  # (24,24)
        self.pl1 = nn.AvgPool2d(2, stride=2)  # (12,12)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            LIFNode()  # nn.LeakyReLU(0.2)
        )
        self.pl2 = nn.AvgPool2d(2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),
            LIFNode()  # nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1),
            MPNode()  # nn.Sigmoid()
        )

    def forward(self, input, is_imgs=False):
        if is_imgs:
            # input are original images
            input = self.encoder(input)
            # print(input.shape)
        # input.shape = (n_steps,...)
        output = []
        for x in input:
            # print(x.shape)
            x = self.conv1(x)
            x = self.pl1(x)
            x = self.conv2(x)
            x = self.pl2(x)
            x = x.view(x.shape[0], -1)
            x = self.fc1(x)
            x = self.fc2(x)
            output.append(x)
        # output.shape = (n_steps, batch_size, 1)
        res_mem = output[-1] / glv.network_config['n_steps']  # (batch_size, 1)
        return res_mem


class Discriminator_EM_CelebA(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1),  # -4
            nn.BatchNorm2d(32),
            LIFNode()  # nn.LeakyReLU(0.2)
        )  # (60,60)
        self.pl1 = nn.AvgPool2d(2, stride=2)  # (30,30)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            LIFNode()  # nn.LeakyReLU(0.2)
        )  # (26,26)
        self.pl2 = nn.AvgPool2d(2, stride=2)  # (13,13)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 13 * 13, 2048),
            LIFNode()  # nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1),
            MPNode()  # nn.Sigmoid()
        )

    def forward(self, input, is_imgs=False):
        if is_imgs:
            # input are original images
            input = self.encoder(input)
            # print(input.shape)
        # input.shape = (n_steps,...)
        output = []
        for x in input:
            # print(x.shape)
            x = self.conv1(x)
            x = self.pl1(x)
            x = self.conv2(x)
            x = self.pl2(x)
            x = x.view(x.shape[0], -1)
            x = self.fc1(x)
            x = self.fc2(x)
            output.append(x)
        # output.shape = (n_steps, batch_size, 1)
        res_mem = output[-1] / glv.network_config['n_steps']  # (batch_size, 1)
        return res_mem


class Discriminator_Mix_CelebA(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1),  # -4
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)  # nn.LeakyReLU(0.2)
        )  # (60,60)
        self.pl1 = nn.AvgPool2d(2, stride=2)  # (30,30)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)  # nn.LeakyReLU(0.2)
        )  # (26,26)
        self.pl2 = nn.AvgPool2d(2, stride=2)  # (13,13)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 13 * 13, 2048),
            nn.LeakyReLU(0.2)  # nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.pl1(x)
        x = self.conv2(x)
        x = self.pl2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Discriminator_MP(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.net = nn.Sequential(nn.Linear(784, 400), LIFNode(),
                                 nn.Linear(400, 1), MPNode())
        self.sig = nn.Sigmoid()

    def forward(self, inputs, is_imgs=False):
        if is_imgs:
            inputs = self.encoder(inputs)
        output = []
        for x in inputs:
            x = self.net(x)
            output.append(x)
        res_mem = output[-1] / glv.network_config['n_steps']  # (batch_size, 1)
        return self.sig(res_mem)


class Discriminator_EM_DVS_64(nn.Module):
    """
    for DVS data 64x64
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=1),  # -4, dvs has 2 channles
            nn.BatchNorm2d(32),
            LIFNode()  # nn.LeakyReLU(0.2)
        )  # (60,60)
        self.pl1 = nn.AvgPool2d(2, stride=2)  # (30,30)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            LIFNode()  # nn.LeakyReLU(0.2)
        )  # (26,26)
        self.pl2 = nn.AvgPool2d(2, stride=2)  # (13,13)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 13 * 13, 2048),
            LIFNode()  # nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1),
            MPNode()  # nn.Sigmoid()
        )

    def forward(self, input, is_imgs=False):
        # input.shape = (B,T)
        input = input.transpose(0, 1)
        output = []
        for x in input:
            # print(x.shape)
            x = self.conv1(x)
            x = self.pl1(x)
            x = self.conv2(x)
            x = self.pl2(x)
            x = x.view(x.shape[0], -1)
            x = self.fc1(x)
            x = self.fc2(x)
            output.append(x)
        # output.shape = (n_steps, batch_size, 1)
        res_mem = output[-1] / glv.network_config['n_steps']  # (batch_size, 1)
        return res_mem


class Discriminator_EM_DVS_28(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=1),  # -4
            nn.BatchNorm2d(32),
            LIFNode()  # nn.LeakyReLU(0.2)
        )  # (24,24)
        self.pl1 = nn.AvgPool2d(2, stride=2)  # (12,12)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            LIFNode()  # nn.LeakyReLU(0.2)
        )
        self.pl2 = nn.AvgPool2d(2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),
            LIFNode()  # nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1),
            MPNode()  # nn.Sigmoid()
        )

    def forward(self, input, is_imgs=False):
        output = []
        for x in input:
            # print(x.shape)
            x = self.conv1(x)
            x = self.pl1(x)
            x = self.conv2(x)
            x = self.pl2(x)
            x = x.view(x.shape[0], -1)
            x = self.fc1(x)
            x = self.fc2(x)
            output.append(x)
        # output.shape = (n_steps, batch_size, 1)
        res_mem = output[-1] / glv.network_config['n_steps']  # (batch_size, 1)
        return res_mem


class Discriminator_MP_DVS_28(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784 * 2, 400), LIFNode(),
                                 nn.Linear(400, 1), MPNode())
        self.sig = nn.Sigmoid()

    def forward(self, inputs):
        # print(1111)
        output = []
        for x in inputs:
            x = self.net(x)
            output.append(x)

        res_mem = output[-1] / glv.network_config['n_steps']  # (batch_size, 1)
        return self.sig(res_mem)
