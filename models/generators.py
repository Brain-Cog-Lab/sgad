import torch
from torch import nn
from models.nodes import *
from models.encoders import *
import global_v as glv
import torch.nn.functional as F
from numbers import Number
from torch.autograd import Variable


class Generator_MP_Scoring_Mnist(nn.Module):
    '''utilizing sigmoid function before scoring'''

    def __init__(self, input_dim=100, is_split=False):
        super().__init__()
        self.is_split = is_split
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            LIFNode()  # nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 7 * 7),
            LIFNode()  # nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), LIFNode())
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
            ScoringMP(scoring_mode=glv.network_config['scoring_mode']
                      )  # nn.Sigmoid()  # nn.Tanh()
        )
        # self.sig = nn.Sigmoid()

    def forward(self, input):
        input = self.encoder(input)
        # input.shape = (num_steps,...)
        output = []
        for x in input:
            x = self.br1(self.fc1(x))
            x = self.br2(self.fc2(x))
            x = x.reshape(-1, 128, 7, 7)
            x = self.conv1(x)
            x = self.conv2(x)  # (batch_size,1,28,28), [0,1]
            output.append(x)
        if not self.is_split:
            img = output[-1]
            img_spike = img.repeat(glv.network_config['n_steps'], 1, 1, 1, 1)
        else:
            output = torch.stack(output, dim=0)
            img_spike = output
        return img_spike


class Generator_MP_Scoring_CelebA(nn.Module):
    '''utilizing sigmoid function before scoring'''

    def __init__(self, input_dim=100, is_split=False):
        super().__init__()
        self.is_split = is_split
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            LIFNode()  # nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 8 * 8)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 8 * 8),
            LIFNode()  # nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), LIFNode())
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), LIFNode())
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
            ScoringMP(scoring_mode=glv.network_config['scoring_mode'],
                      dataset_name="CelebA")  # nn.Sigmoid()  # nn.Tanh()
        )
        # self.sig = nn.Sigmoid()

    def forward(self, input):
        input = self.encoder(input)
        # input.shape = (num_steps,...)
        output = []
        for x in input:
            x = self.br1(self.fc1(x))
            x = self.br2(self.fc2(x))
            x = x.reshape(-1, 128, 8, 8)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)  # (batch_size,1,28,28), [0,1]
            output.append(x)
        if not self.is_split:
            img = output[-1]
            img_spike = img.repeat(glv.network_config['n_steps'], 1, 1, 1, 1)
        else:
            output = torch.stack(output, dim=0)
            img_spike = output
        return img_spike


class Generator_MP(nn.Module):

    def __init__(self, input_dim=100, is_split=False):
        super().__init__()
        self.is_split = is_split
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            LIFNode()  # nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 7 * 7),
            LIFNode()  # nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            LIFNode()  # nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            MPNode()  # nn.Sigmoid()  # nn.Tanh()
        )
        self.sig = nn.Sigmoid()

    def forward(self, input):
        input = self.encoder(input)
        # input.shape = (num_steps,...)
        output = []
        for x in input:
            x = self.br1(self.fc1(x))
            x = self.br2(self.fc2(x))
            x = x.reshape(-1, 128, 7, 7)
            x = self.conv1(x)
            x = self.conv2(x)  # (batch_size,1,28,28)
            output.append(x)
        if not self.is_split:
            res_mem = output[-1] / glv.network_config[
                'n_steps']  # (batch_size, 1, 28, 28)
            img = self.sig(res_mem)
            img_spike = img.repeat(glv.network_config['n_steps'], 1, 1, 1, 1)
        else:
            output = torch.stack(output, dim=0)
            img_spike = self.sig(output)
        return img_spike


class Generator_MP_CelebA(nn.Module):

    def __init__(self, input_dim=100, is_split=False):
        super().__init__()
        self.is_split = is_split
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            LIFNode()  # nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 8 * 8)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 8 * 8),
            LIFNode()  # nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            LIFNode()  # nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), LIFNode())
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # (b,3,64,64)
            MPNode()  # nn.Sigmoid()  # nn.Tanh()
        )
        self.sig = nn.Sigmoid()

    def forward(self, input):
        input = self.encoder(input)
        # input.shape = (num_steps,...)
        output = []
        for x in input:
            x = self.br1(self.fc1(x))
            x = self.br2(self.fc2(x))
            x = x.reshape(-1, 128, 8, 8)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)  # (batch_size,3,64,64)
            output.append(x)
        if not self.is_split:
            res_mem = output[-1] / glv.network_config[
                'n_steps']  # (batch_size, 3, 64, 64)
            img = self.sig(res_mem)
            img_spike = img.repeat(glv.network_config['n_steps'], 1, 1, 1, 1)
        else:
            output = torch.stack(output, dim=0)
            img_spike = self.sig(output)
        return img_spike


class Generator_SNN(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            LIFNode()  # nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 7 * 7),
            LIFNode()  # nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            LIFNode()  # nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            LIFNode()  # nn.Sigmoid()  # nn.Tanh()
        )

    def forward(self, input):
        input = self.encoder(input)
        # input.shape = (num_steps,...)
        output = []
        for x in input:
            x = self.br1(self.fc1(x))
            x = self.br2(self.fc2(x))
            x = x.reshape(-1, 128, 7, 7)
            x = self.conv1(x)
            x = self.conv2(x)
            output.append(x)
        return torch.stack(output, dim=0)  # return.shape = (num_steps,...)


class Generator_MP_Scoring_DVS_64(nn.Module):
    '''for DVS 64x64 size'''

    def __init__(self, input_dim=100, is_split=False):
        super().__init__()
        self.is_split = is_split
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            LIFNode()  # nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 8 * 8)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 8 * 8),
            LIFNode()  # nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), LIFNode())
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), LIFNode())
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 2, 4, stride=2,
                               padding=1),  # dvs data has 2 channels
            nn.Sigmoid(),
            ScoringMP(scoring_mode=glv.network_config['scoring_mode'],
                      dataset_name="dvs_64")  # nn.Sigmoid()  # nn.Tanh()
        )
        # self.sig = nn.Sigmoid()

    def forward(self, input):
        input = self.encoder(input)
        # input.shape = (num_steps,...)
        output = []
        for x in input:
            x = self.br1(self.fc1(x))
            x = self.br2(self.fc2(x))
            x = x.reshape(-1, 128, 8, 8)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)  # (batch_size,1,28,28), [0,1]
            output.append(x)
        if not self.is_split:
            img = output[-1]
            img_spike = img.repeat(glv.network_config['n_steps'], 1, 1, 1, 1)
        else:
            output = torch.stack(output, dim=0)
            img_spike = output
        return img_spike


class Generator_MP_Scoring_DVS_28(nn.Module):
    '''utilizing sigmoid function before scoring'''

    def __init__(self, input_dim=100, is_split=False):
        super().__init__()
        self.is_split = is_split
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            LIFNode()  # nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 7 * 7),
            LIFNode()  # nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), LIFNode())
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 2, 4, stride=2, padding=1),
            nn.Sigmoid(),
            ScoringMP(scoring_mode=glv.network_config['scoring_mode'],
                      dataset_name='dvs_mnist_28')  # nn.Sigmoid()  # nn.Tanh()
        )
        # self.sig = nn.Sigmoid()

    def forward(self, input):
        input = self.encoder(input)
        # input.shape = (num_steps,...)
        output = []
        for x in input:
            x = self.br1(self.fc1(x))
            x = self.br2(self.fc2(x))
            x = x.reshape(-1, 128, 7, 7)
            x = self.conv1(x)
            x = self.conv2(x)  # (batch_size,1,28,28), [0,1]
            output.append(x)
        if not self.is_split:
            img = output[-1]
            img_spike = img.repeat(glv.network_config['n_steps'], 1, 1, 1, 1)
        else:
            output = torch.stack(output, dim=0)
            img_spike = output
        return img_spike


class Generator_MP_DVS_28(nn.Module):

    def __init__(self, input_dim=100, is_split=False):
        super().__init__()
        self.is_split = is_split
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            LIFNode()  # nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 7 * 7),
            LIFNode()  # nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            LIFNode()  # nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 2, 4, stride=2, padding=1),
            MPNode()  # nn.Sigmoid()  # nn.Tanh()
        )
        self.sig = nn.Sigmoid()

    def forward(self, input):
        input = self.encoder(input)
        # input.shape = (num_steps,...)
        output = []
        for x in input:
            x = self.br1(self.fc1(x))
            x = self.br2(self.fc2(x))
            x = x.reshape(-1, 128, 7, 7)
            x = self.conv1(x)
            x = self.conv2(x)  # (batch_size,1,28,28)
            output.append(x)
        if not self.is_split:
            res_mem = output[-1] / glv.network_config[
                'n_steps']  # (batch_size, 1, 28, 28)
            img = self.sig(res_mem)
            img_spike = img.repeat(glv.network_config['n_steps'], 1, 1, 1, 1)
        else:
            output = torch.stack(output, dim=0)
            img_spike = self.sig(output)
        return img_spike


class Generator_SNN_DVS_28(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            LIFNode()  # nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 7 * 7),
            LIFNode()  # nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            LIFNode()  # nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 2, 4, stride=2, padding=1),
            LIFNode()  # nn.Sigmoid()  # nn.Tanh()
        )

    def forward(self, input):
        input = self.encoder(input)
        # input.shape = (num_steps,...)
        output = []
        for x in input:
            x = self.br1(self.fc1(x))
            x = self.br2(self.fc2(x))
            x = x.reshape(-1, 128, 7, 7)
            x = self.conv1(x)
            x = self.conv2(x)
            output.append(x)
        return torch.stack(output, dim=0)  # return.shape = (num_steps,...)
