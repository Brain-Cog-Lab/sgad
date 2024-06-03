import abc
import math
from abc import ABC, abstractmethod
import pstats
from select import select
from turtle import forward
from typing import ForwardRef

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torchvision.utils import save_image
from numbers import Number
from torch.autograd import Variable

ALPHA = 2.


class AtanGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, alpha):
        ctx.save_for_backward(inputs, alpha)
        return inputs.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        grad_alpha = None

        # saved_tensor[0] == nan here !
        shared_c = grad_output / (1 + (ctx.saved_tensors[1] * math.pi / 2 *
                                       ctx.saved_tensors[0]).square())
        if ctx.needs_input_grad[0]:
            grad_x = ctx.saved_tensors[1] / 2 * shared_c
        if ctx.needs_input_grad[1]:
            # 由于alpha只有一个元素，因此梯度需要求和，变成标量
            grad_alpha = (ctx.saved_tensors[0] / 2 * shared_c).sum()
        '''with torch.no_grad():
            print(grad_output.mean())
            print(ctx.saved_tensors[0].mean())
            print(ctx.saved_tensors[1].mean())
            print(" ")'''

        return grad_x, grad_alpha


class BaseNode(nn.Module, abc.ABC):

    def __init__(self, threshold=1., weight_warmup=False, V_reset=0.):
        super(BaseNode, self).__init__()
        self.threshold = threshold
        self.mem = 0.
        self.spike = 0.
        self.weight_warmup = weight_warmup  # 一般在训练静态数据集 较深网络时使用

    @abc.abstractmethod
    def calc_spike(self):
        pass

    @abc.abstractmethod
    def integral(self, inputs):
        pass

    def forward(self, inputs):
        if self.weight_warmup:
            return inputs
        else:
            self.integral(inputs)
            self.calc_spike()
            return self.spike

    def n_reset(self):
        self.mem = 0.
        self.spike = 0.

    def get_n_fire_rate(self):
        if self.spike is None:
            return 0.
        return float((self.spike.detach() >= self.threshold).sum()) / float(
            np.product(self.spike.shape))


class IFNode(BaseNode):

    def __init__(self, threshold=1., act_fun=AtanGrad):
        super().__init__(threshold)
        self.act_fun = act_fun.apply

    def integral(self, inputs):
        self.mem += inputs

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())


class ScoringNet_1(nn.Module):
    # utilizing dense structure
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.dense1 = nn.Linear(num_inputs, num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(num_hiddens, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        return self.sigmoid(self.dense2(self.relu(self.dense1(X))))


class DotProductAttention(nn.Module):

    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys):
        # note that: q_size == k_size
        # queries.shape = (b,n_q,q_size)
        # keys.shape = (b,n_k,k_size)
        # values.shape = (b,n_k,v_size)
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(
            d)  # (b,q,qs)*(b,ks,k) -> (b,q,k)
        self.attention_weights = torch.softmax(scores, dim=-1)
        return self.attention_weights


class AttentionScoring_1(nn.Module):

    def __init__(self, query_size, key_size, num_hiddens):
        super().__init__()
        self.W_q = nn.Linear(query_size, num_hiddens)
        self.W_k = nn.Linear(key_size, num_hiddens)
        self.attention = DotProductAttention()

    def forward(self, queries, keys):
        # queries.shape = (b,2,784)
        # keys.shape = (b,1,784)
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        self.attention_weights = self.attention(queries, keys)
        return self.attention_weights


class HasInfoNCELoss(ABC):
    r"""
    This class is for those modules that need to compute infoNCE loss
    """

    def __init__(self) -> None:
        super().__init__()
        self._project_x = None
        self._project_latent = None

    def batch_infonce_loss(self, X, X_latent):
        r"""
        This function generates positive/negative examples by same/different examples for a minibatch 
        X.shape = (b,...,original_dim)
        X_latent.shape = (b,...,latent_dim)
        """
        # print(X.device)
        # print(next(self._project_x.parameters()).device)
        projected_X = self._project_x(X).unsqueeze(-2).unsqueeze(
            0)  # (b,...,origianl_dim) => (1,b,...,1,compare_vec_dim)
        projected_latent = self._project_latent(X_latent).unsqueeze(
            -1).unsqueeze(
                1)  # (b,...,latent_dim) => (b,1,...,compare_vec_dim,1)

        score_map = torch.matmul(projected_X,
                                 projected_latent).squeeze(-1).squeeze(
                                     -1)  # (bl,bm,...)

        batch_size = score_map.shape[0]
        labels = torch.ones(
            score_map.shape[1:], device=X.device) * torch.arange(
                batch_size, device=X.device).reshape(
                    [batch_size] + [1] * (len(score_map.shape) - 2))  # (b,...)
        # print(labels.device)
        l = F.cross_entropy(score_map, labels.long(),
                            reduction='mean')  # scalar
        return l

    '''@property
    @abstractmethod
    def _project_x(self):
        pass

    @property
    @abstractmethod
    def _project_latent(self):
        pass'''

    @abstractmethod
    def compute_infonce_loss(self):
        pass


class AttentionScoring_2(nn.Module, HasInfoNCELoss):
    r"""
    utilizing infoNCE method
    """

    def __init__(self, query_size, key_size, num_hiddens, latent_dim,
                 compare_dim):
        super().__init__()

        # mapping for queries
        self.mq = nn.Sequential(nn.Linear(query_size, num_hiddens), nn.ReLU(),
                                nn.Linear(num_hiddens, latent_dim))

        # mapping for keys
        self.mk = nn.Sequential(nn.Linear(key_size, num_hiddens), nn.ReLU(),
                                nn.Linear(num_hiddens, latent_dim))

        self.attention = DotProductAttention()
        self.compare_dim = compare_dim
        self.query_size = query_size
        self.key_size = key_size
        self.latent_dim = latent_dim
        self._project_x = nn.Linear(self.query_size, self.compare_dim)
        self._project_latent = nn.Linear(self.latent_dim, self.compare_dim)

    def forward(self, queries, keys):
        # queries.shape = (b,2,784)
        # keys.shape = (b,1,784)
        queries_latent = self.mq(queries)  # (b,2,latent_dim)
        keys_latent = self.mk(keys)  # (b,1,latent_dim)
        self.attention_weights = self.attention(queries_latent, keys_latent)
        infonce_loss = self.compute_infonce_loss(queries, queries_latent, keys,
                                                 keys_latent)
        return self.attention_weights, infonce_loss

    '''@property
    def _project_x(self):
        return nn.Linear(self.query_size, self.compare_dim)

    @property
    def _project_latent(self):
        return nn.Linear(self.latent_dim, self.compare_dim)'''

    def compute_infonce_loss(self, queries, queries_latent, keys, keys_latent):
        return (self.batch_infonce_loss(queries, queries_latent) +
                self.batch_infonce_loss(keys, keys_latent)) / 2


class LIFNode(BaseNode):

    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad):
        super().__init__(threshold)
        self.tau = tau
        self.act_fun = act_fun.apply

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold, torch.tensor(2.))
        self.mem = self.mem * (1 - self.spike.detach())


class MPNode(BaseNode):

    def __init__(self, tau=2.0):
        super().__init__()
        self.tau = tau

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        # self.spike = self.act_fun(self.mem - self.threshold, torch.tensor(2.))
        # self.mem = self.mem * (1 - self.spike.detach())
        self.spike = self.mem


class ScoringMP(BaseNode):

    def __init__(self, scoring_mode, tau=2.0, dataset_name=None):
        super().__init__()
        self.tau = tau
        self.scoring_mode = scoring_mode
        self.scoring_function = None
        self.down_sample_net = None
        self.cnt = 0
        if self.scoring_mode == 'ScoringNet_1':
            self.scoring_function = ScoringNet_1(num_inputs=784,
                                                 num_hiddens=256)
        elif self.scoring_mode == 'AttentionScoring_1':
            self.scoring_function = AttentionScoring_1(784, 784, 256)
        elif self.scoring_mode == 'AttentionScoring_2':
            self.scoring_function = AttentionScoring_2(784, 784, 256, 128, 64)
        elif self.scoring_mode == "AttentionScoring_RGB":
            if dataset_name == "CIFAR-10":
                self.down_sample_net = nn.Sequential(
                    nn.Flatten(), nn.Linear(3 * 32 * 32, 1024), nn.ReLU())
            elif dataset_name == "CelebA":
                self.down_sample_net = nn.Sequential(
                    nn.Flatten(), nn.Linear(3 * 64 * 64, 1024), nn.ReLU())
            elif dataset_name == "dvs_64":
                self.down_sample_net = nn.Sequential(
                    nn.Flatten(), nn.Linear(2 * 64 * 64, 1024), nn.ReLU())
            elif dataset_name == 'dvs_mnist_28':
                self.down_sample_net = nn.Sequential(
                    nn.Flatten(), nn.Linear(2 * 28 * 28, 1024), nn.ReLU())
            self.scoring_function = AttentionScoring_1(1024, 1024, 256)

    def integral(self, inputs):
        # inputs.shape = (b,1,28,28) or (b,784)
        if isinstance(self.mem, float):
            # print("Hello")
            self.mem = inputs
        else:
            if self.scoring_mode == 'ScoringNet_1':
                batch_size = inputs.shape[0]
                mem_score = self.scoring_function(
                    self.mem.reshape((batch_size, -1)))  # (b,1)
                inputs_score = self.scoring_function(
                    inputs.reshape((batch_size, -1)))  # (b,1)
                mem_score = mem_score.reshape(
                    [batch_size, 1] + [1] *
                    (len(inputs.shape) - 2))  # (b,1,1,1)
                inputs_score = inputs_score.reshape(
                    [batch_size, 1] + [1] *
                    (len(inputs.shape) - 2))  # (b,1,1,1)
                self.mem = (mem_score * self.mem +
                            inputs_score * inputs) / (mem_score + inputs_score)
            elif self.scoring_mode == 'AttentionScoring_1':
                batch_size = inputs.shape[0]
                if len(inputs.shape) == 4:
                    inputs_temp = inputs.reshape((batch_size, -1))  # (b,784)
                    mem_temp = self.mem.reshape((batch_size, -1))  # (b,784)
                else:
                    inputs_temp = inputs
                    mem_temp = self.mem

                # (b,784) => (b,1,784)
                inputs_temp = inputs_temp.unsqueeze(1)
                mem_temp = mem_temp.unsqueeze(1)

                keys = torch.cat([mem_temp, inputs_temp], dim=1)  # (b,2,784)
                attention_weights = self.scoring_function(mem_temp,
                                                          keys)  # (b,1,2)
                mem_score = attention_weights[:, 0, 0]  # (b,)
                inputs_score = attention_weights[:, 0, 1]  # (b,)

                mem_score = mem_score.reshape(
                    [batch_size, 1] + [1] *
                    (len(inputs.shape) - 2))  # (b,1,1,1)
                inputs_score = inputs_score.reshape(
                    [batch_size, 1] + [1] *
                    (len(inputs.shape) - 2))  # (b,1,1,1)

                # print(self.cnt)
                '''print(inputs_score[0, 0, 0, 0] /
                      (mem_score[0, 0, 0, 0] + inputs_score[0, 0, 0, 0]))'''
                '''with open('./exps/show_score/scores.txt', mode='a') as f:
                    score = inputs_score[0, 0, 0, 0] / (mem_score[0, 0, 0, 0] + inputs_score[0, 0, 0, 0])
                    score = str(score.item())
                    f.write(score)
                    f.write('\n')'''
                '''save_image(inputs[0, ...],
                           f'./exps/show_score/t/input{self.cnt}.png')'''
                self.cnt += 1

                self.mem = (mem_score * self.mem +
                            inputs_score * inputs) / (mem_score + inputs_score)
            elif self.scoring_mode == "AttentionScoring_RGB":
                batch_size = inputs.shape[0]
                # input.shape = (b,3,h,w)
                inputs_temp = self.down_sample_net(inputs)  # (b,1024)
                mem_temp = self.down_sample_net(self.mem)
                # (b,1024) => (b,1,1024)
                inputs_temp = inputs_temp.unsqueeze(1)
                mem_temp = mem_temp.unsqueeze(1)

                keys = torch.cat([mem_temp, inputs_temp], dim=1)  # (b,2,1024)
                attention_weights = self.scoring_function(mem_temp,
                                                          keys)  # (b,1,2)
                mem_score = attention_weights[:, 0, 0]  # (b,)
                inputs_score = attention_weights[:, 0, 1]  # (b,)

                mem_score = mem_score.reshape(
                    [batch_size, 1] + [1] *
                    (len(inputs.shape) - 2))  # (b,1,1,1)
                inputs_score = inputs_score.reshape(
                    [batch_size, 1] + [1] *
                    (len(inputs.shape) - 2))  # (b,1,1,1)

                # print(self.cnt)
                '''print(inputs_score[0, 0, 0, 0] /
                      (mem_score[0, 0, 0, 0] + inputs_score[0, 0, 0, 0]))'''
                '''with open('./exps/show_score/scores.txt', mode='a') as f:
                    score = inputs_score[0, 0, 0, 0] / (mem_score[0, 0, 0, 0] + inputs_score[0, 0, 0, 0])
                    score = str(score.item())
                    f.write(score)
                    f.write('\n')'''
                '''save_image(inputs[0, ...],
                           f'./exps/show_score/t_rgb/input{self.cnt}.png')'''
                self.cnt += 1

                self.mem = (mem_score * self.mem +
                            inputs_score * inputs) / (mem_score + inputs_score)
            elif self.scoring_mode == "AttentionScoring_2":
                # print("hello")
                batch_size = inputs.shape[0]
                if len(inputs.shape) == 4:
                    inputs_temp = inputs.reshape((batch_size, -1))  # (b,784)
                    mem_temp = self.mem.reshape((batch_size, -1))  # (b,784)
                else:
                    inputs_temp = inputs
                    mem_temp = self.mem

                # (b,784) => (b,1,784)
                inputs_temp = inputs_temp.unsqueeze(1)
                mem_temp = mem_temp.unsqueeze(1)

                keys = torch.cat([mem_temp, inputs_temp], dim=1)  # (b,2,784)
                attention_weights, infonce_loss = self.scoring_function(
                    mem_temp, keys)  # (b,1,2)
                mem_score = attention_weights[:, 0, 0]  # (b,)
                inputs_score = attention_weights[:, 0, 1]  # (b,)

                mem_score = mem_score.reshape(
                    [batch_size, 1] + [1] *
                    (len(inputs.shape) - 2))  # (b,1,1,1)
                inputs_score = inputs_score.reshape(
                    [batch_size, 1] + [1] *
                    (len(inputs.shape) - 2))  # (b,1,1,1)

                self.mem = (mem_score * self.mem +
                            inputs_score * inputs) / (mem_score + inputs_score)

                # print(infonce_loss)

                return infonce_loss

            elif self.scoring_mode is None:
                self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        self.spike = self.mem

    def forward(self, inputs):
        if self.weight_warmup:
            return inputs
        elif self.scoring_mode == "AttentionScoring_2":
            infonce_loss = self.integral(inputs)
            self.calc_spike()
            return self.spike, infonce_loss
        else:
            self.integral(inputs)
            self.calc_spike()
            return self.spike


class mem_encoder_1(nn.Module):

    def __init__(self, input_dim, output_dim=10, compare_vec_dim=4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.compare_vec_dim = compare_vec_dim
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)
        self.project_mem = nn.Linear(input_dim, self.compare_vec_dim)
        self.project_mem_latent = nn.Linear(output_dim, self.compare_vec_dim)

    def forward(self, X):
        # (b,c,h,w,input_dim) => (b,c,h,w,output_dim)
        mem_latent = self.fc2(self.relu1(self.fc1(X)))
        cpc_loss = self.compute_cpc_loss(
            X.detach(), mem_latent)  # should use X.detach() or simply X ?
        return mem_latent, cpc_loss

    def compute_cpc_loss(self, X, mem_latent):
        # X.shape = (b,c,h,w,input_dim)
        # mem_latent.shape = (b,c,h,w,output_dim)
        # compute cpc loss on same position in different batch
        projected_mem = self.project_mem(X).unsqueeze(-2).unsqueeze(
            0)  # (b,c,h,w,input_dim) => (1,b,c,h,w,1,compare_vec_dim)
        projected_mem_latent = self.project_mem_latent(mem_latent).unsqueeze(
            -1).unsqueeze(
                1)  # (b,c,h,w,output_dim) => (b,1,c,h,w,compare_vec_dim,1)
        score_map = torch.matmul(projected_mem,
                                 projected_mem_latent).squeeze(-1).squeeze(
                                     -1)  # (bl,bm,c,h,w)
        # print(score_map.device)
        b, c, h, w = score_map.shape[0], score_map.shape[2], score_map.shape[
            3], score_map.shape[4]
        labels = torch.ones((b, c, h, w), device=X.device) * torch.arange(
            b, device=X.device).reshape((b, 1, 1, 1))  # (b,c,h,w)
        # print(labels.device)
        l = F.cross_entropy(score_map, labels.long(),
                            reduction='mean')  # scalar
        return l


class VcLIFNode(BaseNode):

    def __init__(self, mem_encoder, tau=2.0, act_fun=AtanGrad):
        super().__init__()
        self.tau = tau
        self.mem_encoder = mem_encoder
        self.to_scalar = nn.Sequential(
            nn.Linear(self.mem_encoder.output_dim, 8), nn.ReLU(),
            nn.Linear(8, 1))  # mapping vector membrane to a scalar
        self.act_fun = act_fun.apply
        self.cpc_loss = None

    def integral(self, inputs):
        #  mem.shape = (b,c,h,w,v), v is the dim of the local feature
        self.mem = self.mem + (inputs -
                               self.mem) / self.tau  # mem is a vector here

    def calc_spike(self):
        mem_latent, self.cpc_loss = self.mem_encoder(
            self.mem)  # (b,c,h,w,output_dim)
        mem_scalar = self.to_scalar(mem_latent).squeeze(-1)  # (b,c,h,w)
        self.spike = self.act_fun(mem_scalar - self.threshold,
                                  torch.tensor(2.))  # (b,c,h,w)
        self.mem = self.mem * (1 - self.spike.detach().unsqueeze(-1))

    def forward(self, inputs):
        if self.weight_warmup:
            return inputs
        else:
            self.integral(inputs)
            self.calc_spike()
            return self.spike, self.cpc_loss


class MemoryMPNode(BaseNode):
    """Return a vector mem, according to historical as well as present scalar mems"""

    def __init__(self, tau=2.0, memory_size=8):
        super().__init__()
        self.tau = tau
        self.memory_size = memory_size
        self.vec_mem = None

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau
        # delete first element of vectorlized membrane, and enqueue present mem to last position
        if self.vec_mem is None:
            self.vec_mem = [torch.zeros_like(inputs)] * self.memory_size
        self.vec_mem.pop(0)
        self.vec_mem.append(self.mem)  # (8,b,c,h,w) type:list

    def calc_spike(self):
        # self.spike = self.act_fun(self.mem - self.threshold, torch.tensor(2.))
        # self.mem = self.mem * (1 - self.spike.detach())
        self.spike = torch.stack(self.vec_mem, dim=-1)  # (b,c,h,w,8)

    def n_reset(self):
        self.mem = 0.
        self.spike = 0.
        self.vec_mem = None


class weighted_mapping_1(nn.Module, HasInfoNCELoss):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 5, stride=1),
                                   nn.BatchNorm2d(16), nn.ReLU())
        self.pl1 = nn.AvgPool2d(2, stride=2)  # (12,12)
        self.fc1 = nn.Sequential(nn.Linear(12 * 12 * 16, 256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.flatten = nn.Flatten()
        self._project_x = nn.Sequential(nn.Flatten(), nn.Linear(784, 256),
                                        nn.ReLU(), nn.Linear(256, 64))
        self._project_latent = nn.Linear(256, 64)

    def forward(self, x):
        # x.shape = (b,1,28,28)
        # latent_dim = (b,256)
        temp_x = x
        x = self.conv1(x)
        x = self.pl1(x)
        x = self.flatten(x)
        latent = self.fc1(x)
        infonce_loss = self.compute_infonce_loss(temp_x, latent)
        score = self.fc2(latent)  # (b,1)
        return score, infonce_loss

    def compute_infonce_loss(self, x, latent):
        return self.batch_infonce_loss(x, latent)
