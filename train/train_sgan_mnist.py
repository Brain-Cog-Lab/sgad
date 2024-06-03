import sys
import os

sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

import torch
from torch import nn
import global_v as glv
from datasets import load_datasets
from models.generators import *
from models.discriminators import *
from network_parser import Parse
import torchvision
from torchvision.utils import save_image
import argparse
import logging
from tqdm import tqdm, trange
import math


def reset_net(net: nn.Module):
    for m in net.modules():
        if hasattr(m, 'n_reset'):
            m.n_reset()


def update_D(X, Z, net_D, net_G, loss, trainer_D):
    acc_num = 0
    # X.shape = (batch_size, 784)
    batch_size = X.shape[0]
    ones = torch.ones((batch_size, ), device=X.device)
    # print(ones.shape)
    zeros = torch.zeros((batch_size, ), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X, is_imgs=True)  # real_Y.shape = (n_steps, batch_size, 1)
    # print(real_Y.shape)
    if not glv.network_config['is_mem']:
        real_Y = torch.sum(real_Y, dim=0) / glv.network_config[
            'n_steps']  # real_Y.shape = (batch_size,1)
    with torch.no_grad():
        acc_num += torch.sum((real_Y > 0.8).float()).item()
    # print(real_Y.shape)
    fake_X = net_G(Z)  # fake_X.shape = (n_steps, batch_size, 784)
    n_step = fake_X.shape[0]
    if glv.network_config['net_D_direct_input']:
        fake_X = fake_X.reshape((n_step, batch_size, 1, 28, 28))
    else:
        fake_X = fake_X.reshape((n_step, batch_size, 784))
    fake_Y = net_D(fake_X.detach())  # fake_Y.shape = (n_steps, batch_size, 1)
    if not glv.network_config['is_mem']:
        fake_Y = torch.sum(fake_Y, dim=0) / glv.network_config[
            'n_steps']  # fake_Y.shape = (batch_size,1)
    with torch.no_grad():
        acc_num += torch.sum((fake_Y < 0.2).float()).item()
    fake_mean = fake_Y.mean().item()
    real_mean = real_Y.mean().item()
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) +
              loss(fake_Y, zeros.reshape(
                  fake_Y.shape))) / 2  # let real be 1 and fake be 0
    loss_D.backward()
    trainer_D.step()
    return loss_D, acc_num, fake_mean, real_mean


def update_G(Z, net_D, net_G, loss, trainer_G):
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size, ), device=Z.device)
    trainer_G.zero_grad()
    fake_X = net_G(Z)  # fake_X.shape = (n_steps, batch_size, 784)
    n_step = fake_X.shape[0]
    if glv.network_config['net_D_direct_input']:
        fake_X = fake_X.reshape((n_step, batch_size, 1, 28, 28))
    else:
        fake_X = fake_X.reshape((n_step, batch_size, 784))
    fake_Y = net_D(fake_X)  # shape = (n_steps, batch_size, 1)
    if not glv.network_config['is_mem']:
        fake_Y = torch.sum(
            fake_Y,
            dim=0) / glv.network_config['n_steps']  # shape = (batch_size, 1)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    '''parser.add_argument("--name", required=True, dest='name', type=str)
    parser.add_argument("--exp_index",
                        required=True,
                        dest='exp_index',
                        type=str)'''
    parser.add_argument("--config", required=True, dest='config', type=str)
    args = parser.parse_args()

    config = args.config
    params = Parse(config)
    glv.init(params['Network'])

    data_path = glv.network_config['data_path']
    device = glv.network_config['device']
    name = glv.network_config['name']
    dataset_name = glv.network_config['dataset']
    latent_dim = glv.network_config['latent_dim']
    lr_D = glv.network_config['lr_D']
    lr_G = glv.network_config['lr_G']
    # torch.autograd.set_detect_anomaly(True)

    os.makedirs(f'./exp_results/checkpoints/{name}', exist_ok=True)
    os.makedirs(f'./exp_results/images/{name}', exist_ok=True)
    logging.basicConfig(filename=f'./exp_results/logs/{name}.log',
                        level=logging.INFO)

    # load dataset
    print("loading dataset")
    if dataset_name == 'MNIST':
        trainloader, _ = load_datasets.load_mnist(
            data_path, is_normlized=glv.network_config['is_data_normlized'])
    elif dataset_name == "CelebA":
        trainloader, _ = load_datasets.load_CelebA(
            data_path, is_normlized=glv.network_config['is_data_normlized'])

    # load model
    net_G, net_D = None, None
    if dataset_name == "MNIST":
        net_G = Generator_SNN(input_dim=latent_dim)
        net_D = Discriminator_MP()

    # set optimizer
    optimizer_G = torch.optim.RMSprop(net_G.parameters(), lr=lr_G)
    optimizer_D = torch.optim.RMSprop(net_D.parameters(), lr=lr_D)

    # loss
    loss = nn.BCELoss(reduction='sum')

    # to device
    net_G = net_G.to(device)
    net_D = net_D.to(device)

    init_epoch = 0
    if glv.network_config['from_checkpoint']:
        print("loading checkpoint")
        checkpoint = torch.load(glv.network_config['checkpoint_path'])
        init_epoch = checkpoint['epoch']
        net_D.load_state_dict(checkpoint['model_state_dict_D'])
        net_G.load_state_dict(checkpoint['model_state_dict_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_state_dict_D'])
        optimizer_G.load_state_dict(checkpoint['optimizer_state_dict_G'])

    # load scheduler
    scheduler_G = None
    scheduler_D = None
    if glv.network_config["is_scheduler"]:
        scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D,
                                                                 T_max=20)
        scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G,
                                                                 T_max=20)

    logging.info(glv.network_config)

    print("start training")
    # batch_size = glv.network_config['batch_size']
    img_size = next(iter(trainloader))[0].shape[-1]
    channels = next(iter(trainloader))[0].shape[1]
    for epoch in range(init_epoch, glv.network_config['epochs']):
        loss_D = 0
        loss_G = 0
        total_acc_num = 0
        total_num = 0
        fake_mean = 0
        real_mean = 0
        batch_count = 0
        total_grad = 0
        for X, _ in tqdm(trainloader, colour='blue'):
            batch_count += 1
            batch_size = X.shape[0]
            if not glv.network_config['net_D_direct_input']:
                X = X.reshape((batch_size, -1))
            X = X.to(device)
            Z = torch.randn((batch_size, 100), device=device)
            D_increment, acc_num_increment, mean_increment_fake, mean_increment_real = update_D(
                X, Z, net_D, net_G, loss, optimizer_D)
            loss_D += D_increment
            total_acc_num += acc_num_increment
            fake_mean += mean_increment_fake
            real_mean += mean_increment_real
            reset_net(net_D)
            reset_net(net_G)
            loss_G += update_G(Z, net_D, net_G, loss, optimizer_G)

            reset_net(net_D)
            reset_net(net_G)
            total_num += batch_size

        if glv.network_config['is_scheduler']:
            scheduler_D.step()
            scheduler_G.step()

        logging.info(f'Epoch: {epoch}'
                     f'fake_credit:{fake_mean / batch_count},'
                     f'real_credit:{real_mean / batch_count}')
        print(f'Epoch: {epoch}'
              f'fake_credit:{fake_mean / batch_count},'
              f'real_credit:{real_mean / batch_count}')
