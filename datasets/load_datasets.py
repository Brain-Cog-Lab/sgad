import torchvision
import global_v as glv
from torchvision import transforms
from torch.utils.data import DataLoader


def load_mnist_normalize(data_path):
    batch_size = glv.network_config['batch_size']
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])
    trainset = torchvision.datasets.MNIST(data_path,
                                          train=True,
                                          transform=transform,
                                          download=True)
    testset = torchvision.datasets.MNIST(data_path,
                                         train=False,
                                         transform=transform,
                                         download=True)
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2,
                             pin_memory=True)
    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)
    return trainloader, testloader


def load_mnist(data_path, is_normlized=False):
    batch_size = glv.network_config['batch_size']
    trans_list = [transforms.ToTensor()]
    if is_normlized:
        trans_list.append(transforms.Normalize((0.5, ), (0.5, )))
    transform = transforms.Compose(trans_list)
    trainset = torchvision.datasets.MNIST(data_path,
                                          train=True,
                                          transform=transform,
                                          download=True)
    testset = torchvision.datasets.MNIST(data_path,
                                         train=False,
                                         transform=transform,
                                         download=True)
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2,
                             pin_memory=True)
    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)
    return trainloader, testloader


def load_fmnist_normalize(data_path):
    batch_size = glv.network_config['batch_size']
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])
    trainset = torchvision.datasets.FashionMNIST(data_path,
                                                 train=True,
                                                 transform=transform,
                                                 download=True)
    testset = torchvision.datasets.FashionMNIST(data_path,
                                                train=False,
                                                transform=transform,
                                                download=True)
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2,
                             pin_memory=True)
    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)
    return trainloader, testloader


def load_fmnist(data_path, is_normlized=False):
    batch_size = glv.network_config['batch_size']
    trans_list = [transforms.ToTensor()]
    if is_normlized:
        trans_list.append(transforms.Normalize((0.5, ), (0.5, )))
    transform = transforms.Compose(trans_list)
    trainset = torchvision.datasets.FashionMNIST(data_path,
                                                 train=True,
                                                 transform=transform,
                                                 download=True)
    testset = torchvision.datasets.FashionMNIST(data_path,
                                                train=False,
                                                transform=transform,
                                                download=True)
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2,
                             pin_memory=True)
    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)
    return trainloader, testloader


def load_cifar10(data_path, is_normlized=False):
    batch_size = glv.network_config['batch_size']
    trans_list = [transforms.ToTensor()]
    if is_normlized:
        trans_list.append(transforms.Normalize((0.5, ), (0.5, )))
    transform = transforms.Compose(trans_list)
    trainset = torchvision.datasets.CIFAR10(data_path,
                                            train=True,
                                            transform=transform,
                                            download=True)
    testset = torchvision.datasets.CIFAR10(data_path,
                                           train=False,
                                           transform=transform,
                                           download=True)
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2,
                             pin_memory=True)
    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)
    return trainloader, testloader


def load_CelebA(data_path, is_normlized=False):
    batch_size = glv.network_config['batch_size']
    input_size = 64

    trans_list = [
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ]
    if is_normlized:
        trans_list.append(transforms.Lambda(lambda X: 2 * X - 1.))

    transform = transforms.Compose(trans_list)

    trainset = torchvision.datasets.CelebA(root=data_path,
                                           split='train',
                                           download=True,
                                           transform=transform)
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True)

    testset = torchvision.datasets.CelebA(root=data_path,
                                          split='test',
                                          download=True,
                                          transform=transform)
    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False)
    return trainloader, testloader
