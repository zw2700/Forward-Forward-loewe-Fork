import os
import random
from datetime import timedelta

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy

from hydra.utils import get_original_cwd

from src import ff_mnist, ff_model, ff_cifar10


def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    return opt


def get_model_and_optimizer(opt):
    model = ff_model.FF_model(opt)
    if "cuda" in opt.device:
        model = model.cuda()
    print(model, "\n")

    # Create optimizer with different hyper-parameters for the main model
    # and the downstream classification model.
    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in model.linear_classifier.parameters())
    ]
    optimizer = torch.optim.SGD(
        [
            {
                "params": main_model_params,
                "lr": opt.training.learning_rate,
                "weight_decay": opt.training.weight_decay,
                "momentum": opt.training.momentum,
            },
            {
                "params": model.linear_classifier.parameters(),
                "lr": opt.training.downstream_learning_rate,
                "weight_decay": opt.training.downstream_weight_decay,
                "momentum": opt.training.momentum,
            },
        ]
    )
    return model, optimizer


def get_data(opt, partition):
    if opt.input.dataset == "MNIST":
        dataset = ff_mnist.FF_MNIST(opt, partition)
    elif opt.input.dataset == "CIFAR10":
        dataset = ff_cifar10.FF_CIFAR10(opt, partition)

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
        persistent_workers=True,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_MNIST_partition(opt, partition):
    if partition == "train":
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    elif partition == "val":
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    else:
        raise NotImplementedError    
    return mnist

def get_CIFAR10_partition(opt, partition):
    # data augmentation & transformation
    train_transform = transforms.Compose([
            transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
            transforms.ToImageTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            Cutout(n_holes=1, length=16)
        ])
    test_transform = transforms.Compose([
            transforms.ToImageTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284)),
        ])
    
    if partition == "train":
        cifar10 = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=train_transform,
        )
    
    elif partition in ["val"]:
        cifar10 = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=test_transform,
        )
    
    else:
        raise NotImplementedError
    
    return cifar10

class Cutout(object):
    '''Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    '''
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        '''
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        '''
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def dict_to_cuda(dict):
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict


def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels


def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.learning_rate
    )
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.downstream_learning_rate
    )
    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size


def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict
