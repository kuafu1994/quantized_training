
import torch
import torchvision.transforms as transforms
import random


def scale_crop(input_size, scale_size=None, normalize=None):

    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize) # Here, assume that normalize is a dictionary
    ]

    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)