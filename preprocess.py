
import torch
import torchvision.transforms as transforms
import random

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

def scale_crop(input_size, scale_size=None, normalize=None):

    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize) # Here, assume that normalize is a dictionary
    ]

    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)

def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])


def get_transform(name='imagenet', input_size=None, 
    scale_size=None, normalize=None, augment=True):

    normalize = normalize or __imagenet_stats
    if name == 'imagenet':
        scale_size = scale_size or 256
        input_size = input_size or 224
        if augment:
            return inception_preproccess(input_size, normalize=normalize)
        else:
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)   

    