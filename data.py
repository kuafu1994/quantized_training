

import os
import torchvision.datasets as datasets

__DATASETS_DEFAULT_PATH = '' # configure the dataset


def get_dataset(name, split='train', transform=None, download=False,
                target_transform=None, dataset_path=__DATASETS_DEFAULT_PATH):

    root = os.path.join(dataset_path, name)

    if name.strip().lower() == 'imagenet':

        return datasets.ImageNet(root, split, download=download,
                                 transform=transform, target_transform=target_transform)





