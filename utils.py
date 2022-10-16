import os

from torchvision import datasets,transforms as T
from torch.utils.data import DataLoader

import config


def load_datasets(set_name='mnist', input_size=224, batch_size=16):
    if set_name == 'mnist':

        train_transform = T.Compose([
                    T.Resize((input_size,input_size)),
                    T.RandomHorizontalFlip(0.5),
                    T.ToTensor()
        ])

        test_transform = T.Compose([
                    T.Resize((input_size,input_size)),
                    T.ToTensor()
        ])

        train_dataset = datasets.MNIST(root=os.path.join(config.imagesets, 'MNIST'),
                                       train=True,
                                       transform=train_transform,
                                       download=True)
        test_dataset = datasets.MNIST(root=os.path.join(config.imagesets, 'MNIST'),
                                      train=False,
                                      transform=test_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
       
    else:
        return None, None

    return train_dataset, test_dataset, train_loader, test_loader
