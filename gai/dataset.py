import torch
import os

from beam import beam_device, resource
from beam import UniversalDataset
from beam.data import BeamData

import torchvision
import kornia
from kornia.augmentation.container import AugmentationSequential


class CIFAR10Dataset(UniversalDataset):

    def __init__(self, hparams):
        super().__init__()

        path = resource(hparams.data_path)
        device = beam_device(hparams.device)
        padding = hparams.padding

        self.augmentations = AugmentationSequential(kornia.augmentation.RandomHorizontalFlip(),
                                                    kornia.augmentation.RandomCrop((32, 32), padding=padding,
                                                                                   padding_mode='reflect'))

        self.mu = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view(1, -1, 1, 1).to(device)
        self.sigma = torch.FloatTensor([0.247, 0.243, 0.261]).view(1, -1, 1, 1).to(device)

        # self.t_basic = transforms.Compose([transforms.Lambda(lambda x: (x.half() / 255)),
        #                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        file = path.joinpath('dataset_uint8.pt')
        if file.exists():
            x_train, x_test, y_train, y_test = file.read(map_location=device)

        else:
            dataset_train = torchvision.datasets.CIFAR10(root=path, train=True,
                                                         transform=torchvision.transforms.PILToTensor(), download=True)
            dataset_test = torchvision.datasets.CIFAR10(root=path, train=False,
                                                        transform=torchvision.transforms.PILToTensor(), download=True)

            x_train = torch.stack([dataset_train[i][0] for i in range(len(dataset_train))]).to(device)
            x_test = torch.stack([dataset_test[i][0] for i in range(len(dataset_test))]).to(device)

            y_train = torch.LongTensor(dataset_train.targets).to(device)
            y_test = torch.LongTensor(dataset_test.targets).to(device)

            file.write((x_train, x_test, y_train, y_test))

        self.data = BeamData.simple({'train': x_train, 'test': x_test}, label={'train': y_train, 'test': y_test})
        self.labels = self.data.label
        self.split(validation=.2, test=self.data['test'].index, seed=hparams.split_dataset_seed)
        
    def getitem(self, ind):

        data = self.data[ind]
        x = data.data
        labels = data.label

        x = x.half() / 255

        if self.training:
            x = self.augmentations(x)

        x = (x.float() - self.mu) / self.sigma
        x = x.to(memory_format=torch.channels_last)

        return {'x': x, 'y': labels}