import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.datasets

import os
import dataloader.customData as customData
import dataloader.singleData as singleData

class DataLoader():
    def __init__(self, args):
        self.args = args
        self.num_class = 0

        if not self.args.test_only:

            if self.args.dataset == 'cifar10':
                self.normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                      std=[0.2023, 0.1994, 0.2010])
                print(" Preparing CIFAR-10 dataset...")

                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    self.normalize
                ])

                self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                             transform=transform_train)

                transforms_test = transforms.Compose([
                    transforms.ToTensor(),
                    self.normalize
                ])

                self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                                            transform=transforms_test)
                self.num_class = 10

            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True,
                                                           num_workers=self.args.n_threads, pin_memory=True, sampler=None)
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False,
                                                          num_workers=self.args.n_threads, pin_memory=True)
        else:
            self.num_class = 10
            root = self.args.data_test
            self.normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                  std=[0.2023, 0.2565, 0.2761])
            print(" Preparing My single image...")

            transforms_test = transforms.Compose([
                transforms.Resize(36),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                self.normalize
            ])
            self.trainloader = None
            self.testset = singleData.singleData(img_path=root, data_transforms=transforms_test)
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=1, shuffle=False, pin_memory=True)




