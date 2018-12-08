import torch
import torch.nn as nn

import os
from importlib import import_module


class Model(nn.Module):
    def __init__(self, args, num_class):
        super(Model, self).__init__()
        print("Making model")
        self.args = args
        self.num_class = num_class
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(self.args, self.num_class).to(self.device)

        self.load(pre_train=args.pre_train)

    def forward(self, x):
        return self.model(x)

    def load(self, pre_train='.'):
        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.model.load_state_dict(torch.load(pre_train), strict=False)

    def save(self, apath, is_best=False):
        target = self.model
        torch.save(
            target.state_dict(),
            os.path.join(apath,'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_best.pt')
            )
