import os

import torch.optim as optim
import torch.optim.lr_scheduler as lrs



class checkpoint():
    def __init__(self, args):
        self.args = args
        self.dir = args.save

        def _make_dir(path):
            if not os.path.exists(path): os.mkdir(path)

        _make_dir(self.dir)

    def save(self, trainer, is_best=False):
        trainer.model.save(self.dir, is_best)


def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    print(net)

    print('Total number of parameters: %d' % num_params)