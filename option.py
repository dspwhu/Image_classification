import argparse

parser = argparse.ArgumentParser(description='Image Classification ')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=16,
                    help='number of threads for data loading')

parser.add_argument('--cpu', action='store_true', default= False,
                    help='use cpu only')

# main parser
parser.add_argument('--model', default='ResNet',
                    choices=('LeNet', 'ResNet'),
                    help='choose a classification model')

parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')

parser.add_argument('--dataset', type=str, default='cifar10',
                    help='train dataset, cifar10')

parser.add_argument('--save', type=str, default='./test/',
                    help='file name to save')


# only for test images
parser.add_argument('--test_only', '-t', action= 'store_true', default= False, help= 'Test your picture with the saved model')

parser.add_argument('--data_test', type=str, default='.',
                    help='test dataset location')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre_train model location')


# Training specifications
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-1,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=60,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='SGD',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay')

args = parser.parse_args()
