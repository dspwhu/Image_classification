
import utility
import torch.nn as nn
import torch
import sys

class Trainer():
    def __init__(self, args, loader, model, ckp):
        self.args = args
        self.loader_train = loader.trainloader
        self.loader_test = loader.testloader
        self.model = model
        self.ckp = ckp

        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        self.loss = nn.CrossEntropyLoss()
        self.best_acc = 0

        utility.print_network(self.model)

    def train(self):
        total = 0
        correct = 0
        self.scheduler.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.model.train()

        print('\n => Training Epoch #%d, LR=%.4f' %(epoch, lr))
        for batch, (inputs, labels) in enumerate(self.loader_train):
            inputs, labels = self.prepare([inputs, labels])
            self.optimizer.zero_grad()
            result = self.model(inputs)
            loss = self.loss(result, labels)
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(result.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum().item()

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t Loss: %f   ACC@1: %.3f%%'
                             %(epoch, self.args.epochs, batch+1, len(self.loader_train),
                              loss.item(), 100*correct / total))
            sys.stdout.flush()

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        total = 0
        correct = 0

        no_eval = False
        self.model.eval()
        with torch.no_grad():
            for batch, (inputs, labels) in enumerate(self.loader_test):
                no_eval = (labels.nelement() == 1)
                if no_eval:
                    inputs = self.prepare(inputs)[0]
                    inputs = torch.unsqueeze(inputs, 0)
                else:
                    inputs, labels = self.prepare([inputs, labels])

                results = self.model(inputs)

                if not no_eval:
                    _, predicted = torch.max(results.data, 1)
                    total += labels.shape[0]
                    correct += predicted.eq(labels.data).cpu().sum().item()
                else:
                    key= {0:'airplane', 1:'automobile', 2: 'bird', 3:'cat', 4:'deer', 5:'dog',
                          6:'frog', 7:'horse', 8:'ship', 9:'truck'}
                    _, predicted = torch.max(results.data, 1)
                    print("\n| Test your image's label :  #%s\t" % key[predicted.item()])
                    break

            if not no_eval:
                acc = 100*correct/total
                print("\n| Validation Epoch #%d\t\t\t\t\t\t\tACC@1: %.2f%%" %(epoch, acc))
                self.ckp.save(self, is_best=(acc > self.best_acc))
                if acc > self.best_acc:
                    self.best_acc = acc

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
