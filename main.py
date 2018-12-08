
import dataloader
import model
import utility
from option import args
from train import Trainer

checkpoint = utility.checkpoint(args)

loader = dataloader.DataLoader(args)
model = model.Model(args, loader.num_class)
t = Trainer(args, loader, model, checkpoint)
while not t.terminate():
    t.train()
    t.test()

print('Finished!')

