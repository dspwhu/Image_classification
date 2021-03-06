from torch.utils.data import Dataset
from PIL import Image
import os

def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Can't read image: {}".format(path))

# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class singleData(Dataset):
    def __init__(self, img_path, data_transforms=None, loader=default_loader):
        self.img_name = img_path
        self.data_transforms = data_transforms
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
#        img_name = self.img_name[item]
        img_name = self.img_name
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, -1
