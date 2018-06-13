import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CycleGAN_Dataset(Dataset):
    def __init__(self, A_dir, B_dir, img_height, img_width):

        self.transform = transforms.Compose([
            transforms.RandomCrop((img_height, img_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])

        self.A_list = [os.path.join(A_dir,f) for f in os.listdir(A_dir) if os.path.isfile(os.path.join(A_dir, f))]
        self.A_list.sort()
        self.A_length = len(self.A_list)
        self.B_list = [os.path.join(B_dir,f) for f in os.listdir(B_dir) if os.path.isfile(os.path.join(B_dir, f))]
        self.B_list.sort()
        self.B_length = len(self.B_list)


    def __len__(self):
        return max(self.A_length, self.B_length)

    def __getitem__(self, idx):

        imgA = self.transform(Image.open(self.A_list[np.random.randint(self.A_length)]))
        imgB = self.transform(Image.open(self.B_list[np.random.randint(self.B_length)]))

        return {'A': imgA, 'B': imgB}








