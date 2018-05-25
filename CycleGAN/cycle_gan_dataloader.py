import os
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys



class CycleGAN_Dataset(Dataset):
    def __init__(self, monet2photo_dir, dataset='Monet', transform=None):

        if dataset == 'Monet':
            self.dataset_dir = monet2photo_dir + '/monet2photo/trainA/'

        elif dataset == 'Photo':
            self.dataset_dir = monet2photo_dir + '/monet2photo/trainB/'

        else:
            print('Invalid dataset option.  Please input: \'Monet\' or \'Photo\'')


        self.monet2photo_dir = monet2photo_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(self.dataset_dir) if os.path.isfile(os.path.join(self.dataset_dir, f))]

    def __len__(self):
        return int(len(self.file_list))

    def __getitem__(self, idx):

        filename = self.file_list[idx]
        img = io.imread(self.dataset_dir + filename)

        if self.transform is not None:
            img = self.transforms(img)

        return img



# transform = transforms.Compose([
#                     transforms.toPILImage(),
#                     transforms.RandomResizedCrop(128),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


transform = None



monet2photo_dir = '/home/ckoguchi/Documents/ECE_285/Datasets'

MonetDataset = CycleGAN_Dataset(monet2photo_dir, dataset='Monet', transform=transform)
PhotoDataset = CycleGAN_Dataset(monet2photo_dir, dataset='Photo', transform=transform)

batch_size = 128
max_epochs = 1

MonetDataLoader = DataLoader(MonetDataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)

PhotoDataLoader = DataLoader(PhotoDataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)

for epoch in range(max_epochs):
    for i, data in enumerate(MonetDataLoader, 0):

        if i == 1:
            break
        data = np.array(data)
        print(data.shape)
























