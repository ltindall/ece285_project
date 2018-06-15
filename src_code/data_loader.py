import os
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, transform
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



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


class Data(Dataset):

    def __init__(self, root_dir, batch_size = 128):
        
        """
        Args:
            root_dir (string): Directory with all the images.
            file_list (list): List of strings of filename in batch
            img_list: List of images
        """

        self.root_dir = root_dir
        self.batch_size = batch_size

        normalize = transforms.Compose([
             transforms.Scale(256),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             ])

        self.transform = normalize

        # Go to root directory and create list of all filenames
        self.file_list = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]


    def __len__(self):
        return int(len(self.file_list)/self.batch_size)

    def __getitem__(self, idx):

        # TODO: Get Transforms working: ERROR Related PIL and numpy transform...

        # Fetch file from directory
        start = idx*self.batch_size
        end = (idx + 1)*self.batch_size
        
        
        # print('\tFetching batch: ', idx)

        filename = self.file_list[idx*self.batch_size:(idx + 1)*self.batch_size]
        
        # print(len(filename))

        # Images stored as Numpy Array (256, 256, 3)
        img_list = [io.imread(self.root_dir+f) for f in filename]
        # img = io.imread(self.root_dir+filename)

        # Store samples as dictionary {String filename, ndarray image}
        # sample = {'filename': filename, 'img': img}

        # Perform transformatation
        # img_list = [self.transform(img) for img in img_list]
        
        # sample['img'] = self.transform(sample['img'])

        # can return filename too!

        return img_list

def imshow(sample):

    plt.title(sample['filename'])
    plt.imshow(sample['img'])
    plt.show()










