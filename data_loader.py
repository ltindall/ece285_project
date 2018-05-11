import os
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from matplotlib import pyplot as plt
from torch import nn




class Data(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            file_list (list): List of string of ALL FILES in directory
        """
        self.root_dir = root_dir
        self.transform = transform

        # Go to root directory and create list of all filenames
       
        self.file_list = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        # Fetch file from directory
        filename = self.file_list[idx]

        # Images stored as Numpy Array (256, 256, 3)
        img = io.imread(self.root_dir+filename)

        # Store samples as dictionary {String filename, ndarray image}
        sample = {'filename': filename, 'img': img}

        # Perform transformation
        if self.transform:
            sample['img'] = self.transform(sample['img'])

        return sample

def imshow(sample):

    plt.title(sample['filename'])
    plt.imshow(sample['img'])
    plt.show()


class ConvGenerator(nn.Module):
    def __init__(self):
        super(ConvGenerator, self).__init__()

        self.encode = self._encode_layers()
        self.decode = self._decode_layers()

    def _encode_layers(self):
        """
        Generate encode layers for network
        :return: layers - list of layers describing network architecture

        Note: How to calculate output width after a convolution:
            Width = (Width - Kernel + 2*Padding) / Stride + 1

        """

        layers = []

        # Input: 256 x 256 x 3   Output: 128 x 128 x 16
        layers.append(torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.LeakyReLU(inplace=True))

        # Input: 128 x 128 x 16   Output: 64 x 64 x 32
        layers.append(torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.LeakyReLU(inplace=True))

        return torch.nn.Sequential(*layers)

    def _decode_layers(self):

        """
        Generate decode layers for network
        :return: layers - list of layers describing network architecture

        Note: How to calculate output width after a de-convolution:
            Width = Stride(Width - 1) + Kernel - 2*Padding

        """

        layers = []

        # Input: 64 x 64 x 32  Output: 128 x 128 x 16
        layers.append(torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.BatchNorm2d(32))
        layers.append(torch.nn.LeakyReLU(inplace=True))

        # Input: 128 x 128 x 16  Output: 256 x 256 x 3
        layers.append(torch.nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.BatchNorm2d(3))
        layers.append(torch.nn.LeakyReLU(inplace=True))

        return torch.nn.Sequential(*layers)


    def forward(self, inputs):
        x = inputs
        x = self.encode(x)
        x = self.decode(x)

        return x


if __name__ == "__main__": 

    G = ConvGenerator()


    dataset_dir = '/home/ckoguchi/Documents/ECE 285/Datasets/monet2photo/trainB'
    landscape_dataset = Data(dataset_dir)


    for i in range(len(landscape_dataset)):
        if i == 1:
            break

        sample = landscape_dataset[i]
        # y = G.forward(sample)

        print(sample['img'].shape)
        print(type(sample['img']))

        imshow(sample)












