
# coding: utf-8

# # Imports

# In[7]:


from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from  torch.utils.data import DataLoader 
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torchvision.datasets import ImageFolder
import sys
from torch.utils.data import Dataset
from skimage import io, transform
from matplotlib import pyplot as plt
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import datasets

from models import *
from data_loader import *


# # Utility Functions

# In[8]:


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_pretrained_models(modelG, modelD, output_dir):
    
    modelG.load_state_dict(torch.load(output_dir + 'bestG.pth'))
    modelD.load_state_dict(torch.load(output_dir + 'bestD.pth'))

    return modelG, modelD

def imshow(img):
    plt.imshow(img)
    plt.show()
    
class LandscapeDataset(Dataset):

    def __init__(self, root_dir, transforms=None):
        
        """
        Args:
            root_dir (string): Directory with all the images.
            file_list (list): List of strings of filename in batch
            img_list: List of images
        """

        self.root_dir = root_dir
        self.transforms = transforms
        

        # Go to root directory and create list of all filenames
        self.file_list = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]


    def __len__(self):
        return int(len(self.file_list))

    
    def __getitem__(self, idx):
  
        filename = self.file_list[idx]
        img = io.imread(self.root_dir+filename)

        if self.transforms is not None:
            img = self.transforms(img)


        return img

    
def imshow_batch(batch):
    
    
    plt.figure()
    for batch_num in range(5):
        plt.subplot(151 + batch_num)
        to_plot = np.moveaxis(batch[batch_num, :, :, :], 0, 2)
        plt.imshow(to_plot)
        plt.axis('off')
    


# # Prep GPU

# In[9]:


GPU = torch.cuda.is_available()
print("GPU is {}enabled \n".format(['not ', ''][GPU]))


# # Initialize Network

# In[10]:


print('Initializing the Network...\n')

# Specify Models
z_dim = 28
g_filter_size = 64
out_channels = 1

netG = Generator_128(100, 32, 3)
netD = Patch_Discriminator_128(input_channels=3)

warm_start = True
output_dir = './gan_output/'

if warm_start:
    netG, netD = load_pretrained_models(netG, netD, output_dir)


# netG = netG.apply(weights_init)
# netD = netD.apply(weights_init)

if GPU: 
    netG = netG.cuda()
    netD = netD.cuda()

criterion = nn.BCELoss() # "Binary Cross Entropy", not "Before Common Era"

# Specify Optimizer ############################################ LEARNING RATE HERE!

lr_G = 0.00005
lr_D = 0.00005

beta = 0.5 # momentum?

optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta, 0.999))

print('Done!')


# # Prep Data

# In[11]:


print('Prepping Data...')

batch_size = 64 ############################################ BATCH SIZE HERE!

num_epochs = 500

root_dir = './monet2photo/trainB/'

down_sample = 128

data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(down_sample),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5],
                             std=[.5, .5, .5])
    ])

#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])


dataset = LandscapeDataset(root_dir, transforms=data_transform)
train_loader = DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4
                         )





print('Done!')


# # Train

# In[ ]:


# real_label = np.random.uniform(0.7, 1.1)
# fake_label = np.random.uniform(0.0, 0.3)

real_label = 1
fake_label = 0

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        

        
        data = np.array(data)

        
        
        #### Training Discriminator Network ####

        # train with real training data

        netD.zero_grad()

        real_img = Variable(torch.FloatTensor(data))


        if GPU: 
            real_img = real_img.cuda()

        minibatch_size = real_img.shape[0]

        
        real_lbl = Variable(torch.FloatTensor(np.ones(minibatch_size,)*real_label))
       
        if GPU: 
            real_lbl = real_lbl.cuda()
        
        output = netD(real_img)
        
        errD_real = criterion(output, real_lbl)
        D_x = output.mean().data
        


        # train with generated data
        
        noise = Variable(torch.randn(minibatch_size, 100, 1, 1))
        fake_lbl = Variable(torch.FloatTensor(np.ones(minibatch_size,)*fake_label))
        
        if GPU: 
            noise = noise.cuda()
            fake_lbl = fake_lbl.cuda()
            
        fake = netG(noise)
        

        output = netD(fake.detach())
        errD_fake = criterion(output, fake_lbl)
         
        D_G_z1 = output.mean().data

        
        errD = errD_real + errD_fake

        errD.backward() ################################################### BACKPROP HERE
        optimizerD.step()

        
        
        #### Training Generator Network ####
        
        netG.zero_grad()
        
        noise = Variable(torch.randn(minibatch_size, 100, 1, 1))
        
        if GPU: 
            noise = noise.cuda()
        
        fake = netG(noise)
        output = netD(fake)
        errG = criterion(output, real_lbl)
        
        errG.backward() ################################################### BACKPROP HERE

        D_G_z2 = output.mean().data
        optimizerG.step()
        
        


        if i % 10 == 0:ss
            
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(train_loader),
                     errD.data, errG.data, D_x, D_G_z1, D_G_z2))
            
            fake = netG(noise)
            disp_img = np.moveaxis(fake.data.cpu().numpy()[0], 0, 2)
            
#             print(disp_img.shape)
            
            plt.imshow((disp_img + 1) / 2)
            plt.axis('off')
            plt.show()
            
#             vutils.save_image(fake.data.cpu().numpy()[0],
#                               '%s/fake_samples_epoch_%d_%d.png' % (output_dir, epoch, i),
#                               normalize=True)

            # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (output_dir, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (output_dir, epoch))

        

        




# In[31]:


# for i, data in enumerate(train_loader, 0):
#     data = np.array(data)
#    
#    plt.figure()
#    to_plot = np.moveaxis(data[i, :, :, :], 0, 2)
#    plt.imshow(to_plot)
#    plt.axis('off')
#
#    imshow_batch(data)
#    
#    if i == 5:
#        break


# In[ ]:





# In[ ]:




