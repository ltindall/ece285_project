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

from models import *
from data_loader import *



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_pretrained_models(modelG, modelD, output_dir):
    
    modelG.load_state_dict(torch.load(output_dir + 'netG_epoch_6.pth'))
    modelD.load_state_dict(torch.load(output_dir + 'netD_epoch_6.pth'))

    return modelG, modelD





# Prep GPU
GPU = torch.cuda.is_available()
print("GPU is {}enabled \n".format(['not ', ''][GPU]))

################################################################################
# Network Initialization
################################################################################

print('Initializing the Network...\n')

# Specify Models
z_dim = 512
g_filter_size = 64
out_channels = 3

netG = unet(3, 3, n_filters_start=128)
netD = Patch_Discriminator()

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

# Specify Optimizer

lr_G = 0.001
lr_D = 0.000005
beta = 0.5 # momentum?

optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta, 0.999))

# Specify data

root_dir = './monet2photo/trainB/'


print('Loading Dataset...\n')

# Specify Data Preprocessing

normalize = transforms.Compose([
     transforms.Scale(256),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])



# Fetch Data
# imagefolder = ImageFolder(root=root_dir)
# dataloader = DataLoader(imagefolder, batch_size=batch_size, shuffle=True)


batch_size = 4
dataloader = Data(root_dir, batch_size=batch_size)



################################################################################
# Training
################################################################################

num_epochs = 100

real_label = 0.90
fake_label = 0
# fixed_noise = Variable(torch.randn(batch_size, z_dim, 1, 1))

fixed_noise = Variable(torch.randn(1, 3, 256, 256))

if GPU: 
    fixed_noise = fixed_noise.cuda()


for epoch in range(num_epochs):
    for i in range(len(dataloader)):
        
        data = np.array(dataloader[i])
        data = np.moveaxis(data, 3, 1)


        #### Training Discriminator Network ####


        # train with real training data

        netD.zero_grad()

        # real_img = Variable(torch.FloatTensor(img))
        real_img = Variable(torch.FloatTensor(data))


        if GPU: 
            real_img = real_img.cuda()

        batch_size = real_img.shape[0]

        
        label = Variable(torch.FloatTensor(np.ones(batch_size,)*real_label))
       
        if GPU: 
            label = label.cuda()

        output = netD(real_img)
        errD_real = criterion(output, label)
        D_x = output.mean().data
        
        errD_real.backward() # hack for backpropagating only sometimes...

        # if np.abs(D_x - fake_label) > 0.3 and i > 1:
        #    errD_real.backward() # hack for backpropagating only sometimes...
        #    print('Backprop: D')


        # train with generated data

        # noise = Variable(torch.randn(batch_size, z_dim, 1, 1))
        noise = Variable(torch.randn(batch_size, 3, 256, 256))
        
        if GPU: 
            noise = noise.cuda()
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
         
        D_G_z1 = output.mean().data

        errD_fake.backward() # hack for backpropagating only sometimes...

        # if np.abs(D_G_z1- real_label/2) > 0.3 and i > 1:  
        #     errD_fake.backward() # hack for backpropagating only sometimes...
        #     print('Backprop: D')

        errD = errD_real + errD_fake
        optimizerD.step()

        #### Training Generator Network ####
        
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        
        errG.backward()

        D_G_z2 = output.mean().data
        optimizerG.step()

        if i % batch_size == 0:
            
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.data, errG.data, D_x, D_G_z1, D_G_z1))
            
        if i % 500 == 0:
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach().data,
                              '%s/fake_samples_epoch_%d_%d.png' % (output_dir, epoch, i),
                              normalize=True)

            # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (output_dir, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (output_dir, epoch))
