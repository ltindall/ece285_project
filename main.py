from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

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


# Prep GPU
GPU = torch.cuda.is_available()
print("GPU is {}enabled ".format(['not ', ''][GPU]))



########################################################################################################################
# Network Initialization
########################################################################################################################

# Specify Models
z_dim = 256
g_filter_size = 64
out_channels = 3
netG = Generator(z_dim, g_filter_size, out_channels)
netD = Patch_Discriminator()

if GPU: 
    netG = netG.cuda()
    netD = netD.cuda()

    
    
# Specify Loss
criterion = nn.BCELoss() # "Binary Cross Entropy", not "Before Common Era"

# Specify Optimizer

lr = 0.001
beta = 0.95 # momentum?

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta, 0.999))

# Specify data


dataset_dir = './datasets/monet2photo/trainB/'
root_dir = './datasets/monet2photo/trainB/'
output_dir = './gan_output/'

dataloader = Data(root_dir)

########################################################################################################################
# Training
########################################################################################################################

num_epochs = 100
batch_size = 10



real_label = 1
fake_label = 0
fixed_noise = Variable(torch.randn(batch_size, z_dim, 1, 1))
if GPU: 
    fixed_noise = fixed_noise.cuda()


for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        
        orig_img = data['img']
        img = np.moveaxis(data['img'], -1, 0)
        

        #### Training Discriminator Network ####

        # train with real training data

        netD.zero_grad()

        real_img = Variable(torch.FloatTensor(img))

        if GPU: 
            real_img = real_img.cuda()

        real_img.unsqueeze_(0)

        batch_size = real_img.shape[0]
        

        
        
        label = Variable(torch.FloatTensor(np.ones(batch_size,)*real_label))
        if GPU: 
            label = label.cuda()

        output = netD(real_img)

        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().data

        
        # train with generated data

        noise = Variable(torch.randn(batch_size, z_dim, 1, 1))
        if GPU: 
            noise = noise.cuda()
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().data
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

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, i, len(dataloader),
                 errD.data, errG.data, D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_img.data,
                              '%s/real_samples.png' % output_dir,
                              normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach().data,
                              '%s/fake_samples_epoch_%03d.png' % (output_dir, epoch),
                              normalize=True)

            # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (output_dir, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (output_dir, epoch))
