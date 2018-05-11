from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models import *
from data_loader import *





########################################################################################################################
# Network Initialization
########################################################################################################################

# Specify Models
netG = Generator()
netD = Patch_Discriminator()

# Specify Loss
criterion = nn.BCELoss() # "Binary Cross Entropy", not "Before Common Era"

# Specify Optimizer

lr = 0.001
beta = 0.95 # momentum?

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta, 0.999))

# Specify data

root_dir = ''
output_dir = ''

dataloader = Data(root_dir)

########################################################################################################################
# Training
########################################################################################################################

num_epochs = 100
batch_size = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
z_dim = 256


real_label = 1
fake_label = 0
fixed_noise = torch.randn(batch_size, z_dim, 1, 1, device=device)


for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):

        #### Training Discriminator Network ####

        # train with real training data

        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with generated data

        noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        #### Training Generator Network ####

        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                              '%s/real_samples.png' % output_dir,
                              normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                              '%s/fake_samples_epoch_%03d.png' % (output_dir, epoch),
                              normalize=True)

            # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (output_dir, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (output_dir, epoch))


































