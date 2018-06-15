import sys
import os
import numpy as np
import math
import itertools
import datetime
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image


from models import *
from data_loader import *

######################################
# Hyperparameters
######################################

# control verbose output
verbose = True

# epoch to start training from
start_epoch = 142

# number of training epochs
n_epochs = 200

# dataset name
dataset_name = "monet2photo"

# batch size
batch_size = 1

# validation batch size
val_batch_size = 5

# learning rate
lr = 0.0002

# starting epoch for weight decay
decay_epoch = 100

# training image height
img_height= 256

# training image width
img_width = 256

# create sample every n batches
sample_val_batch = 100

# create model checkpoints every n epochs
checkpoint_epoch = 1

# patch size for Patch Discriminator
patch_size = (1, img_height // 2**4, img_width // 2**4)
######################################



print("Starting trainging for %s at epoch [%d/%d]\n" % (dataset_name, start_epoch, n_epochs))


# Make directory for generated images and model checkpoints
os.makedirs('generated_images/%s' % dataset_name, exist_ok=True)
os.makedirs('checkpoints/%s' % dataset_name, exist_ok=True)
print("Generated images will be saved in ./generated_images/%s" % dataset_name)
print("Model checkpoints will be saved in ./checkpoints/%s \n" % dataset_name)


#############################################
# Load datasets
#############################################
# Training set
train_dataloader = DataLoader(CycleGAN_Dataset("datasets/%s/trainA/" % dataset_name,"datasets/%s/trainB/" % dataset_name, img_height, img_width),batch_size=batch_size, num_workers=4)
# Validation set
val_dataloader = DataLoader(CycleGAN_Dataset("datasets/%s/testA/" % dataset_name, "datasets/%s/testB/" % dataset_name, img_height, img_width),batch_size=val_batch_size, num_workers=1)
#############################################

#############################################
# Loss functions and loss weights
#############################################
# Losses
loss_GAN = torch.nn.MSELoss()
loss_cycle = torch.nn.L1Loss()
loss_identity = torch.nn.L1Loss()

# Loss weights
lambda_cyc = 10
lambda_id = 0.5 * lambda_cyc
#############################################


###############################################
# Create Networks and initialize weights
###############################################
# Generator and Discriminator Networks
G_AB = ResNet_Generator(res_blocks=9)
G_BA = ResNet_Generator(res_blocks=9)
D_A = Patch_Discriminator()
D_B = Patch_Discriminator()

# Prep GPU
GPU = torch.cuda.is_available()
print("GPU is {}enabled \n".format(['not ', ''][GPU]))

if GPU:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()

if start_epoch != 0:
    G_AB.load_state_dict(torch.load('checkpoints/%s/G_AB_%d.pth' % (dataset_name, start_epoch)))
    G_BA.load_state_dict(torch.load('checkpoints/%s/G_BA_%d.pth' % (dataset_name, start_epoch)))
    D_A.load_state_dict(torch.load('checkpoints/%s/D_A_%d.pth' % (dataset_name, start_epoch)))
    D_B.load_state_dict(torch.load('checkpoints/%s/D_B_%d.pth' % (dataset_name, start_epoch)))
else:
    G_AB.apply(weights_init)
    G_BA.apply(weights_init)
    D_A.apply(weights_init)
    D_B.apply(weights_init)
###############################################


#################################################
# Pytorch optimizers and learning rate schedulers
#################################################
# Optimizers
G_params = itertools.chain(G_AB.parameters(), G_BA.parameters())
optimizer_G = torch.optim.Adam(G_params,lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))


# Learning rate schedulers
# begin to anneal the learning rate after decay_epoch epochs have passed
lambdaGAN = lambda epoch: 1.0 - max(0, epoch + start_epoch - decay_epoch) / (n_epochs - decay_epoch)
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambdaGAN)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambdaGAN)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambdaGAN)
#################################################


###################################
# Replay buffers
###################################
buffer_size=50
fake_A_buffer = []
fake_B_buffer = []

def get_buffer_batch(fake_buffer, fakes, buffer_size=50):

    mini_batch = []
    for f in fakes:
        if len(fake_buffer) < buffer_size:
            fake_buffer.append(f)
            mini_batch.append(f)
        else:
            if np.random.randint(0,2,1):
                i = np.random.randint(buffer_size)
                mini_batch.append(fake_buffer[i].clone())
                fake_buffer[i] = f
            else:
                mini_batch.append(f)

    return Variable(torch.cat(mini_batch)).view(-1,3, 256, 256)
###############################################################

#####################################
# Generate Validation images
#####################################
def generate_val_imgs(epoch_count, batch_count):
    val_imgs = next(iter(val_dataloader))

    val_real_A = Variable(torch.FloatTensor(val_imgs['A']))
    val_real_B = Variable(torch.FloatTensor(val_imgs['B']))
    if GPU:
        val_real_A = val_real_A.cuda()
        val_real_B = val_real_B.cuda()

    val_fake_A = G_BA(val_real_B.detach())
    val_fake_B = G_AB(val_real_A.detach())

    stacked_image = torch.cat((val_real_A.data, val_fake_B.data, val_real_B.data, val_fake_A.data), 0)
    save_image(stacked_image, 'generated_images/%s/%d_%d.png' % (dataset_name, epoch_count, batch_count), nrow=val_batch_size, normalize=True)
#####################################


################################################
# Training loop
################################################
for epoch in tqdm(range(start_epoch, n_epochs)):
    for i, AB_img_batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

        # Get batch of images from A and B set
        real_A = Variable(torch.FloatTensor(AB_img_batch['A']))
        real_B = Variable(torch.FloatTensor(AB_img_batch['B']))

        # Create matrix of real and fake target values for discriminator MSE loss
        real = Variable(torch.FloatTensor(np.ones((real_A.size(0), *patch_size))), requires_grad=False)
        fake = Variable(torch.FloatTensor(np.zeros((real_A.size(0), *patch_size))), requires_grad=False)

        # Send variables to gpu
        if GPU:
            real_A = real_A.cuda()
            real_B = real_B.cuda()
            real = real.cuda()
            fake = fake.cuda()

        ##############################################
        # Train the Generator Networks simultaneously
        ##############################################

        # zero out G Net gradients
        optimizer_G.zero_grad()

        # Adverserial (GAN) loss
        ## Generate AB fakes, send through D nets and compute MSE loss
        fake_A = G_BA(real_B)
        fake_B = G_AB(real_A)
        loss_GAN_AB = loss_GAN(D_B(fake_B), real)
        loss_GAN_BA = loss_GAN(D_A(fake_A), real)
        total_loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        ## Send generated fakes back through opposite G nets, compute L1 loss with original reals
        recov_A = G_BA(fake_B)
        recov_B = G_AB(fake_A)
        loss_cycle_A = loss_cycle(recov_A, real_A)
        loss_cycle_B = loss_cycle(recov_B, real_B)
        total_loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Identity loss
        ## Send real imgs through their own generator, compute L1 loss with original reals
        loss_id_A = loss_identity(G_BA(real_A), real_A)
        loss_id_B = loss_identity(G_AB(real_B), real_B)
        total_loss_identity = (loss_id_A + loss_id_B) / 2

        # Total GAN loss (weighted sum of adverserial, cycle, and identity losses)
        loss_G = total_loss_GAN + (lambda_cyc * total_loss_cycle) + (lambda_id * total_loss_identity)

        # Backpropagate through the generator networks
        loss_G.backward()
        optimizer_G.step()
        ####################################

        ####################################
        # Train the A Discriminator Network
        ####################################

        # zero out D_A gradients
        optimizer_D_A.zero_grad()

        # compute loss using real and fake A imgs sent through D_A
        loss_real = loss_GAN(D_A(real_A), real)
        fake_A_ = get_buffer_batch(fake_A_buffer, fake_A, buffer_size=50)
        loss_fake = loss_GAN(D_A(fake_A_.detach()), fake)
        loss_D_A = (loss_real + loss_fake) / 2

        # backpropagate D_A network
        loss_D_A.backward()
        optimizer_D_A.step()
        ####################################


        ####################################
        # Train the B Discriminator Network
        ####################################

        # zero out D_B gradients
        optimizer_D_B.zero_grad()

        # compute loss using real and fake B imgs sent through D_B
        loss_real = loss_GAN(D_B(real_B), real)
        fake_B_ = get_buffer_batch(fake_B_buffer, fake_B, buffer_size=50)
        loss_fake = loss_GAN(D_B(fake_B_.detach()), fake)
        loss_D_B = (loss_real + loss_fake) / 2

        # backpropagate D_B network
        loss_D_B.backward()
        optimizer_D_B.step()
        ##############################################


        # Total Discriminator loss from both networks
        loss_D = (loss_D_A + loss_D_B) / 2

        # Print the loss values every batch if verbose is enabled
        if verbose:
            description = "\r[D loss: {}] [G loss: {}, adverserial: {}, cycle: {}, identity: {}]".format(loss_D.item(), loss_G.item(), total_loss_GAN.item(), total_loss_cycle.item(), total_loss_identity.item())
            tqdm.write(description)

        # Periodicaly generate images from validation set
        if i % sample_val_batch == 0:
            generate_val_imgs(epoch, i)


    # Update the learning rates using annealing scheduler
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # save model files every epoch
    if epoch % checkpoint_epoch == 0:
        torch.save(G_AB.state_dict(), 'checkpoints/%s/G_AB_%d.pth' % (dataset_name, epoch))
        torch.save(G_BA.state_dict(), 'checkpoints/%s/G_BA_%d.pth' % (dataset_name, epoch))
        torch.save(D_A.state_dict(), 'checkpoints/%s/D_A_%d.pth' % (dataset_name, epoch))
        torch.save(D_B.state_dict(), 'checkpoints/%s/D_B_%d.pth' % (dataset_name, epoch))
