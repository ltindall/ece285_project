# ECE 285 Spring 2018
# Blended Pineapple Juice

import torch.nn as nn
import torch.nn.functional as F
import torch

###########################################
# Weight Initializations
#
## Apply normal distribution initialization
## to convolutional and batch norm layers.
###########################################
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.zero_()
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()
############################
# End Weight Initializations
############################

#############################################
# Residual block for ResNet Generator Network
#############################################
class ResBlock(nn.Module):
    def __init__(self, input_channels):
        super(ResBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channels, input_channels, 3),
            nn.InstanceNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channels, input_channels, 3),
            nn.InstanceNorm2d(input_channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)
####################
# End Residual block
####################

##########################
# ResNet Generator Network
##########################
class ResNet_Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, res_blocks=9):
        super(ResNet_Generator, self).__init__()

        residual_blocks = []
        for i in range(res_blocks):
            residual_blocks.append(ResBlock(256))

        self.model = nn.Sequential(

            # initial block
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # downsample block 1
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            # downsample block 2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            # residual blocks
            *residual_blocks,

            # upsample block 1
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            # upsample block 2
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # output block
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, kernel_size=7),
            nn.Tanh()
        )


    def forward(self, img):
        return self.model(img)
##############################
# End ResNet Generator Network
##############################

##############################DDDDDD############
# Patch Discriminator network model for PatchGAN
####################################DDDDDD######
class Patch_Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Patch_Discriminator, self).__init__()

        def convolutional_block(input_filters, output_filters, normalize=True):
            layers = [nn.Conv2d(input_filters, output_filters, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(output_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *convolutional_block(input_filters=input_channels, output_filters=64, normalize=False),
            *convolutional_block(input_filters=64, output_filters=128),
            *convolutional_block(input_filters=128, output_filters=256),
            *convolutional_block(input_filters=256, output_filters=512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
##############################################
# End Discriminator network model for PatchGAN
##############################################
