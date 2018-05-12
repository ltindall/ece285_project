import torch
from torch import nn


class Patch_Discriminator(nn.Module):

    def __init__(self, input_channels=3):
        super(Patch_Discriminator, self).__init__()

        def convolutional_block(input_filters, output_filters, normalization=True):
            """

            :param input_filters: input number channels
            :param output_filters: output number channels
            :param normalization: Batch Normalization (1st layer has no normalization in paper)
            :return: list of convolution-blocks (conv, batch-norm, leakyReLU)

            """

            # Output Image Size:  W := (W - K + 2P)/S + 1

            layers = [nn.Conv2d(input_filters, output_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(output_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers


        self.model = nn.Sequential(
            *convolutional_block(input_filters=input_channels, output_filters=32, normalization=False),
            *convolutional_block(input_filters=32, output_filters=64),
            *convolutional_block(input_filters=64, output_filters=128),
            *convolutional_block(input_filters=128, output_filters=256),
            *convolutional_block(input_filters=256, output_filters=512),
            *convolutional_block(input_filters=512, output_filters=1024),
            #nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )



    def forward(self, input):
        output =  self.model(input)
        return output.view(-1, 1).squeeze(1)



class Generator(nn.Module):
    def __init__(self, z_size, g_filter_size, out_channels ):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # input is Z, going into a convolution
            
            UpBlock(z_size, g_filter_size*32, 4, 1, 0),
            # size = (g_filter_size*32) x 4 x 4
            
            UpBlock(g_filter_size*32, g_filter_size*16, 4, 2, 1),
            # size = (g_filter_size*16) x 8 x 8
            
            UpBlock(g_filter_size*16, g_filter_size*8, 4, 2, 1),
            # size = (g_filter_size*32) x 16 x 16
            
            UpBlock(g_filter_size*8, g_filter_size*4, 4, 2, 1),
            # size = (g_filter_size*32) x 32 x 32
            
            UpBlock(g_filter_size*4, g_filter_size*2, 4, 2, 1), 
            # size = (g_filter_size*32) x 64 x 64
            
            UpBlock(g_filter_size*2, g_filter_size, 4, 2, 1),
            # size = (g_filter_size*32) x 128 x 128
            
            nn.ConvTranspose2d(g_filter_size, out_channels, 4, 2, 1),
            nn.Tanh()
            # size = (g_filter_size*32) x 256 x 256
        )

    def forward(self, input):
        
        return self.net(input)
    
class UpBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UpBlock, self).__init__()
        
        self.up_block = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):

        return self.up_block(x)

