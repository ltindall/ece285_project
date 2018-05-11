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
            *convolutional_block(input_filters=input_channels*2, output_filters=64, normalization=False),
            *convolutional_block(input_filters=64, output_filters=128),
            *convolutional_block(input_filters=128, output_filters=256),
            *convolutional_block(input_filters=256, output_filters=512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1, bias=False),
            nn.Sigmoid()
        )



    def forward(self, input):
        output =  self.model(input)
        return output.view(-1, 1).squeeze(1)