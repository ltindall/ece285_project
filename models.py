import torch
from torch import nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Patch_Discriminator_128(nn.Module):

    def __init__(self, input_channels=3):
        super(Patch_Discriminator_128, self).__init__()
        
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

        print(input_channels)
       
        self.model = nn.Sequential(
            *convolutional_block(input_filters=input_channels, output_filters=128, normalization=False),
            *convolutional_block(input_filters=128, output_filters=256),
            *convolutional_block(input_filters=256, output_filters=512),
            #*convolutional_block(input_filters=128, output_filters=256),
            #*convolutional_block(input_filters=256, output_filters=512),
            #*convolutional_block(input_filters=512, output_filters=1024),
            ##nn.ZeroPad2d((1, 0, 1, 0)),
            #nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            #nn.Sigmoid()
            nn.LeakyReLU(0.2, inplace=True), 
            Flatten(), 
            nn.Linear(1*13*13,1), 
            nn.Sigmoid()
        )



    def forward(self, input):
        #print(input.shape)
        output =  self.model(input)
        #print("output.shape = ",output.shape)
       
        #y = self.conv1(input)
        #y = self.conv2(y)
        #y = self.conv3(y)
        #y = self.r1(y)
        
        #print(y.shape)
        #y = self.lin1(y)
        #return y
        # (batch_size * out_channels * n * m)
        return output.squeeze() 
        #return output.view(-1, 1).squeeze(1)
        
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class Patch_Discriminator_mnist(nn.Module):

    def __init__(self, input_channels=3):
        super(Patch_Discriminator_mnist, self).__init__()
        
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

        print(input_channels)
       
        self.model = nn.Sequential(
            *convolutional_block(input_filters=input_channels, output_filters=128, normalization=False),
            *convolutional_block(input_filters=128, output_filters=256),
            #*convolutional_block(input_filters=64, output_filters=128),
            #*convolutional_block(input_filters=128, output_filters=256),
            #*convolutional_block(input_filters=256, output_filters=512),
            #*convolutional_block(input_filters=512, output_filters=1024),
            ##nn.ZeroPad2d((1, 0, 1, 0)),
            #nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            #nn.Sigmoid()
            nn.LeakyReLU(0.2, inplace=True), 
            Flatten(), 
            nn.Linear(1*13*13,1), 
            nn.Sigmoid()
        )
        '''
        self.conv1 =  convolutional_block(input_filters=input_channels, output_filters=32, normalization=False)
        self.conv2 =  convolutional_block(input_filters=32, output_filters=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)
        self.r1 = nn.LeakyReLU(0.2, inplace=True)
        self.lin1 = nn.Linear(16*1*4*4, 1)
        '''


    def forward(self, input):
        #print(input.shape)
        output =  self.model(input)
        #print("output.shape = ",output.shape)
       
        #y = self.conv1(input)
        #y = self.conv2(y)
        #y = self.conv3(y)
        #y = self.r1(y)
        
        #print(y.shape)
        #y = self.lin1(y)
        #return y
        # (batch_size * out_channels * n * m)
        return output.squeeze() 
        #return output.view(-1, 1).squeeze(1)
        
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)



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

        print(input_channels)
       
        self.model = nn.Sequential(
            *convolutional_block(input_filters=input_channels, output_filters=32, normalization=False),
            *convolutional_block(input_filters=32, output_filters=64),
            *convolutional_block(input_filters=64, output_filters=128),
            *convolutional_block(input_filters=128, output_filters=256),
            *convolutional_block(input_filters=256, output_filters=512),
            *convolutional_block(input_filters=512, output_filters=1024),
            #nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            #nn.Conv2d(in_channels=64, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            #nn.LeakyReLU(0.2, inplace=True), 
            #Flatten(), 
            #nn.Linear(1*4*4,1)
        )
        '''
        self.conv1 =  convolutional_block(input_filters=input_channels, output_filters=32, normalization=False)
        self.conv2 =  convolutional_block(input_filters=32, output_filters=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)
        self.r1 = nn.LeakyReLU(0.2, inplace=True)
        self.lin1 = nn.Linear(16*1*4*4, 1)
        '''


    def forward(self, input):
        #print(input.shape)
        output =  self.model(input)
        #print("forward output = ")
        #print(output.shape)
        #y = self.conv1(input)
        #y = self.conv2(y)
        #y = self.conv3(y)
        #y = self.r1(y)
        
        #print(y.shape)
        #y = self.lin1(y)
        #return y
        # (batch_size * out_channels * n * m)
        #return output.squeeze() 
        return output.view(-1, 1).squeeze(1)
    
class Generator_128(nn.Module):
    def __init__(self, z_size, g_filter_size, out_channels ):
        super(Generator_128, self).__init__()
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
            
            UpBlock(g_filter_size*4, g_filter_size*2, 4, 2,1),
            # size = (g_filter_size*32) x 64 x 64 
            
            nn.ConvTranspose2d(g_filter_size*2, out_channels, 4, 2, 1),
            nn.Tanh()
            # size = (g_filter_size*32) x 128 x 128
        )
        
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        
        return self.net(input)

class Generator_mnist(nn.Module):
    def __init__(self, z_size, g_filter_size, out_channels ):
        super(Generator_mnist, self).__init__()
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
            
            nn.ConvTranspose2d(g_filter_size*4, out_channels, 4, 2, 1),
            nn.Tanh()
            # size = (g_filter_size*32) x 64 x 64
        )
        
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        
        return self.net(input)

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
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):

        return self.up_block(x)
    
    
class conv_block(nn.Module):
    # conv -> batch norm -> relu -> conv -> batch norm -> relu
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class unet(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters_start=64 ):
        super(unet, self).__init__()
        #self.net = nn.Sequential( 
        #)
        
        self.conv1 = conv_block(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        
        self.conv2 = conv_block(64, 128)
        self.down2 = nn.MaxPool2d(2)
        
        self.conv3 = conv_block(128, 256)
        self.down3 = nn.MaxPool2d(2)
        
        self.conv4 = conv_block(256, 512)
        self.down4 = nn.MaxPool2d(2)
        
        self.conv5 = conv_block(512, 1024)
        
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        # concat conv4 output with up1 and send to conv6
        self.conv6 = conv_block(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = conv_block(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = conv_block(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = conv_block(128,64)
        
        self.conv10 = nn.Conv2d(64, n_classes, 1)
        

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
        '''

    def forward(self, input):
        
        x1 = input
        
        z1 = self.conv1(x1)
        
        x2 = self.down1(z1)
        z2 = self.conv2(x2)
        
        x3 = self.down2(z2)
        z3 = self.conv3(x3)
        
        x4 = self.down3(z3)
        z4 = self.conv4(x4)
        
        x5 = self.down4(z4)
        z5 = self.conv5(x5)
        
        
        x6 = torch.cat([z4,self.up1(z5)], dim=1)
        z6 = self.conv6(x6)
        
        x7 = torch.cat([z3, self.up2(z6)], dim=1)
        z7 = self.conv7(x7)
        
        x8 = torch.cat([z2, self.up3(z7)], dim=1)
        z8 = self.conv8(x8)
        
        x9 = torch.cat([z1, self.up4(z8)], dim=1)
        z9 = self.conv9(x9)
        
        z = self.conv10(z9)
        
        z = F.sigmoid(z)
        
        return z
    
    
    
    
class unet_mnist(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters_start=1 ):
        super(unet_mnist, self).__init__()
        #self.net = nn.Sequential( 
        #)
        self.conv_pre = conv_block(n_channels, 64*n_filters_start)
        self.conv1 = conv_block(64*n_filters_start, 64*n_filters_start)
        self.down1 = nn.MaxPool2d(2)
        
        self.conv2 = conv_block(64*n_filters_start, 128*n_filters_start)
        self.down2 = nn.MaxPool2d(2)
        
       
        
        self.conv3 = conv_block(128*n_filters_start,256*n_filters_start)
        self.conv_sp = conv_block(256*n_filters_start,256*n_filters_start)
        
        self.up1 = nn.ConvTranspose2d(256*n_filters_start, 128*n_filters_start, 2, stride=2)
        # concat conv4 output with up1 and send to conv6
        self.conv4 = conv_block( 256*n_filters_start, 128*n_filters_start)
        
        
        
        
        self.up2 = nn.ConvTranspose2d(128*n_filters_start, 64*n_filters_start, 2, stride=2)
        self.conv5 = conv_block(128*n_filters_start,64*n_filters_start)
        
        self.conv6 = nn.Conv2d(64*n_filters_start, n_classes, 1)
        

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
        '''

    def forward(self, input):
        
        x1 = input
        z0 = self.conv_pre(x1)
        z1 = self.conv1(z0)
        
        x2 = self.down1(z1)
        z2 = self.conv2(x2)
        
        x3 = self.down2(z2)
        z3 = self.conv3(x3)
        
        z_sp = self.conv_sp(z3)
        
        
        x4 = torch.cat([z2,self.up1(z_sp)], dim=1)
        z4 = self.conv4(x4)
        
   
        x5 = torch.cat([z1, self.up2(z4)], dim=1)
        z5 = self.conv5(x5)
        
        z = self.conv6(z5)
        
        z = F.sigmoid(z)
        
        return z
    
    

