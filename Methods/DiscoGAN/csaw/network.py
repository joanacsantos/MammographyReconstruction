import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, extra_layers=False):
        super(Generator, self).__init__()

        if extra_layers == True: # For Car & Face DB
            self.main = nn.Sequential(
                # [-1, 1, 64x64] -> [-1, 64, 32x32]
                nn.Conv2d(1, 64, 4, 2, 1, bias=False),  
                nn.LeakyReLU(0.2, inplace=True),

                # [-1, 128, 16x16]
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                # [-1, 256, 8x8]
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),

                # [-1, 512, 4x4]
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),

                # [-1, 100, 1x1]
                nn.Conv2d(512, 100, 4, 1, 0, bias=False),
                nn.BatchNorm2d(100),
                nn.LeakyReLU(0.2, inplace=True),

                # [-1, 512, 4x4]
                nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),

                # [-1, 256, 8x8]
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),

                # [-1, 128, 16x16]
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),

                # [-1, 64, 32x32]
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                # [-1, 1, 64x64] -> Troquei o 3
                nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False), 
                nn.Tanh()  #Substituir por Sigmoid
            )

        if extra_layers == False: # For Edges/Shoes/Handbags and Facescrub
            self.main = nn.Sequential(
                # [-1, 1, 64x64] -> [-1, 64, 32x32]  -> Troquei o 3
                nn.Conv2d(1, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                # [-1, 128, 16x16]
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                # [-1, 256, 8x8]
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),

                # [-1, 512, 4x4]
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),

                # [-1, 256, 8x8]
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),

                # [-1, 128, 16x16]
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),

                # [-1, 256, 32x32]
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                # [-1, 1, 64x64]  -> Troquei o 3
                nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
                nn.Tanh() #voltar ao Tanh
            )

    def forward(self, input):
        return self.main( input )

class Discriminator(nn.Module):
    def __init__(self):

        super(Discriminator, self).__init__()

        # [-1, 1, 64x64] -> [-1, 64, 32x32] 
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1, bias=False)
        self.layer1 = nn.LeakyReLU(0.2, inplace=True)

        # -> [-1, 128, 16x16]
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = nn.LeakyReLU(0.2, inplace=True)

        # -> [-1, 256, 8x8]
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer3 = nn.LeakyReLU(0.2, inplace=True)

        # -> [-1, 512, 4x4]
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.layer4 = nn.LeakyReLU(0.2, inplace=True)

        # -> [-1, 1, 1x1]
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0, bias=False)

    def forward(self, input):
        layer1 = self.layer1( self.conv1( input ) )
        layer2 = self.layer2( self.bn2( self.conv2( layer1 ) ) )
        layer3 = self.layer3( self.bn3( self.conv3( layer2 ) ) )
        layer4 = self.layer4( self.bn4( self.conv4( layer3 ) ) )

        feature = [layer2, layer3, layer4]

        m = nn.Sigmoid()
        results_jo = m( self.conv5(layer4) ) 

        return results_jo , feature