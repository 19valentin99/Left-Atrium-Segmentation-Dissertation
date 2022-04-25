#**************************************************
#Author:  Valentin Craciun                        *
#Project: Heart Segmentation using Deep Learning  *
#Date:    08.04.2022                              *
#**************************************************

########################################################################################################################################################
# Credits to:                                                                                                                                          #
# original paper: https://arxiv.org/abs/1505.04597                                                                                                     #
# github repository: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet#
########################################################################################################################################################

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# default class for double cnn relation
class Double_cnn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Double_cnn,self).__init__()

        self.sequence= nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.sequence(x)


class UNET(nn.Module):                                                               # in channels: how many channels does the image have
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]): # out channels 1: binary segmentation
        super(UNET,self).__init__()                                                  
        self.downs = nn.ModuleList()                
        self.ups   = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down connections UNET
        for feature in features:
            self.downs.append(Double_cnn(in_channels, feature))   # in channels => mapped to out channels(features) ex 1 to 64
            in_channels = feature

        # Rising connections UNET
        for feature in reversed(features):                        # structure: UP -> 2 convs || UP -> 2 convs and so on...
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2   # *2 because at the bottom will add a layer which duplicates the in features
                                                                  # by setting kernel size to 2 and stride 2, it will double the size of the image
                )
            )
            self.ups.append(Double_cnn(feature*2, feature))

        # Bottom connection (bottle neck layer)
        self.bottleneck = Double_cnn(features[-1], features[-1]*2) # features[-1] because we want the last element in the list, ie 512

        # Final conv layer that changes the nr of channels and performs segmentation
        self.final_conf = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []           # are the horizontal arrows

        for down in self.downs:         # down convolutions
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]   # skipping connections

        for idx in range(0, len(self.ups), 2):      # up convolutions
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x =TF.resize(x, size=skip_connection.shape[2:])

            concatenate_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concatenate_skip)

        return self.final_conf(x)   # final restult
