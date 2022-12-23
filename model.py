from turtle import forward
import torch
from torch import nn
from torch import optim

class siamies(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=(3,1))
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3,1))
        self.act2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,1))
        self.act3 = nn.LeakyReLU()
        
        self.convT1 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3,1))
        self.act4 = nn.LeakyReLU()
        self.convT2 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(3,1))
        self.act5 = nn.LeakyReLU()
        self.convT3 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=(2,1), stride=(2,1))
        self.act6 = nn.LeakyReLU()


    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)

        out = self.convT1(out)
        out = self.act4(out)
        out = self.convT2(out)
        out = self.act5(out)
        out = self.convT3(out)
        out = self.act6(out)
        return out
