import torch
from torch import nn
from torch.nn import functional as F

class FluidalResidualBlock(nn.Module):

    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, 
                stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(),
            nn.Conv3d(mid_channels, in_channels, 3, 
                stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm3d(in_channels)
        )

    def forward(self, x):
        out = self.conv_layers(x)        
        out = out + x
        return F.relu(out)

class EncodingStartBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups= 4, bias= False):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, 
                stride=1, padding=1, dilation=1, groups=groups, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, 3, 
                stride=1, padding=1, dilation=1, groups=groups, bias=False),
            nn.BatchNorm3d(in_channels)
        )
        self.squeeze_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, 1, 
                stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        out = self.conv_layers(x)
        out = out+x # residual connection
        out = self.squeeze_layer(out)
        return out

class DownSampleBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, bias= False):
        super().__init__()
        self.down_sample_layers = nn.Sequential(
            # FluidalResidualBlock(in_channels),
            nn.Conv3d(in_channels, out_channels, 1, 
                stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.MaxPool3d(2,2)
        )
        
    def forward(self, x):
        return self.down_sample_layers(x)

class UpSampleDecodeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bias= False):
        super().__init__()
        self.down_sample_layers = nn.Sequential(
            FluidalResidualBlock(in_channels, in_channels//2),
            nn.Conv3d(in_channels, out_channels, 1, 
                stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.down_sample_layers(x)

class LightFluidalResidualBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, 
                stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv_layers(x)        
        out = out + x
        return F.relu(out)

class LightEncodingStartBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups= 4, bias= False):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, 
                stride=1, padding=1, dilation=1, groups=groups, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU()
        )
        self.squeeze_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, 1, 
                stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        out = self.conv_layers(x)
        out = out+x # residual connection
        out = self.squeeze_layer(out)
        return out

class LightDownSampleBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, bias= False):
        super().__init__()
        self.down_sample_layers = nn.Sequential(
            # FluidalResidualBlock(in_channels),
            nn.Conv3d(in_channels, out_channels, 1, 
                stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.MaxPool3d(2,2)
        )
        
    def forward(self, x):
        return self.down_sample_layers(x)

class LightUpSampleDecodeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bias= False):
        super().__init__()
        self.down_sample_layers = nn.Sequential(
            LightFluidalResidualBlock(in_channels),
            nn.Conv3d(in_channels, out_channels, 1, 
                stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.down_sample_layers(x)
