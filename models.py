import einops
import torch 
from torch import nn
from torch.nn import functional as F 
from layers import *

class FluidalUnet(nn.Module):

    def __init__(self, lags, latent_dim):
        super().__init__()

        self.maxpool3d = nn.MaxPool3d(2,2)

        self.down0 = EncodingStartBlock(4*lags, latent_dim)
        self.pool0 = nn.Sequential(
            nn.Conv3d(latent_dim, 2*latent_dim, 3, 
                stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm3d(2*latent_dim),
            nn.ReLU(), 
            nn.MaxPool3d(2,2)
        ) 
        self.down1 = FluidalResidualBlock(2*latent_dim, latent_dim)
        self.pool1 = DownSampleBlock(2*latent_dim, 4*latent_dim)

        self.down2 = FluidalResidualBlock(4*latent_dim, 2*latent_dim)
        self.pool2 = DownSampleBlock(4*latent_dim, 8*latent_dim)

        self.encode = FluidalResidualBlock(8*latent_dim, 4*latent_dim)

        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(8*latent_dim, 4*latent_dim, 2,2, bias= False),
            nn.BatchNorm3d(4*latent_dim),
            nn.ReLU()
        )
        self.up2_decode = UpSampleDecodeBlock(8*latent_dim, 4*latent_dim)
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(4*latent_dim, 2*latent_dim, 2,2, bias= False),
            nn.BatchNorm3d(2*latent_dim),
            nn.ReLU()
        )
        self.up1_decode = UpSampleDecodeBlock(4*latent_dim, 2*latent_dim)
        
        self.up0 = nn.Sequential(
            nn.ConvTranspose3d(2*latent_dim, latent_dim, 2,2, bias= False),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU()
        )
        self.up0_decode = UpSampleDecodeBlock(2*latent_dim, latent_dim)

        self.decode = nn.Sequential(
            nn.Conv3d(latent_dim, 4, 3, 1, 1, 1, 1, False),
            nn.BatchNorm3d(4),
            nn.ReLU()
        )
        self.lags= lags
        self.latent_dim = latent_dim
    
    def forward(self, x):
        x = einops.rearrange(x, "b l z x y w -> b (l w) z x y")
        # bs, lagx4, z, x, y
        c0 = self.down0(x) # bs, k, z, x, y
        c1 = self.pool0(c0) 
        # print(c0.shape)

        c1 = self.down1(c1) # bs, 2k, z/2, x/2, y/2
        c2 = self.pool1(c1) 
        # print(c1.shape)

        c3 = self.down2(c2) # bs, 4k, z/4, x/4, y/4
        c3 = self.pool2(c3) 
        # print(c2.shape)

        c3 = self.encode(c3) # bs, 8k, z/8, x/8, y/8
        # print(c3.shape)

        c3 = self.up2(c3) # bs, 4k, z/4, x/4, y/4
        c2 = torch.cat([c2,c3], dim= 1)
        del c3
        c2 = self.up2_decode(c2)

        c2 = self.up1(c2) # bs, 2k, z/2, x/2, y/2
        c1 = torch.cat([c1,c2], dim= 1)
        del c2
        c1 = self.up1_decode(c1)

        c1 = self.up0(c1)
        c0 = torch.cat([c0,c1], dim= 1)
        del c1
        c0 = self.up0_decode(c0)

        c0 = self.decode(c0) # bs 4 z x y
        c0 = einops.rearrange(c0, "b v z x y -> b z x y v")
        return c0

class LightFluidalUnet(nn.Module):

    def __init__(self, lags, latent_dim):
        super().__init__()

        self.maxpool3d = nn.MaxPool3d(2,2)

        self.down0 = LightEncodingStartBlock(4*lags, latent_dim)
        self.pool0 = nn.Sequential(
            nn.Conv3d(latent_dim, 2*latent_dim, 3, 
                stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm3d(2*latent_dim),
            nn.ReLU(), 
            nn.MaxPool3d(2,2)
        ) 
        self.down1 = LightFluidalResidualBlock(2*latent_dim)
        self.pool1 = LightDownSampleBlock(2*latent_dim, 4*latent_dim)

        self.down2 = LightFluidalResidualBlock(4*latent_dim)
        self.pool2 = LightDownSampleBlock(4*latent_dim, 8*latent_dim)

        self.encode = LightFluidalResidualBlock(8*latent_dim)

        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(8*latent_dim, 4*latent_dim, 2,2, bias= False),
            nn.BatchNorm3d(4*latent_dim),
            nn.ReLU()
        )
        self.up2_decode = LightUpSampleDecodeBlock(8*latent_dim, 4*latent_dim)
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(4*latent_dim, 2*latent_dim, 2,2, bias= False),
            nn.BatchNorm3d(2*latent_dim),
            nn.ReLU()
        )
        self.up1_decode = LightUpSampleDecodeBlock(4*latent_dim, 2*latent_dim)
        
        self.up0 = nn.Sequential(
            nn.ConvTranspose3d(2*latent_dim, latent_dim, 2,2, bias= False),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU()
        )
        self.up0_decode = LightUpSampleDecodeBlock(2*latent_dim, latent_dim)

        self.decode = nn.Sequential(
            nn.Conv3d(latent_dim, 4, 3, 1, 1, 1, 1, False),
            nn.BatchNorm3d(4),
            nn.ReLU()
        )
        self.lags= lags
        self.latent_dim = latent_dim
    
    def forward(self, x):
        x = einops.rearrange(x, "b l z x y w -> b (l w) z x y")
        # bs, lagx4, z, x, y
        c0 = self.down0(x) # bs, k, z, x, y
        c1 = self.pool0(c0) 
        # print(c0.shape)

        c1 = self.down1(c1) # bs, 2k, z/2, x/2, y/2
        c2 = self.pool1(c1) 
        # print(c1.shape)

        c3 = self.down2(c2) # bs, 4k, z/4, x/4, y/4
        c3 = self.pool2(c3) 
        # print(c2.shape)

        c3 = self.encode(c3) # bs, 8k, z/8, x/8, y/8
        # print(c3.shape)

        c3 = self.up2(c3) # bs, 4k, z/4, x/4, y/4
        c2 = torch.cat([c2,c3], dim= 1)
        del c3
        c2 = self.up2_decode(c2)

        c2 = self.up1(c2) # bs, 2k, z/2, x/2, y/2
        c1 = torch.cat([c1,c2], dim= 1)
        del c2
        c1 = self.up1_decode(c1)

        c1 = self.up0(c1)
        c0 = torch.cat([c0,c1], dim= 1)
        del c1
        c0 = self.up0_decode(c0)

        c0 = self.decode(c0) # bs 4 z x y
        c0 = einops.rearrange(c0, "b v z x y -> b z x y v")
        return c0