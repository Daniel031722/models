import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """The Core Brain with Batch Norm for stability"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_c=64):
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder (Learning 'What')
        self.down1 = DoubleConv(in_channels, base_c)
        self.down2 = DoubleConv(base_c, base_c * 2)
        self.down3 = DoubleConv(base_c * 2, base_c * 4)
        self.down4 = DoubleConv(base_c * 4, base_c * 8)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_c * 8, base_c * 8) # Flat bottleneck for VRAM efficiency
        
        # Decoder (Learning 'Where' via Skip Connections)
        self.up4 = nn.ConvTranspose2d(base_c * 8, base_c * 8, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(base_c * 16, base_c * 4)
        
        self.up3 = nn.ConvTranspose2d(base_c * 4, base_c * 4, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(base_c * 8, base_c * 2)
        
        self.up2 = nn.ConvTranspose2d(base_c * 2, base_c * 2, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(base_c * 4, base_c)
        
        self.up1 = nn.ConvTranspose2d(base_c, base_c, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(base_c * 2, base_c)
        
        self.final_conv = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        # Down
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(x4))
        
        # Up + Skip Connections (torch.cat)
        d4 = self.up_conv4(torch.cat([self.up4(b), x4], dim=1))
        d3 = self.up_conv3(torch.cat([self.up3(d4), x3], dim=1))
        d2 = self.up_conv2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.up_conv1(torch.cat([self.up1(d2), x1], dim=1))
        
        return self.final_conv(d1)