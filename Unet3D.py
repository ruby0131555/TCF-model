import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class LayerNorm3d(nn.Module):
    # 3D Layer Normalization for (B, C, D, H, W) tensors
    def __init__(self, num_channels, eps=1e-5):
        super(LayerNorm3d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)
        var = x.var(dim=(1, 2, 3, 4), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return x

class SpatialAttention(nn.Module):
    # Simplified 3D spatial attention mechanism
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, max(in_channels // 16, 4), kernel_size=1),  
            LayerNorm3d(max(in_channels // 16, 4)),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(in_channels // 16, 4), 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.conv(x)
        return x * attention

class Conv3d(nn.Module):
    # Residual 3D Convolutional block
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dropout=0.1):
        super(Conv3d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            LayerNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()  
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            LayerNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        
        # Identity shortcut for residual connection
        if in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
                LayerNorm3d(out_channel)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.residual(x)

class Down(nn.Module):
    # Downsampling block using MaxPool and Conv3d
    def __init__(self, in_channel, out_channel, kernel_size, stride, dropout=0.1):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=kernel_size, stride=stride),
            Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    # Upsampling block with skip connection concatenation
    def __init__(self, x1_in, x2_in, out_channel, kernel_size, stride, padding, dropout=0.1):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(x1_in, x1_in, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            LayerNorm3d(x1_in),
            nn.ReLU(inplace=True)
        )
        self.conv = Conv3d(x1_in + x2_in, out_channel, kernel_size=3, stride=1, padding=1, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle potential padding differences
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        if diffD > 0 or diffH > 0 or diffW > 0:
            x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                            diffH // 2, diffH - diffH // 2,
                            diffD // 2, diffD - diffD // 2])

        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):
    # Final output block with multi-scale feature fusion
    def __init__(self, in_channel_list, out_channel, kernel_size, stride, padding, obs_len, dropout=0.1):
        super(OutConv, self).__init__()
        self.obs_len = obs_len
        self.up_list = nn.ModuleList()

        # Build upsampling layers for different feature scales
        for i, channel in enumerate(in_channel_list):
            if i == len(in_channel_list) - 1:
                continue
            
            scale_factor = np.power(2, (len(in_channel_list) - 1) - i)
            self.up_list.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        channel, channel,
                        kernel_size=[1, scale_factor, scale_factor],
                        stride=[1, scale_factor, scale_factor],
                        padding=padding, bias=False
                    ),
                    LayerNorm3d(channel),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(channel, in_channel_list[-1], kernel_size=3, stride=1, padding=1, bias=False),
                    LayerNorm3d(in_channel_list[-1]),
                    nn.ReLU(inplace=True)
                )
            )

        self.depth_upsample_x6 = nn.Sequential(
            nn.ConvTranspose3d(in_channel_list[-1], in_channel_list[-1],
                               kernel_size=[2, 1, 1], stride=[2, 1, 1], padding=0, bias=False),
            LayerNorm3d(in_channel_list[-1]),
            nn.ReLU(inplace=True)
        )

        self.spatial_attention = SpatialAttention(in_channel_list[-1] * len(in_channel_list))
        self.conv = nn.Sequential(
            nn.Conv3d(in_channel_list[-1] * len(in_channel_list), in_channel_list[-1] * 2,
                      kernel_size=[1, 3, 3], stride=[1, 1, 1], padding=[0, 1, 1], bias=False),
            LayerNorm3d(in_channel_list[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),  
            nn.Conv3d(in_channel_list[-1] * 2, in_channel_list[-1],
                      kernel_size=[1, 3, 3], stride=[1, 1, 1], padding=[0, 1, 1], bias=False),
            LayerNorm3d(in_channel_list[-1]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),  
            nn.Conv3d(in_channel_list[-1], out_channel, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x_list_from_unet):
        x6, x7, x8, x9 = tuple(x_list_from_unet)
        x6_processed = self.up_list[0](x6)
        x7_processed = self.up_list[1](x7)
        x8_processed = self.up_list[2](x8)

        # Ensure temporal dimension matches observation length
        if x6_processed.size(2) != self.obs_len:
            x6_processed = self.depth_upsample_x6(x6_processed)

        x_fused = torch.cat([x6_processed, x7_processed, x8_processed, x9], dim=1)
        x_fused = self.spatial_attention(x_fused)
        return self.conv(x_fused)

class Unet3D(nn.Module):
    # Lightweight 3D Unet for meteorological feature extraction
    def __init__(self, in_channels, out_channels, obs_len, base_channels=16, dropout=0.15):
        super(Unet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.obs_len = obs_len

        # Define channel dimensions for each level
        c1, c2, c3, c4, c5 = base_channels, base_channels*2, base_channels*4, base_channels*6, base_channels*6

        # Encoder stages
        self.inc = Conv3d(in_channels, c1, kernel_size=3, stride=1, padding=1, dropout=dropout)
        self.down1 = Down(c1, c2, kernel_size=[1, 2, 2], stride=[1, 2, 2], dropout=dropout)
        self.down2 = Down(c2, c3, kernel_size=[1, 2, 2], stride=[1, 2, 2], dropout=dropout)
        self.down3 = Down(c3, c4, kernel_size=[2, 2, 2], stride=[2, 2, 2], dropout=dropout)
        self.down4 = Down(c4, c5, kernel_size=[2, 2, 2], stride=[2, 2, 2], dropout=dropout)

        self.bottleneck_attention = SpatialAttention(c5)

        # Decoder stages
        self.up1 = Up(c5, c4, c3, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0, dropout=dropout)
        self.up2 = Up(c3, c3, c2, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0, dropout=dropout)
        self.up3 = Up(c2, c2, c1, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0, dropout=dropout)
        self.up4 = Up(c1, c1, c1, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0, dropout=dropout)

        # Output head
        self.outc = OutConv([c3, c2, c1, c1], out_channels, 
                           kernel_size=[1, 1, 1], stride=[1, 1, 1], 
                           padding=[0, 0, 0], obs_len=obs_len, dropout=dropout)

    def forward(self, x):
        batch_size, channels, time_steps, height, width = x.size()
        # Shape validation
        assert channels == self.in_channels
        assert time_steps == self.obs_len
        
        # Encoder forward
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottleneck_attention(x5)

        # Decoder forward
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        out = self.outc([x6, x7, x8, x9])
        return out

if __name__ == '__main__':
    # Test script for Unet3D configurations
    print("Testing Unet3D with various configurations")
    test_configs = [
        {'in_ch': 1, 'obs_len': 4, 'desc': 'Single channel baseline'},
        {'in_ch': 13, 'obs_len': 4, 'desc': '13 channels (u, v, z, sst)'},
    ]
    
    for config in test_configs:
        in_ch, obs_len = config['in_ch'], config['obs_len']
        print(f"Testing: {config['desc']} | Channels: {in_ch}, Seq: {obs_len}")
        
        x = torch.randn((2, in_ch, obs_len, 64, 64)).cuda()
        net = Unet3D(in_channels=in_ch, out_channels=1, obs_len=obs_len).cuda()
        
        with torch.no_grad():
            out = net(x)
        
        print(f"Output shape: {tuple(out.shape)} | Status: Success")
        del net, x, out
        torch.cuda.empty_cache()
