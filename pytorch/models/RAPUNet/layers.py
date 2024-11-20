import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, ):
        super(self, ResNetBlock).__init__()
        def CR(in_channels, out_channels, dilation_rate=1):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation_rate, padding=0),
                
            )
        
        self.conv1 = nn.Conv2d()