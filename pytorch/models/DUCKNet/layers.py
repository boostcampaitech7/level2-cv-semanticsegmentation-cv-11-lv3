import torch.nn as nn

class SeparatedConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,size), padding=(0,padding)),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(size,1), padding=(padding,0)),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    
class MidScopeConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class WideScopeConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
class ResNetConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super().__init__()
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=dilation_rate),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        skip = self.conv_skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += skip
        x = self.bn(x)
        return x
    
class DUCKv2Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.widescope = WideScopeConv2DBlock(in_channels, out_channels)
        self.midscope = MidScopeConv2DBlock(in_channels, out_channels)
        self.resnet1 = ResNetConv2DBlock(in_channels, out_channels)
        self.resnet2 = nn.Sequential(
            ResNetConv2DBlock(in_channels, out_channels),
            ResNetConv2DBlock(out_channels, out_channels)
        )
        self.resnet3 = nn.Sequential(
            ResNetConv2DBlock(in_channels, out_channels),
            ResNetConv2DBlock(out_channels, out_channels),
            ResNetConv2DBlock(out_channels, out_channels)
        )
        self.separated = SeparatedConv2DBlock(in_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.bn1(x)
        x1 = self.widescope(x)
        x2 = self.midscope(x)
        x3 = self.resnet1(x)
        x4 = self.resnet2(x)
        x5 = self.resnet3(x)
        x6 = self.separated(x)
        x = x1 + x2 + x3 + x4 + x5 + x6
        x = self.bn2(x)
        return x

class DoubleConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
def conv_block_2D(block_type, in_channels, out_channels, repeat=1, dilation_rate=1, size=3, padding=1):
    layers = []
    for _ in range(repeat):
        if block_type == 'separated':
            layers.append(SeparatedConv2DBlock(in_channels, out_channels))
        elif block_type == 'duckv2':
            layers.append(DUCKv2Conv2DBlock(in_channels, out_channels))
        elif block_type == 'midscope':
            layers.append(MidScopeConv2DBlock(in_channels, out_channels))
        elif block_type == 'widescope':
            layers.append(WideScopeConv2DBlock(in_channels, out_channels))
        elif block_type == 'resnet':
            layers.append(ResNetConv2DBlock(in_channels, out_channels, dilation_rate=dilation_rate))
        elif block_type == 'conv':
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=size, padding=padding),
                nn.ReLU()
            ))
        elif block_type == 'double':
            layers.append(DoubleConvBN(in_channels, out_channels, dilation_rate=dilation_rate))
        else:
            raise ValueError(f'알수없는 블록 타입 {block_type}')
        in_channels = out_channels
    return nn.Sequential(*layers)