import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class NestedUNet(nn.Module):
    def __init__(self, in_channels, num_classes, deep_supervision=False):
        super().__init__()
        filters = [32, 64, 128, 256, 512]
        
        self.deep_supervision = deep_supervision
        
        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = VGGBlock(in_channels, filters[0], filters[0])
        self.conv1_0 = VGGBlock(filters[0], filters[1], filters[1])
        self.conv2_0 = VGGBlock(filters[1], filters[2], filters[2])
        self.conv3_0 = VGGBlock(filters[2], filters[3], filters[3])
        self.conv4_0 = VGGBlock(filters[3], filters[4], filters[4])
        
        self.conv0_1 = VGGBlock(filters[0]+filters[1], filters[0], filters[0])
        self.conv1_1 = VGGBlock(filters[1]+filters[2], filters[1], filters[1])
        self.conv2_1 = VGGBlock(filters[2]+filters[3], filters[2], filters[2])
        self.conv3_1 = VGGBlock(filters[3]+filters[4], filters[3], filters[3])
        
        self.conv0_2 = VGGBlock(filters[0]*2+filters[1], filters[0], filters[0])
        self.conv1_2 = VGGBlock(filters[1]*2+filters[2], filters[1], filters[1])
        self.conv2_2 = VGGBlock(filters[2]*2+filters[3], filters[2], filters[2])
        
        self.conv0_3 = VGGBlock(filters[0]*3+filters[1], filters[0], filters[0])
        self.conv1_3 = VGGBlock(filters[1]*3+filters[2], filters[1], filters[1])
        
        self.conv0_4 = VGGBlock(filters[0]*4+filters[1], filters[0], filters[0])
        
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)
    
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output