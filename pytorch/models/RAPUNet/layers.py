import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='relu', padding='same'):
        super(ConvBnAct, self).__init__()
        if padding == 'same':
            padding = (kernel_size - 1) // 2
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')  # He 초기화
        self.bn = nn.BatchNorm2d(out_channels)

        if isinstance(activation, str):
            if activation.lower() == 'relu':
                self.act = nn.ReLU(inplace=True)
            elif activation.lower() == 'sigmoid':
                self.act = nn.Sigmoid()
            elif activation.lower() == 'tanh':
                self.act = nn.Tanh()
            elif activation.lower() == 'leaky_relu':
                self.act = nn.LeakyReLU(inplace=True)
            elif activation.lower() == 'gelu':
                self.act = nn.GELU()
            elif activation.lower() == 'none':
                self.act = nn.Identity()
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
        elif isinstance(activation, nn.Module):
            self.act = activation
        else:
            raise TypeError(f"activation must be a string or nn.Module, got {type(activation)}")
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(ResNetBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=dilation_rate)
        nn.init.kaiming_uniform_(self.conv1x1.weight, nonlinearity='relu')
        self.relu = nn.ReLU(inplace=True)
        padding = dilation_rate
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation_rate)
        nn.init.kaiming_uniform_(self.conv3x3_1.weight, nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv3x3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation_rate)
        nn.init.kaiming_uniform_(self.conv3x3_2.weight, nonlinearity='relu')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn_final = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x1 = self.conv1x1(x)
        x1 = self.relu(x1)
        x2 = self.conv3x3_1(x)
        x2 = self.relu(x2)
        x2 = self.bn1(x2)
        x2 = self.conv3x3_2(x2)
        x2 = self.relu(x2)
        x2 = self.bn2(x2)
        x_out = x1 + x2
        x_out = self.bn_final(x_out)
        return x_out

class AtrousBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AtrousBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        return x

class RAPU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RAPU, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.atrous_block = AtrousBlock(in_channels, out_channels)
        self.resnet_block = ResNetBlock(in_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.bn1(x)
        x1 = self.atrous_block(x)
        x2 = self.resnet_block(x)
        x = x1 + x2
        x = self.bn2(x)
        return x

class SBA(nn.Module):
    def __init__(self, in_channels_L, in_channels_H, out_channels=16):
        super(SBA, self).__init__()
        self.dim = out_channels
        self.conv1_L = nn.Conv2d(in_channels_L, out_channels, kernel_size=1, padding=0, bias=False)
        nn.init.kaiming_uniform_(self.conv1_L.weight, nonlinearity='relu')
        self.conv1_H = nn.Conv2d(in_channels_H, out_channels, kernel_size=1, padding=0, bias=False)
        nn.init.kaiming_uniform_(self.conv1_H.weight, nonlinearity='relu')
        self.sigmoid = nn.Sigmoid()
        self.convf_bn_act_L = ConvBnAct(out_channels, out_channels, kernel_size=1, activation='relu', padding='same')
        self.convf_bn_act_H = ConvBnAct(out_channels, out_channels, kernel_size=1, activation='relu', padding='same')
        self.convf_bn_act_out = ConvBnAct(out_channels*2, out_channels*2, kernel_size=3, activation='relu', padding='same')
        self.conv_out = nn.Conv2d(out_channels*2, 1, kernel_size=1, padding=0, bias=False)
        nn.init.kaiming_uniform_(self.conv_out.weight, nonlinearity='relu')
        
    def forward(self, L_input, H_input):
        L_input = self.conv1_L(L_input)
        H_input = self.conv1_H(H_input)
        g_L = self.sigmoid(L_input)
        g_H = self.sigmoid(H_input)
        L_input = self.convf_bn_act_L(L_input)
        H_input = self.convf_bn_act_H(H_input)
        temp_H = g_H * H_input
        temp_L = g_L * L_input
        temp_H_upsampled = F.interpolate(temp_H, scale_factor=2, mode='bilinear')
        L_feature = L_input + L_input * g_L + (1 - g_L) * temp_H_upsampled
        H_input_size = H_input.size()
        H_input_H = H_input_size[2]
        H_input_W = H_input_size[3]
        temp_L_resized = F.interpolate(temp_L, size=(H_input_H, H_input_W), mode='bilinear')
        H_feature = H_input + H_input * g_H + (1 - g_H) * temp_L_resized
        H_feature = F.interpolate(H_feature, scale_factor=2, mode='bilinear')
        out = torch.cat([L_feature, H_feature], dim=1)
        out = self.convf_bn_act_out(out)
        out = self.conv_out(out)
        return out
