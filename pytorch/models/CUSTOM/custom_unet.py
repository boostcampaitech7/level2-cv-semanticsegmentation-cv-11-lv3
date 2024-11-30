import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ChannelAttention, SpartialAttention, FusionBlock

class CustomUNet(nn.Module):
    """
    Custom U-Net 모델.
    
    특징:
        - Spatial Attention과 Channel Attention을 포함한 FusionBlock 사용.
        - Dropout을 통해 정규화 추가.

    Args:
        in_channels (int): 입력 채널 수 (기본값: 3).
        num_classes (int): 출력 클래스 수 (기본값: 29).
    """
    def __init__(self, in_channels=3, num_classes=29):
        super(CustomUNet, self).__init__()
        
        def CBR(in_channels, out_channels, kernel_size=3, dilation=1, padding=1, bias=True):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels, 
                          kernel_size=kernel_size, 
                          padding=padding, 
                          dilation=dilation, 
                          bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
        
        def unpool(in_channels, out_channels, kernel_size=2, stride=2, bias=True):
            return nn.ConvTranspose2d(in_channels=in_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=kernel_size, 
                                      stride=stride, 
                                      bias=bias)
        
        self.enc1_1 = CBR(in_channels, 64, dilation=2, padding=2)
        self.enc1_2 = CBR(64, 64, dilation=2, padding=2)
        self.spa1 = SpartialAttention(out_channels=64, kernel_size=5)
        self.ca1 = ChannelAttention(in_channels=64, reduction=4)
        self.sum_casp1 = FusionBlock(in_channels=64)
        self.cat_casp1 = FusionBlock(in_channels=64)

        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2_1 = CBR(64, 128, dilation=2, padding=2)
        self.enc2_2 = CBR(128, 128, dilation=2, padding=2)
        self.spa2 = SpartialAttention(out_channels=128, kernel_size=5)
        self.ca2 = ChannelAttention(in_channels=128, reduction=8)
        self.sum_casp2 = FusionBlock(in_channels=128)
        self.cat_casp2 = FusionBlock(in_channels=128)

        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3_1 = CBR(128, 256, dilation=2, padding=2)
        self.enc3_2 = CBR(256, 256, dilation=2, padding=2)
        self.spa3 = SpartialAttention(out_channels=256, kernel_size=5)
        self.ca3 = ChannelAttention(in_channels=256, reduction=16)
        self.sum_casp3 = FusionBlock(in_channels=256)
        self.cat_casp3 = FusionBlock(in_channels=256)

        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4_1 = CBR(256, 512, dilation=2, padding=2)
        self.enc4_2 = CBR(512, 512, dilation=2, padding=2)
        self.spa4 = SpartialAttention(out_channels=512, kernel_size=5)
        self.ca4 = ChannelAttention(in_channels=512, reduction=32)
        self.sum_casp4 = FusionBlock(in_channels=512)
        self.cat_casp4 = FusionBlock(in_channels=512)

        self.pool4 = nn.MaxPool2d(2)
        
        self.bottlenect5_1 = CBR(in_channels=512, out_channels=1024)
        self.bottlenect5_2 = CBR(in_channels=1024, out_channels=512)
        
        self.dec4_2 = unpool(512, 512)
        # self.decfusion4 = FusionBlock(512, 'cat')
        self.decfusion4 = FusionBlock(512, 'sum')
        self.dec4_1 = CBR(512, 512)

        self.dec3_2 = unpool(512, 256)
        # self.decfusion3 = FusionBlock(256, 'cat')
        self.decfusion3 = FusionBlock(256, 'sum')
        self.dec3_1 = CBR(256, 256)

        self.dec2_2 = unpool(256, 128)
        # self.decfusion2 = FusionBlock(128, 'cat')
        self.decfusion2 = FusionBlock(128, 'sum')
        self.dec2_1 = CBR(128, 128)

        self.dec1_2 = unpool(128, 64)
        # self.decfusion1 = FusionBlock(64, 'cat')
        self.decfusion1 = FusionBlock(64, 'sum')
        self.dec1_1 = CBR(64, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        enc1_1_out = self.enc1_1(x)
        enc1_2_out = self.enc1_2(enc1_1_out)
        spa1_out = self.spa1(enc1_2_out)
        ca1_out = self.ca1(enc1_2_out)
        sum_casp1_out = self.sum_casp1(spa1_out, ca1_out)
        cat_casp1_out = self.cat_casp1(spa1_out, ca1_out)
        enc1_out = sum_casp1_out + cat_casp1_out
        pool1_out = self.pool1(enc1_out)
        
        enc2_1_out = self.enc2_1(pool1_out)
        enc2_2_out = self.enc2_2(enc2_1_out)
        spa2_out = self.spa2(enc2_2_out)
        ca2_out = self.ca2(enc2_2_out)
        sum_casp2_out = self.sum_casp2(spa2_out, ca2_out)
        cat_casp2_out = self.cat_casp2(spa2_out, ca2_out)
        enc2_out = sum_casp2_out + cat_casp2_out
        pool2_out = self.pool2(enc2_out)
        
        enc3_1_out = self.enc3_1(pool2_out)
        enc3_2_out = self.enc3_2(enc3_1_out)
        spa3_out = self.spa3(enc3_2_out)
        ca3_out = self.ca3(enc3_2_out)
        sum_casp3_out = self.sum_casp3(spa3_out, ca3_out)
        cat_casp3_out = self.cat_casp3(spa3_out, ca3_out)
        enc3_out = sum_casp3_out + cat_casp3_out
        pool3_out = self.pool3(enc3_out)
        
        enc4_1_out = self.enc4_1(pool3_out)
        enc4_2_out = self.enc4_2(enc4_1_out)
        spa4_out = self.spa4(enc4_2_out)
        ca4_out = self.ca4(enc4_2_out)
        sum_casp4_out = self.sum_casp4(spa4_out, ca4_out)
        cat_casp4_out = self.cat_casp4(spa4_out, ca4_out)
        enc4_out = sum_casp4_out + cat_casp4_out
        pool4_out = self.pool4(enc4_out)
        
        bottleneck5_1 = self.bottlenect5_1(pool4_out)
        bottleneck5_2 = self.bottlenect5_2(bottleneck5_1)
        bottleneck5_2 = F.dropout(bottleneck5_2, p=0.3, training=self.training)
        
        dec4_2 = self.dec4_2(bottleneck5_2)
        decfusion4 = self.decfusion4(dec4_2, enc4_out)
        decfusion4 = F.dropout(decfusion4, p=0.3, training=self.training)
        dec4_1 = self.dec4_1(decfusion4)
        
        dec3_2 = self.dec3_2(dec4_1)
        decfusion3 = self.decfusion3(dec3_2, enc3_out)
        decfusion3 = F.dropout(decfusion3, p=0.3, training=self.training)
        dec3_1 = self.dec3_1(decfusion3)

        dec2_2 = self.dec2_2(dec3_1)
        decfusion2 = self.decfusion2(dec2_2, enc2_out)
        decfusion2 = F.dropout(decfusion2, p=0.3, training=self.training)
        dec2_1 = self.dec2_1(decfusion2)

        dec1_2 = self.dec1_2(dec2_1)
        decfusion1 = self.decfusion1(dec1_2, enc1_out)
        decfusion1 = F.dropout(decfusion1, p=0.3, training=self.training)
        dec1_1 = self.dec1_1(decfusion1)
        
        out = self.out_conv(dec1_1)
        return out