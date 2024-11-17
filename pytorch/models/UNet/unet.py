import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet,self).__init__()
        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            
        def unpool(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        
        self.enc1_1 = CBR(in_channels, 64)
        self.enc1_2 = CBR(64, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2_1 = CBR(64, 128)
        self.enc2_2 = CBR(128, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3_1 = CBR(128, 256)
        self.enc3_2 = CBR(256, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4_1 = CBR(256, 512)
        self.enc4_2 = CBR(512, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.enc5_1 = CBR(512, 1024)
        
        self.dec5_1 = CBR(1024, 512)
        
        self.unpool4 = unpool(512,512)
        self.dec4_2 = CBR(512*2,512)
        self.dec4_1 = CBR(512,256)
        
        self.unpool3 = unpool(256, 256)
        self.dec3_2 = CBR(256*2, 256)
        self.dec3_1 = CBR(256, 128)
        
        self.unpool2 = unpool(128, 128)
        self.dec2_2 = CBR(128*2, 128)
        self.dec2_1 = CBR(128, 64)
        
        self.unpool1 = unpool(64, 64)
        self.dec1_2 = CBR(64*2, 64)
        self.dec1_1 = CBR(64, 64)
        
        self.fc = nn.Conv2d(64, num_classes, 1, 1, 0, bias=True)
        
    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)
        
        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)
        
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)
        
        enc5_1 = self.enc5_1(pool4)
        
        dec5_1 = self.dec5_1(enc5_1)
        
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        
        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        
        x = self.fc(dec1_1)
        return x
        
        
        