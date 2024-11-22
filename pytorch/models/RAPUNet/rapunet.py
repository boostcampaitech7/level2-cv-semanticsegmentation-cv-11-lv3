import torch
import torch.nn as nn
import torch.nn.functional as F
from .Backbone import metaformer_baseline
from .layers import RAPU, ResNetBlock, SBA, ConvBnAct

class CAFormerBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(CAFormerBackbone, self).__init__()
        self.backbone = metaformer_baseline.caformer_s18(pretrained=pretrained)
        self.num_stage = self.backbone.num_stage
        self.downsample_layers = self.backbone.downsample_layers
        self.stages = self.backbone.stages
        
    def forward(self, x):
        layers = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            layers.append(x)
        return layers
    
class RAPUNet(nn.Module):
    def __init__(self, in_channels, num_classes, starting_filters):
        super(RAPUNet, self).__init__()
        self.starting_filters = starting_filters
        self.dim = 32
        self.backbone = CAFormerBackbone(pretrained=True)

        self.p1_conv = nn.Conv2d(in_channels, starting_filters * 2, kernel_size=3, stride=2, padding=1)

        self.p2_conv = nn.Conv2d(64, starting_filters * 4, kernel_size=1, padding=0)
        self.p3_conv = nn.Conv2d(128, starting_filters * 8, kernel_size=1, padding=0)
        self.p4_conv = nn.Conv2d(320, starting_filters * 16, kernel_size=1, padding=0)
        self.p5_conv = nn.Conv2d(512, starting_filters * 32, kernel_size=1, padding=0)

        self.t0_RAPU = RAPU(in_channels, starting_filters)

        self.l1i_conv = nn.Conv2d(starting_filters, starting_filters * 2, kernel_size=2, stride=2)
        self.t1_RAPU = RAPU(starting_filters * 2, starting_filters * 2)

        self.l2i_conv = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2)
        self.t2_RAPU = RAPU(starting_filters * 4, starting_filters * 4)

        self.l3i_conv = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2)
        self.t3_RAPU = RAPU(starting_filters * 8, starting_filters * 8)

        self.l4i_conv = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2)
        self.t4_RAPU = RAPU(starting_filters * 16, starting_filters * 16)

        self.l5i_conv = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2)

        self.t51_resnet1 = ResNetBlock(starting_filters * 32, starting_filters * 32)
        self.t51_resnet2 = ResNetBlock(starting_filters * 32, starting_filters * 32)

        self.t53_resnet1 = ResNetBlock(starting_filters * 32, starting_filters * 16)
        self.t53_resnet2 = ResNetBlock(starting_filters * 16, starting_filters * 16)

        self.outd_conv1 = ConvBnAct(starting_filters * 40, self.dim, kernel_size=1, activation=nn.ReLU(inplace=True))
        self.outd_conv2 = nn.Conv2d(self.dim, 1, kernel_size=1, bias=False)

        self.L_input_conv = ConvBnAct(starting_filters * 4, self.dim, kernel_size=3, activation=nn.ReLU(inplace=True))
        self.H_input_conv1 = ConvBnAct(starting_filters * 32, self.dim, kernel_size=1, activation=nn.ReLU(inplace=True))

        self.sba = SBA(self.dim, self.dim, out_channels=self.dim)

        self.final_conv = nn.Conv2d(1, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        backbone_features = self.backbone(x)
        backbone_features = [feature.permute(0, 3, 1, 2) for feature in backbone_features]
        backbone_features = backbone_features[::-1]
            
        p1 = self.p1_conv(x)

        p2 = self.p2_conv(backbone_features[3])
        p3 = self.p3_conv(backbone_features[2])
        p4 = self.p4_conv(backbone_features[1])
        p5 = self.p5_conv(backbone_features[0])

        t0 = self.t0_RAPU(x)

        l1i = self.l1i_conv(t0)
        s1 = l1i + p1
        t1 = self.t1_RAPU(s1)

        l2i = self.l2i_conv(t1)
        s2 = l2i + p2
        t2 = self.t2_RAPU(s2)

        l3i = self.l3i_conv(t2)
        s3 = l3i + p3
        t3 = self.t3_RAPU(s3)

        l4i = self.l4i_conv(t3)
        s4 = l4i + p4
        t4 = self.t4_RAPU(s4)

        l5i = self.l5i_conv(t4)
        s5 = l5i + p5

        t51 = self.t51_resnet1(s5)
        t51 = self.t51_resnet2(t51)

        t53 = self.t53_resnet1(t51)
        t53 = self.t53_resnet2(t53)

        up_t53_4x = F.interpolate(t53, scale_factor=4, mode='bilinear')
        up_t4_2x = F.interpolate(t4, scale_factor=2, mode='bilinear')

        outd = torch.cat([up_t53_4x, up_t4_2x, t3], dim=1)
        outd = self.outd_conv1(outd)
        outd = self.outd_conv2(outd)

        L_input = self.L_input_conv(t2)

        up_t53_2x = F.interpolate(t53, scale_factor=2, mode='bilinear')
        H_input = torch.cat([up_t53_2x, t4], dim=1)
        H_input = self.H_input_conv1(H_input)
        H_input = F.interpolate(H_input, scale_factor=2, mode='bilinear')

        out2 = self.sba(L_input, H_input)

        out1 = F.interpolate(outd, scale_factor=8, mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, scale_factor=4, mode='bilinear', align_corners=False)

        out_dual = out1 + out2
        output = self.final_conv(out_dual)
        return output
