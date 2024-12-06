import torch
import torch.nn as nn
import torch.functional as F
from .layers import conv_block_2D
import torch.nn.init as init

class DUCKNet(nn.Module):
    """
    DUCKNet 모델.

    Args:
        in_channels (int): 입력 채널 수.
        num_classes (int): 출력 클래스 수.
        starting_filters (int): 처음 컨볼루션 레이어의 필터 수.
    """
    def __init__(self, in_channels, num_classes, starting_filters):
        super(DUCKNet, self).__init__()
        self.input_layer = nn.Identity()

        self.p1 = nn.Conv2d(in_channels, starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.p2 = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.p3 = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.p4 = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding=0)
        self.p5 = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding=0)

        self.t0 = conv_block_2D('duckv2', in_channels, starting_filters)

        self.l1i = nn.Conv2d(starting_filters, starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.t1 = conv_block_2D('duckv2', starting_filters * 2, starting_filters * 2)

        self.l2i = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.t2 = conv_block_2D('duckv2', starting_filters * 4, starting_filters * 4)

        self.l3i = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.t3 = conv_block_2D('duckv2', starting_filters * 8, starting_filters * 8)

        self.l4i = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding=0)
        self.t4 = conv_block_2D('duckv2', starting_filters * 16, starting_filters * 16)

        self.l5i = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding=0)
        self.t51 = conv_block_2D('resnet', starting_filters * 32, starting_filters * 32, repeat=2)
        self.t53 = conv_block_2D('resnet', starting_filters * 32, starting_filters * 16, repeat=2)

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.q4 = conv_block_2D('duckv2', starting_filters * 16, starting_filters * 8)

        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.q3 = conv_block_2D('duckv2', starting_filters * 8, starting_filters * 4)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.q6 = conv_block_2D('duckv2', starting_filters * 4, starting_filters * 2)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.q1 = conv_block_2D('duckv2', starting_filters * 2, starting_filters)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.z1 = conv_block_2D('duckv2', starting_filters, starting_filters)

        self.output_layer = nn.Conv2d(starting_filters, num_classes, kernel_size=1)

        self.apply(self._initialize_weights)
        
    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)
    
    def forward(self, x):
        input_layer = self.input_layer(x)

        p1 = self.p1(input_layer)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        t0 = self.t0(input_layer)

        l1i = self.l1i(t0)
        s1 = l1i + p1
        t1 = self.t1(s1)

        l2i = self.l2i(t1)
        s2 = l2i + p2
        t2 = self.t2(s2)

        l3i = self.l3i(t2)
        s3 = l3i + p3
        t3 = self.t3(s3)

        l4i = self.l4i(t3)
        s4 = l4i + p4
        t4 = self.t4(s4)

        l5i = self.l5i(t4)
        s5 = l5i + p5

        t51 = self.t51(s5)
        t53 = self.t53(t51)

        l5o = self.up5(t53)
        c4 = l5o + t4
        q4 = self.q4(c4)

        l4o = self.up4(q4)
        c3 = l4o + t3
        q3 = self.q3(c3)

        l3o = self.up3(q3)
        c2 = l3o + t2
        q6 = self.q6(c2)

        l2o = self.up2(q6)
        c1 = l2o + t1
        q1 = self.q1(c1)

        l1o = self.up1(q1)
        c0 = l1o + t0
        z1 = self.z1(c0)

        output = self.output_layer(z1)
        return output