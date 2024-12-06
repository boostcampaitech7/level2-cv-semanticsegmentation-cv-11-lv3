import torch
import torch.nn as nn
from torch.nn import init

def weights_init_kaiming(m):
    """
    Kaiming He 초기화를 적용

    Args:
        m (nn.Module): 초기화를 적용할 레이어.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        
def init_weights(net):
    """
    네트워크의 모든 레이어에 대해 weights_init_kaiming을 적용.

    Args:
        net (nn.Module): 초기화를 적용할 네트워크.

    Returns:
        nn.Module: 초기화된 네트워크.
    """
    return net.apply(weights_init_kaiming)

class unetConv2(nn.Module):
    """
    U-Net의 Convolution 블록.

    Args:
        in_size (int): 입력 채널 수.
        out_size (int): 출력 채널 수.
        is_batchnorm (bool): Batch Normalization 사용 여부.
        n (int): Convolution 연산의 반복 횟수 (기본값: 2).
        ks (int): 커널 크기 (기본값: 3).
        stride (int): 스트라이드 크기 (기본값: 1).
        padding (int): 패딩 크기 (기본값: 1).
    """
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size), nn.ReLU(inplace=True),)
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m)


    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x