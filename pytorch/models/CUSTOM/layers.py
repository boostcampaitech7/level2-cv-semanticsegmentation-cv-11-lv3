import torch
import torch.nn as nn

class SpartialAttention(nn.Module):
    """
    Spatial Attention 모듈.
    
    입력 특성 맵의 공간적 정보를 강

    Args:
        out_channels (int): 출력 채널 수.
        kernel_size (int): 컨볼루션 필터 크기 (기본값: 5).
    """
    def __init__(self, out_channels, kernel_size=5):
        super(SpartialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sig(out)
        return out

class ChannelAttention(nn.Module):
    """
    Channel Attention 모듈.
    
    입력 특성 맵의 채널 정보를 강조

    Args:
        in_channels (int): 입력 채널 수.
        reduction (int): 채널 감소율 (기본값: 16).
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # GMP (Optional)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        attn = self.fc(avg_out + max_out)
        attn = attn.view(b, c, 1, 1)
        return x * attn

class FusionBlock(nn.Module):
    """
    FusionBlock 모듈.
    
    Spatial Attention과 Channel Attention 블록의 출력 결합

    Args:
        in_channels (int): 입력 채널 수.
        fusion_type (str): 결합 방식 ("sum" 또는 "cat", 기본값: "sum").
    """
    def __init__(self, in_channels, fusion_type="sum"):
        super(FusionBlock, self).__init__()
        self.fusion_type = fusion_type
        if fusion_type == "cat":
            self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        
    def forward(self, sp_block, ca_block):
        if self.fusion_type == "sum":
            return sp_block + ca_block
        elif self.fusion_type == "cat":
            fused = torch.cat((sp_block, ca_block), dim=1)
            return self.conv(fused)
