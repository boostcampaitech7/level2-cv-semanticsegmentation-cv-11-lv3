import torch
import torch.nn as nn


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)  
        x = self.norm(x)
        return x

class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, reduction_ratio=16, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # Q, K, V linear layers
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Sequence Reduction
        self.reduction_ratio = reduction_ratio
        self.reduction = nn.Linear(self.head_dim * reduction_ratio, self.head_dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, C], N = sequence length, C = embedding dimension
        B, N, C = x.shape

        # Query, Key, Value projections
        Q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Sequence reduction on K and V
        K_reduced = K.reshape(B, self.num_heads, N // self.reduction_ratio, -1)
        K_reduced = self.reduction(K_reduced)
        V_reduced = V.reshape(B, self.num_heads, N // self.reduction_ratio, -1)
        V_reduced = self.reduction(V_reduced)

        # Scaled Dot-Product Attention
        attn = (Q @ K_reduced.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Attention output
        output = (attn @ V_reduced).permute(0, 2, 1, 3).reshape(B, N, C)
        output = self.out_proj(output)
        return output




class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
       
        B, N, C = x.shape
        H = W = int(N ** 0.5)  
        x = x.permute(0, 2, 1).view(B, C, H, W)  
        x = self.act(self.conv1(x))
        x = self.dropout(self.conv2(x))
        x = x.flatten(2).permute(0, 2, 1)  
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, reduction_ratio=16, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, reduction_ratio, dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixFFN(dim, hidden_dim, dropout)

    def forward(self, x):
        B, C, H, W = x.shape  
        N = H * W  
        x_flat = x.flatten(2).permute(0, 2, 1)  

        # Self-Attention + Residual
        x_attn = self.attn(self.norm1(x_flat)) 
        x = x + x_attn.permute(0, 2, 1).reshape(B, C, H, W)  

        # Mix-FFN + Residual
        x_flat = x.flatten(2).permute(0, 2, 1) 
        x_ffn = self.mlp(self.norm2(x_flat))  
        x = x + x_ffn.permute(0, 2, 1).reshape(B, C, H, W)  

        return x



class MixTransformerEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], depths=[3, 4, 6, 3], reduction_ratios=[64, 16, 4, 1]):
        super().__init__()
        self.stages = nn.ModuleList()
        for i in range(len(embed_dims)):
            
            patch_embed = OverlapPatchEmbed(
                in_channels=in_channels if i == 0 else embed_dims[i-1],
                embed_dim=embed_dims[i],
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                padding=3 if i == 0 else 1
            )
            
            blocks = nn.Sequential(
                *[TransformerBlock(embed_dims[i], num_heads[i], reduction_ratio=reduction_ratios[i]) for _ in range(depths[i])]
            )
            self.stages.append(nn.Sequential(patch_embed, blocks))

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features  


class SegFormerDecoder(nn.Module):
    def __init__(self, embed_dims=[64, 128, 256, 512], num_classes=21):
        super().__init__()
        self.linear_fuse = nn.Conv2d(sum(embed_dims), 256, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, features):
        
        x = torch.cat([nn.functional.interpolate(f, scale_factor=2**i, mode='bilinear', align_corners=False)
                       for i, f in enumerate(features)], dim=1)
        x = self.linear_fuse(x)  
        x = self.upsample(x)  
        x = self.classifier(x) 
        return x


class SegFormer(nn.Module):
    def __init__(self, in_channels=3, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], depths=[3, 4, 6, 3], num_classes=21):
        super().__init__()
        self.encoder = MixTransformerEncoder(in_channels, embed_dims, num_heads, depths)
        self.decoder = SegFormerDecoder(embed_dims, num_classes)

    def forward(self, x):
        features = self.encoder(x) 
        segmentation_map = self.decoder(features)  
        return segmentation_map


