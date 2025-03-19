import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

def weight_standardize(w, axis, eps=1e-8): 
    mean = torch.mean(w, dim=axis, keepdim=True)·
    std = torch.std(w, dim=axis, keepdim=True) + eps
    w = (w - mean) / std
    return w

class GroupNorm(nn.BatchNorm2d):

    def __init__(self, num_channels):
        super(GroupNorm, self).__init__(num_features=num_channels)
        
        
class ResidualUnit(nn.Module):

    def __init__(self, in_channels, features, strides=(1, 1)):
        super(ResidualUnit, self).__init__()
        self.features = features
        self.strides = strides

        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=1, stride=1, bias=False)
        self.gn1 = GroupNorm(features)

        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=strides, padding=1, bias=False)
        self.gn2 = GroupNorm(features)

        self.conv3 = nn.Conv2d(features, features * 4, kernel_size=1, stride=1, bias=False)
        self.gn3 = GroupNorm(features * 4)

        if in_channels != features * 4 or strides != (1, 1):
            self.proj = nn.Conv2d(in_channels, features * 4, kernel_size=1, stride=strides, bias=False)
            self.gn_proj = GroupNorm(features * 4)
        else:
            self.proj = None

    def forward(self, x):
        residual = x
        if self.proj is not None:
            residual = self.proj(x)
            residual = self.gn_proj(residual)

        y = self.conv1(x)
        y = self.gn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.gn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.gn3(y)

        y = F.relu(residual + y)
        return y

class ResNetStage(nn.Module):
    """A ResNet stage."""

    def __init__(self, block_size, in_channels, out_channels, first_stride):
        super(ResNetStage, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(ResidualUnit(in_channels, out_channels, strides=first_stride))
        for _ in range(1, block_size):
            self.blocks.append(ResidualUnit(out_channels * 4, out_channels))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class R50Backbone(nn.Module):
    def __init__(self):
        super(R50Backbone, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            GroupNorm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage1 = ResNetStage(3, 64, 64, first_stride=(1, 1))
        self.stage2 = ResNetStage(4, 256, 128, first_stride=(2, 2))
        self.stage3 = ResNetStage(9, 512, 256, first_stride=(2, 2))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self,  patch_size=1, in_chans=1024, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
     
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, C, N) -> (B, N, C)
        return x

class PatchEmbed(nn.Module):
    def __init__(self,  patch_size=1, in_chans=1024, embed_dim=768):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_chans, embed_dim)
        self.backbone = R50Backbone()

    def forward(self, x):
        x = self.backbone(x)
        x = self.patch_embed(x)
        return x
    

class ConcatClassTokenAddPosEmbed(nn.Module):
    def __init__(self, embed_dim=768, num_patches=196):
        super(ConcatClassTokenAddPosEmbed, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, mlp_ratio=4.0, drop=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, int(in_features * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(in_features * mlp_ratio), in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = nn.Identity() if drop_path_ratio == 0 else nn.Dropout(drop_path_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, drop=drop_ratio)

    def forward(self, x):
        b,t,d = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=1, in_channel =1024, embed_dim=768, depth=12, num_heads=12, 
                 qkv_bias=True, qk_scale=None, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed,
                 representation_size=None, num_classes=4):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = embed_layer(patch_size=patch_size, in_chans=in_channel, embed_dim=embed_dim)

        self.cls_token_pos_embed = ConcatClassTokenAddPosEmbed(embed_dim=embed_dim, num_patches=196)

        self.pos_drop = nn.Dropout(drop_ratio)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, qkv_bias, qk_scale, 
                                           drop_ratio, attn_drop_ratio, drop_path_ratio)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        

        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Linear(embed_dim, representation_size)
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
            

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.cls_token_pos_embed(x)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)
        return x

def vit_base_patch16_224(num_classes=4, has_logits=True):
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                              representation_size=768 if has_logits else None, num_classes=num_classes)
    return model
