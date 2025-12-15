
# 修改613
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Block
import math
import torchvision
import torch.utils.checkpoint as checkpoint
import gazelle.utils as utils
from gazelle.backbone import DinoV2Backbone

# === HiLo Attention ===
class HiLo(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim / num_heads)
        self.dim = dim
        self.l_heads = int(num_heads * alpha)
        self.l_dim = self.l_heads * head_dim
        self.h_heads = num_heads - self.l_heads
        self.h_dim = self.h_heads * head_dim
        self.ws = window_size

        if self.ws == 1:
            self.h_heads, self.h_dim = 0, 0
            self.l_heads, self.l_dim = num_heads, dim

        self.scale = qk_scale or head_dim ** -0.5

        # 低频头的层定义
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # 高频头的层定义
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    def compute_qkv(self, x, heads, dim, is_hifi=True):
        B, H, W, C = x.shape
        if is_hifi:
            # 高频部分的QKV计算
            x = x.reshape(B, H // self.ws, self.ws, W // self.ws, self.ws, C).transpose(2, 3)
            qkv = self.h_qkv(x).reshape(B, H // self.ws * W // self.ws, heads, dim // heads).permute(0, 2, 1, 3)
        else:
            # 低频部分的QKV计算
            qkv = self.l_qkv(x).reshape(B, H * W, heads, dim // heads).permute(0, 2, 1, 3)

        k, v = qkv[1], qkv[2]
        return k, v

    def hifi(self, x):
        B, H, W, C = x.shape
        k, v = self.compute_qkv(x, self.h_heads, self.h_dim, is_hifi=True)
        attn = (k @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, H // self.ws, W // self.ws, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, H // self.ws * self.ws, W // self.ws * self.ws, self.h_dim)
        return self.h_proj(x)

    def lofi(self, x):
        B, H, W, C = x.shape
        k, v = self.compute_qkv(x, self.l_heads, self.l_dim, is_hifi=False)
        attn = (k @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        return self.l_proj(x)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.reshape(B, H, W, C)
        if self.h_heads == 0:
            return checkpoint.checkpoint(self.lofi, x).reshape(B, N, C)
        if self.l_heads == 0:
            return checkpoint.checkpoint(self.hifi, x).reshape(B, N, C)
        hifi_out = checkpoint.checkpoint(self.hifi, x)
        lofi_out = checkpoint.checkpoint(self.lofi, x)
        return torch.cat((hifi_out, lofi_out), dim=-1).reshape(B, N, C)


# === Pixel Attention ===
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)

    def forward(self, x, pattn1):
        x = x.unsqueeze(dim=2)
        pattn1 = pattn1.unsqueeze(dim=2)
        x2 = torch.cat([x, pattn1], dim=2)
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        return pattn2  # 移除sigmoid


# === CGA Attention ===
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result


# === GazeLLE ===
class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
        super().__init__()
        self.inout = inout  # 外部控制inout
        self.num_layers = num_layers  # 外部配置num_layers
        self.backbone = backbone
        self.dim = dim
        self.in_size = in_size
        self.out_size = out_size
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.linear = nn.Conv2d(backbone.get_dimension(), dim, 1)
        self.head_token = nn.Embedding(1, dim)
        if inout:
            self.inout_token = nn.Embedding(1, dim)
        self.transformer = nn.Sequential(*[
            Block(dim=dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
            for _ in range(self.num_layers)
        ])
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 2, 2),
            nn.Conv2d(dim, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.attention_fusion = CGAFusion(dim)  # 使用CGA融合

    def forward(self, input):
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        x = self.linear(self.backbone(input["images"]))
        depth_map = input["depth_map"]
        depth_features = F.interpolate(depth_map, size=x.shape[2:], mode='bilinear', align_corners=False)
        x, depth_features = self.interaction(x, depth_features)
        x = self.fusion(x, depth_features)
        x = self.calibration(x)
        x = self.enhancement(x)
        x = self.attention_fusion(x, depth_features)  # 使用CGA融合
        pos_embed = self.get_depth_guided_positional_encoding(depth_features)
        x = x + pos_embed
        x = utils.repeat_tensors(x, num_ppl_per_img)
        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
        head_map_embeddings = head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings
        x = x.flatten(start_dim=2).permute(0, 2, 1)
        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1), x], dim=1)
        x = self.transformer(x)
        if self.inout:
            inout_tokens = x[:, 0, :]
            inout_preds = self.inout_head(inout_tokens).squeeze(-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :]
        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
        x = self.heatmap_head(x).squeeze(1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = utils.split_tensors(x, num_ppl_per_img)
        return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

    def get_depth_guided_positional_encoding(self, depth_features):
        B, _, H, W = depth_features.shape
        key = (H, W)
        if key not in self._pos_cache:
            self._pos_cache[key] = positionalencoding2d(self.dim, H, W).to(depth_features.device)
        return self._pos_cache[key]


# === Positional Encoding ===
def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError("位置编码要求 d_model 为4的倍数")
    pe = torch.zeros(d_model, height, width)
    d_model = d_model // 2
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe


# === 模型工厂函数 ===
def get_gazelle_model(model_name):
    factory = {
        "gazelle_cgaf_dinov2_vits14": gazelle_cgaf_dinov2_vits14,
        "gazelle_cgaf_dinov2_vitb14": gazelle_cgaf_dinov2_vitb14,
        "gazelle_cgaf_dinov2_vitl14": gazelle_cgaf_dinov2_vitl14,
        "gazelle_cgaf_dinov2_vitb14_inout": gazelle_cgaf_dinov2_vitb14_inout,
        "gazelle_cgaf_dinov2_vitl14_inout": gazelle_cgaf_dinov2_vitl14_inout,
    }
    assert model_name in factory, "invalid model name"
    return factory[model_name]()

def gazelle_cgaf_dinov2_vits14():
    backbone = DinoV2Backbone('dinov2_vits14')
    transform = backbone.get_transform((448, 448))
    return GazeLLE(backbone), transform

def gazelle_cgaf_dinov2_vitb14():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    return GazeLLE(backbone), transform

def gazelle_cgaf_dinov2_vitl14():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    return GazeLLE(backbone), transform

def gazelle_cgaf_dinov2_vitb14_inout():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    return GazeLLE(backbone, inout=True), transform

def gazelle_cgaf_dinov2_vitl14_inout():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    return GazeLLE(backbone, inout=True), transform
