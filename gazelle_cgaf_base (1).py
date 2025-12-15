# 改进 CGA
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
import math
import torchvision
import gazeloom.utils as utils
from gazeloom.backbone import DinoV2Backbone
from einops.layers.torch import Rearrange

# === CGA 注意力模块 ===
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

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)
        pattn1 = pattn1.unsqueeze(dim=2)
        x2 = torch.cat([x, pattn1], dim=2)
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

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

# === 特征融合模块替换为 CGAFusion ===
class FeatureWeightedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight_image = nn.Parameter(torch.tensor(0.5))
        self.weight_depth = nn.Parameter(torch.tensor(0.5))

    def forward(self, image_features, depth_features):
        return image_features * self.weight_image + depth_features * self.weight_depth

class FeatureCalibration(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        return features * self.sigmoid(self.conv(features))

class FeatureEnhancement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        return features + self.conv2(self.relu(self.conv1(features)))

class ModalityInteraction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.depth_to_image = nn.Conv2d(1, dim, 1)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, image_feat, depth_feat):
        B, C, H, W = image_feat.shape
        depth_feat = self.depth_to_image(depth_feat)
        image_flat = image_feat.flatten(2).permute(0, 2, 1)
        depth_flat = depth_feat.flatten(2).permute(0, 2, 1)
        q = self.norm1(image_flat)
        k = self.norm1(depth_flat)
        v = self.norm1(depth_flat)
        attn_output, _ = self.attn(q, k, v)
        fused = image_flat + attn_output
        fused = fused + self.ffn(self.norm2(fused))
        out = fused.permute(0, 2, 1).reshape(B, C, H, W)
        return out, depth_feat

class GazeLoom(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.linear = nn.Conv2d(backbone.get_dimension(), dim, 1)
        self.head_token = nn.Embedding(1, dim)
        if inout:
            self.inout_token = nn.Embedding(1, dim)
        self.transformer = nn.Sequential(*[
            Block(dim=dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
            for _ in range(num_layers)
        ])
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 2, 2),
            nn.Conv2d(dim, 1, 1, bias=False),
            nn.Sigmoid()
        )
        if inout:
            self.inout_head = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        self.interaction = ModalityInteraction(dim)
        self.fusion = FeatureWeightedFusion(dim)
        self.calibration = FeatureCalibration(dim)
        self.enhancement = FeatureEnhancement(dim)
        self.attention_fusion = CGAFusion(dim)
        self._pos_cache = {}

    def get_depth_guided_positional_encoding(self, depth_features):
        B, _, H, W = depth_features.shape
        key = (H, W)
        if key not in self._pos_cache:
            self._pos_cache[key] = positionalencoding2d(self.dim, H, W).to(depth_features.device)
        return self._pos_cache[key]

    def forward(self, input):
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        x = self.linear(self.backbone(input["images"]))
        depth_map = input["depth_map"]
        depth_features = F.interpolate(depth_map, size=x.shape[2:], mode='bilinear', align_corners=False)
        x, depth_features = self.interaction(x, depth_features)
        x = self.fusion(x, depth_features)
        x = self.calibration(x)
        x = self.enhancement(x)
        x = self.attention_fusion(x, depth_features)
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

    def get_input_head_maps(self, bboxes):
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                head_map = torch.zeros((self.featmap_h, self.featmap_w))
                if bbox is not None:
                    xmin, ymin, xmax, ymax = bbox
                    xmin = round(xmin * self.featmap_w)
                    ymin = round(ymin * self.featmap_h)
                    xmax = round(xmax * self.featmap_w)
                    ymax = round(ymax * self.featmap_h)
                    head_map[ymin:ymax, xmin:xmax] = 1
                img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps

    def get_gazelle_state_dict(self, include_backbone=False):
        return self.state_dict() if include_backbone else {
            k: v for k, v in self.state_dict().items() if not k.startswith("backbone")
        }

    def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
        current_state_dict = self.state_dict()
        keys1 = set(k for k in current_state_dict if include_backbone or not k.startswith("backbone"))
        keys2 = set(k for k in ckpt_state_dict if include_backbone or not k.startswith("backbone"))
        for k in keys1 & keys2:
            current_state_dict[k] = ckpt_state_dict[k]
        self.load_state_dict(current_state_dict, strict=False)

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

def get_gazeloom_model(model_name):
    factory = {
        "gazeloom_cgaf_dinov2_vits14": gazeloom_cgaf_dinov2_vits14,
        "gazeloom_cgaf_dinov2_vitb14": gazeloom_cgaf_dinov2_vitb14,
        "gazeloom_cgaf_dinov2_vitl14": gazeloom_cgaf_dinov2_vitl14,
        "gazeloom_cgaf_dinov2_vitb14_inout": gazeloom_cgaf_dinov2_vitb14_inout,
        "gazeloom_cgaf_dinov2_vitl14_inout": gazeloom_cgaf_dinov2_vitl14_inout,
    }
    assert model_name in factory, "invalid model name"
    return factory[model_name]()

def gazeloom_cgaf_dinov2_vits14():
    backbone = DinoV2Backbone('dinov2_vits14')
    transform = backbone.get_transform((448, 448))
    return GazeLoom(backbone), transform

def gazeloom_cgaf_dinov2_vitb14():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    return GazeLoom(backbone), transform

def gazeloom_cgaf_dinov2_vitl14():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    return GazeLoom(backbone), transform

def gazeloom_cgaf_dinov2_vitb14_inout():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    return GazeLoom(backbone, inout=True), transform

def gazeloom_cgaf_dinov2_vitl14_inout():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    return GazeLoom(backbone, inout=True), transform
