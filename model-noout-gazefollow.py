# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.vision_transformer import Block
# import math
# import torchvision
# import gazelle.utils as utils
# from gazelle.backbone import DinoV2Backbone


# # === 特征加权融合模块 ===
# class FeatureWeightedFusion(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.weight_image = nn.Parameter(torch.tensor(0.5))
#         self.weight_depth = nn.Parameter(torch.tensor(0.5))

#     def forward(self, image_features, depth_features):
#         return image_features * self.weight_image + depth_features * self.weight_depth


# # === 特征校准模块：通过1x1卷积+Sigmoid调整特征重要性 ===
# class FeatureCalibration(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv = nn.Conv2d(dim, dim, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, features):
#         return features * self.sigmoid(self.conv(features))


# # === 特征增强模块：使用残差结构提升局部表达 ===
# class FeatureEnhancement(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, features):
#         return features + self.conv2(self.relu(self.conv1(features)))


# # === 注意力融合模块：基于图像特征生成注意力 mask 融合模态 ===
# class AttentionFusion(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.attention = nn.Sequential(
#             nn.Conv2d(dim, 1, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, image_features, depth_features):
#         att = self.attention(image_features)
#         return image_features * att + depth_features * (1 - att)


# # === 图像与深度模态交互模块 ===
# class ModalityInteraction(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.depth_to_image = nn.Conv2d(1, dim, 1)
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

#     def forward(self, image_feat, depth_feat):
#         B, C, H, W = image_feat.shape
#         depth_feat = self.depth_to_image(depth_feat)

#         # flatten to (B, N, C)
#         image_flat = image_feat.flatten(2).permute(0, 2, 1)
#         depth_flat = depth_feat.flatten(2).permute(0, 2, 1)

#         # 双向注意力
#         img_out, _ = self.attn(image_flat, depth_flat, depth_flat)
#         dep_out, _ = self.attn(depth_flat, image_flat, image_flat)

#         # reshape back
#         image_feat = image_feat + img_out.permute(0, 2, 1).reshape(B, C, H, W)
#         depth_feat = depth_feat + dep_out.permute(0, 2, 1).reshape(B, C, H, W)
#         return image_feat, depth_feat


# # === 主模型 ===
# class GazeLLE(nn.Module):
#     def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
#         super().__init__()
#         self.backbone = backbone
#         self.dim = dim
#         self.in_size = in_size
#         self.out_size = out_size
#         self.inout = inout
#         self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)

#         self.linear = nn.Conv2d(backbone.get_dimension(), dim, 1)
#         self.head_token = nn.Embedding(1, dim)

#         if inout:
#             self.inout_token = nn.Embedding(1, dim)

#         self.transformer = nn.Sequential(*[
#             Block(dim=dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
#             for _ in range(num_layers)
#         ])

#         self.heatmap_head = nn.Sequential(
#             nn.ConvTranspose2d(dim, dim, 2, 2),
#             nn.Conv2d(dim, 1, 1, bias=False),
#             nn.Sigmoid()
#         )

#         if inout:
#             self.inout_head = nn.Sequential(
#                 nn.Linear(dim, 128),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(128, 1),
#                 nn.Sigmoid()
#             )

#         self.interaction = ModalityInteraction(dim)
#         self.fusion = FeatureWeightedFusion(dim)
#         self.calibration = FeatureCalibration(dim)
#         self.enhancement = FeatureEnhancement(dim)
#         self.attention_fusion = AttentionFusion(dim)
#         self._pos_cache = {}

#     def get_depth_guided_positional_encoding(self, depth_features):
#         B, _, H, W = depth_features.shape
#         key = (H, W)
#         if key not in self._pos_cache:
#             self._pos_cache[key] = positionalencoding2d(self.dim, H, W).to(depth_features.device)
#         return self._pos_cache[key]

#     def forward(self, input):
#         num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
#         x = self.linear(self.backbone(input["images"]))
#         depth_map = input["depth_map"]
#         depth_features = F.interpolate(depth_map, size=x.shape[2:], mode='bilinear', align_corners=False)

#         # 图像/深度交互融合
#         x, depth_features = self.interaction(x, depth_features)
#         x = self.fusion(x, depth_features)
#         x = self.calibration(x)
#         x = self.enhancement(x)
#         x = self.attention_fusion(x, depth_features)

#         # 添加位置编码
#         pos_embed = self.get_depth_guided_positional_encoding(depth_features)
#         x = x + pos_embed

#         # 重复 batch 以匹配 head maps 数量
#         x = utils.repeat_tensors(x, num_ppl_per_img)

#         # 加入 head token 引导
#         head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
#         head_map_embeddings = head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
#         x = x + head_map_embeddings

#         x = x.flatten(start_dim=2).permute(0, 2, 1)

#         if self.inout:
#             x = torch.cat([self.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1), x], dim=1)

#         x = self.transformer(x)

#         if self.inout:
#             inout_tokens = x[:, 0, :]
#             inout_preds = self.inout_head(inout_tokens).squeeze(-1)
#             inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
#             x = x[:, 1:, :]

#         x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
#         x = self.heatmap_head(x).squeeze(1)
#         x = torchvision.transforms.functional.resize(x, self.out_size)
#         heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

#         return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

#     def get_input_head_maps(self, bboxes):
#         head_maps = []
#         for bbox_list in bboxes:
#             img_head_maps = []
#             for bbox in bbox_list:
#                 head_map = torch.zeros((self.featmap_h, self.featmap_w))
#                 if bbox is not None:
#                     xmin, ymin, xmax, ymax = bbox
#                     xmin = round(xmin * self.featmap_w)
#                     ymin = round(ymin * self.featmap_h)
#                     xmax = round(xmax * self.featmap_w)
#                     ymax = round(ymax * self.featmap_h)
#                     head_map[ymin:ymax, xmin:xmax] = 1
#                 img_head_maps.append(head_map)
#             head_maps.append(torch.stack(img_head_maps))
#         return head_maps

#     def get_gazelle_state_dict(self, include_backbone=False):
#         return self.state_dict() if include_backbone else {
#             k: v for k, v in self.state_dict().items() if not k.startswith("backbone")
#         }

#     def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
#         current_state_dict = self.state_dict()
#         keys1 = set(k for k in current_state_dict if include_backbone or not k.startswith("backbone"))
#         keys2 = set(k for k in ckpt_state_dict if include_backbone or not k.startswith("backbone"))
#         for k in keys1 & keys2:
#             current_state_dict[k] = ckpt_state_dict[k]
#         self.load_state_dict(current_state_dict, strict=False)


# # === 2D位置编码 ===
# def positionalencoding2d(d_model, height, width):
#     if d_model % 4 != 0:
#         raise ValueError("位置编码要求 d_model 为4的倍数")
#     pe = torch.zeros(d_model, height, width)
#     d_model = d_model // 2
#     div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
#     pos_w = torch.arange(0., width).unsqueeze(1)
#     pos_h = torch.arange(0., height).unsqueeze(1)
#     pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     return pe


# # === 工厂方法：根据名称构建模型 ===
# def get_gazelle_model(model_name):
#     factory = {
#         "gazelle_dinov2_vits14": gazelle_dinov2_vits14,
#         "gazelle_dinov2_vitb14": gazelle_dinov2_vitb14,
#         "gazelle_dinov2_vitl14": gazelle_dinov2_vitl14,
#         "gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout,
#         "gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout,
#     }
#     assert model_name in factory, "invalid model name"
#     return factory[model_name]()


# def gazelle_dinov2_vits14():
#     backbone = DinoV2Backbone('dinov2_vits14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform


# def gazelle_dinov2_vitb14():
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform


# def gazelle_dinov2_vitl14():
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform


# def gazelle_dinov2_vitb14_inout():
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone, inout=True), transform


# def gazelle_dinov2_vitl14_inout():
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone, inout=True), transform

# 2.  CBAM 注意力模块

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.vision_transformer import Block
# import math
# import torchvision
# import gazelle.utils as utils
# from gazelle.backbone import DinoV2Backbone

# # === CBAM 注意力模块 ===
# class CBAM(nn.Module):
#     def __init__(self, dim, reduction=16, kernel_size=7):
#         super().__init__()
#         self.channel_attn = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim, dim // reduction, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(dim // reduction, dim, 1, bias=False),
#             nn.Sigmoid()
#         )
#         self.spatial_attn = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x_ca = self.channel_attn(x) * x
#         max_pool = torch.max(x_ca, dim=1, keepdim=True)[0]
#         avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
#         sa_input = torch.cat([max_pool, avg_pool], dim=1)
#         x_sa = self.spatial_attn(sa_input) * x_ca
#         return x_sa


# # === 特征融合模块 ===
# class FeatureWeightedFusion(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.weight_image = nn.Parameter(torch.tensor(0.5))
#         self.weight_depth = nn.Parameter(torch.tensor(0.5))

#     def forward(self, image_features, depth_features):
#         return image_features * self.weight_image + depth_features * self.weight_depth


# class FeatureCalibration(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv = nn.Conv2d(dim, dim, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, features):
#         return features * self.sigmoid(self.conv(features))


# class FeatureEnhancement(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, features):
#         return features + self.conv2(self.relu(self.conv1(features)))


# class AttentionFusion(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.attn = CBAM(dim)

#     def forward(self, image_features, depth_features):
#         fused = image_features + depth_features
#         return self.attn(fused)


# class ModalityInteraction(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.depth_to_image = nn.Conv2d(1, dim, 1)
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.ReLU(),
#             nn.Linear(dim * 4, dim)
#         )
#         self.norm2 = nn.LayerNorm(dim)

#     def forward(self, image_feat, depth_feat):
#         B, C, H, W = image_feat.shape
#         depth_feat = self.depth_to_image(depth_feat)
#         image_flat = image_feat.flatten(2).permute(0, 2, 1)
#         depth_flat = depth_feat.flatten(2).permute(0, 2, 1)
#         q = self.norm1(image_flat)
#         k = self.norm1(depth_flat)
#         v = self.norm1(depth_flat)
#         attn_output, _ = self.attn(q, k, v)
#         fused = image_flat + attn_output
#         fused = fused + self.ffn(self.norm2(fused))
#         out = fused.permute(0, 2, 1).reshape(B, C, H, W)
#         return out, depth_feat


# class GazeLLE(nn.Module):
#     def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
#         super().__init__()
#         self.backbone = backbone
#         self.dim = dim
#         self.in_size = in_size
#         self.out_size = out_size
#         self.inout = inout
#         self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
#         self.linear = nn.Conv2d(backbone.get_dimension(), dim, 1)
#         self.head_token = nn.Embedding(1, dim)
#         if inout:
#             self.inout_token = nn.Embedding(1, dim)
#         self.transformer = nn.Sequential(*[
#             Block(dim=dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
#             for _ in range(num_layers)
#         ])
#         self.heatmap_head = nn.Sequential(
#             nn.ConvTranspose2d(dim, dim, 2, 2),
#             nn.Conv2d(dim, 1, 1, bias=False),
#             nn.Sigmoid()
#         )
#         if inout:
#             self.inout_head = nn.Sequential(
#                 nn.Linear(dim, 128),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(128, 1),
#                 nn.Sigmoid()
#             )
#         self.interaction = ModalityInteraction(dim)
#         self.fusion = FeatureWeightedFusion(dim)
#         self.calibration = FeatureCalibration(dim)
#         self.enhancement = FeatureEnhancement(dim)
#         self.attention_fusion = AttentionFusion(dim)
#         self._pos_cache = {}

#     def get_depth_guided_positional_encoding(self, depth_features):
#         B, _, H, W = depth_features.shape
#         key = (H, W)
#         if key not in self._pos_cache:
#             self._pos_cache[key] = positionalencoding2d(self.dim, H, W).to(depth_features.device)
#         return self._pos_cache[key]

#     def forward(self, input):
#         num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
#         x = self.linear(self.backbone(input["images"]))
#         depth_map = input["depth_map"]
#         depth_features = F.interpolate(depth_map, size=x.shape[2:], mode='bilinear', align_corners=False)
#         x, depth_features = self.interaction(x, depth_features)
#         x = self.fusion(x, depth_features)
#         x = self.calibration(x)
#         x = self.enhancement(x)
#         x = self.attention_fusion(x, depth_features)
#         pos_embed = self.get_depth_guided_positional_encoding(depth_features)
#         x = x + pos_embed
#         x = utils.repeat_tensors(x, num_ppl_per_img)
#         head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
#         head_map_embeddings = head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
#         x = x + head_map_embeddings
#         x = x.flatten(start_dim=2).permute(0, 2, 1)
#         if self.inout:
#             x = torch.cat([self.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1), x], dim=1)
#         x = self.transformer(x)
#         if self.inout:
#             inout_tokens = x[:, 0, :]
#             inout_preds = self.inout_head(inout_tokens).squeeze(-1)
#             inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
#             x = x[:, 1:, :]
#         x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
#         x = self.heatmap_head(x).squeeze(1)
#         x = torchvision.transforms.functional.resize(x, self.out_size)
#         heatmap_preds = utils.split_tensors(x, num_ppl_per_img)
#         return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

#     def get_input_head_maps(self, bboxes):
#         head_maps = []
#         for bbox_list in bboxes:
#             img_head_maps = []
#             for bbox in bbox_list:
#                 head_map = torch.zeros((self.featmap_h, self.featmap_w))
#                 if bbox is not None:
#                     xmin, ymin, xmax, ymax = bbox
#                     xmin = round(xmin * self.featmap_w)
#                     ymin = round(ymin * self.featmap_h)
#                     xmax = round(xmax * self.featmap_w)
#                     ymax = round(ymax * self.featmap_h)
#                     head_map[ymin:ymax, xmin:xmax] = 1
#                 img_head_maps.append(head_map)
#             head_maps.append(torch.stack(img_head_maps))
#         return head_maps

#     def get_gazelle_state_dict(self, include_backbone=False):
#         return self.state_dict() if include_backbone else {
#             k: v for k, v in self.state_dict().items() if not k.startswith("backbone")
#         }

#     def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
#         current_state_dict = self.state_dict()
#         keys1 = set(k for k in current_state_dict if include_backbone or not k.startswith("backbone"))
#         keys2 = set(k for k in ckpt_state_dict if include_backbone or not k.startswith("backbone"))
#         for k in keys1 & keys2:
#             current_state_dict[k] = ckpt_state_dict[k]
#         self.load_state_dict(current_state_dict, strict=False)


# def positionalencoding2d(d_model, height, width):
#     if d_model % 4 != 0:
#         raise ValueError("位置编码要求 d_model 为4的倍数")
#     pe = torch.zeros(d_model, height, width)
#     d_model = d_model // 2
#     div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
#     pos_w = torch.arange(0., width).unsqueeze(1)
#     pos_h = torch.arange(0., height).unsqueeze(1)
#     pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     return pe


# def get_gazelle_model(model_name):
#     factory = {
#         "gazelle_dinov2_vits14": gazelle_dinov2_vits14,
#         "gazelle_dinov2_vitb14": gazelle_dinov2_vitb14,
#         "gazelle_dinov2_vitl14": gazelle_dinov2_vitl14,
#         "gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout,
#         "gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout,
#     }
#     assert model_name in factory, "invalid model name"
#     return factory[model_name]()


# def gazelle_dinov2_vits14():
#     backbone = DinoV2Backbone('dinov2_vits14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform


# def gazelle_dinov2_vitb14():
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform


# def gazelle_dinov2_vitl14():
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform


# def gazelle_dinov2_vitb14_inout():
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone, inout=True), transform


# def gazelle_dinov2_vitl14_inout():
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone, inout=True), transform


# 3.CGA

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.vision_transformer import Block
# import math
# import torchvision
# import gazelle.utils as utils
# from gazelle.backbone import DinoV2Backbone
# from einops.layers.torch import Rearrange

# # === CGA 注意力模块 ===
# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)
 
#     def forward(self, x):
#         x_avg = torch.mean(x, dim=1, keepdim=True)
#         x_max, _ = torch.max(x, dim=1, keepdim=True)
#         x2 = torch.cat([x_avg, x_max], dim=1)
#         sattn = self.sa(x2)
#         return sattn

# class ChannelAttention(nn.Module):
#     def __init__(self, dim, reduction=8):
#         super(ChannelAttention, self).__init__()
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.ca = nn.Sequential(
#             nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
#         )

#     def forward(self, x):
#         x_gap = self.gap(x)
#         cattn = self.ca(x_gap)
#         return cattn

# class PixelAttention(nn.Module):
#     def __init__(self, dim):
#         super(PixelAttention, self).__init__()
#         self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, pattn1):
#         B, C, H, W = x.shape
#         x = x.unsqueeze(dim=2)
#         pattn1 = pattn1.unsqueeze(dim=2)
#         x2 = torch.cat([x, pattn1], dim=2)
#         x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
#         pattn2 = self.pa2(x2)
#         pattn2 = self.sigmoid(pattn2)
#         return pattn2

# class CGAFusion(nn.Module):
#     def __init__(self, dim, reduction=8):
#         super(CGAFusion, self).__init__()
#         self.sa = SpatialAttention()
#         self.ca = ChannelAttention(dim, reduction)
#         self.pa = PixelAttention(dim)
#         self.conv = nn.Conv2d(dim, dim, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, y):
#         initial = x + y
#         cattn = self.ca(initial)
#         sattn = self.sa(initial)
#         pattn1 = sattn + cattn
#         pattn2 = self.sigmoid(self.pa(initial, pattn1))
#         result = initial + pattn2 * x + (1 - pattn2) * y
#         result = self.conv(result)
#         return result

# # === 特征融合模块替换为 CGAFusion ===
# class FeatureWeightedFusion(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.weight_image = nn.Parameter(torch.tensor(0.5))
#         self.weight_depth = nn.Parameter(torch.tensor(0.5))

#     def forward(self, image_features, depth_features):
#         return image_features * self.weight_image + depth_features * self.weight_depth

# class FeatureCalibration(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv = nn.Conv2d(dim, dim, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, features):
#         return features * self.sigmoid(self.conv(features))

# class FeatureEnhancement(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, features):
#         return features + self.conv2(self.relu(self.conv1(features)))

# class ModalityInteraction(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.depth_to_image = nn.Conv2d(1, dim, 1)
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.ReLU(),
#             nn.Linear(dim * 4, dim)
#         )
#         self.norm2 = nn.LayerNorm(dim)

#     def forward(self, image_feat, depth_feat):
#         B, C, H, W = image_feat.shape
#         depth_feat = self.depth_to_image(depth_feat)
#         image_flat = image_feat.flatten(2).permute(0, 2, 1)
#         depth_flat = depth_feat.flatten(2).permute(0, 2, 1)
#         q = self.norm1(image_flat)
#         k = self.norm1(depth_flat)
#         v = self.norm1(depth_flat)
#         attn_output, _ = self.attn(q, k, v)
#         fused = image_flat + attn_output
#         fused = fused + self.ffn(self.norm2(fused))
#         out = fused.permute(0, 2, 1).reshape(B, C, H, W)
#         return out, depth_feat

# class GazeLLE(nn.Module):
#     def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
#         super().__init__()
#         self.backbone = backbone
#         self.dim = dim
#         self.in_size = in_size
#         self.out_size = out_size
#         self.inout = inout
#         self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
#         self.linear = nn.Conv2d(backbone.get_dimension(), dim, 1)
#         self.head_token = nn.Embedding(1, dim)
#         if inout:
#             self.inout_token = nn.Embedding(1, dim)
#         self.transformer = nn.Sequential(*[
#             Block(dim=dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
#             for _ in range(num_layers)
#         ])
#         self.heatmap_head = nn.Sequential(
#             nn.ConvTranspose2d(dim, dim, 2, 2),
#             nn.Conv2d(dim, 1, 1, bias=False),
#             nn.Sigmoid()
#         )
#         if inout:
#             self.inout_head = nn.Sequential(
#                 nn.Linear(dim, 128),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(128, 1),
#                 nn.Sigmoid()
#             )
#         self.interaction = ModalityInteraction(dim)
#         self.fusion = FeatureWeightedFusion(dim)
#         self.calibration = FeatureCalibration(dim)
#         self.enhancement = FeatureEnhancement(dim)
#         self.attention_fusion = CGAFusion(dim)
#         self._pos_cache = {}

#     def get_depth_guided_positional_encoding(self, depth_features):
#         B, _, H, W = depth_features.shape
#         key = (H, W)
#         if key not in self._pos_cache:
#             self._pos_cache[key] = positionalencoding2d(self.dim, H, W).to(depth_features.device)
#         return self._pos_cache[key]

#     def forward(self, input):
#         num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
#         x = self.linear(self.backbone(input["images"]))
#         depth_map = input["depth_map"]
#         depth_features = F.interpolate(depth_map, size=x.shape[2:], mode='bilinear', align_corners=False)
#         x, depth_features = self.interaction(x, depth_features)
#         x = self.fusion(x, depth_features)
#         x = self.calibration(x)
#         x = self.enhancement(x)
#         x = self.attention_fusion(x, depth_features)
#         pos_embed = self.get_depth_guided_positional_encoding(depth_features)
#         x = x + pos_embed
#         x = utils.repeat_tensors(x, num_ppl_per_img)
#         head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
#         head_map_embeddings = head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
#         x = x + head_map_embeddings
#         x = x.flatten(start_dim=2).permute(0, 2, 1)
#         if self.inout:
#             x = torch.cat([self.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1), x], dim=1)
#         x = self.transformer(x)
#         if self.inout:
#             inout_tokens = x[:, 0, :]
#             inout_preds = self.inout_head(inout_tokens).squeeze(-1)
#             inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
#             x = x[:, 1:, :]
#         x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
#         x = self.heatmap_head(x).squeeze(1)
#         x = torchvision.transforms.functional.resize(x, self.out_size)
#         heatmap_preds = utils.split_tensors(x, num_ppl_per_img)
#         return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

#     def get_input_head_maps(self, bboxes):
#         head_maps = []
#         for bbox_list in bboxes:
#             img_head_maps = []
#             for bbox in bbox_list:
#                 head_map = torch.zeros((self.featmap_h, self.featmap_w))
#                 if bbox is not None:
#                     xmin, ymin, xmax, ymax = bbox
#                     xmin = round(xmin * self.featmap_w)
#                     ymin = round(ymin * self.featmap_h)
#                     xmax = round(xmax * self.featmap_w)
#                     ymax = round(ymax * self.featmap_h)
#                     head_map[ymin:ymax, xmin:xmax] = 1
#                 img_head_maps.append(head_map)
#             head_maps.append(torch.stack(img_head_maps))
#         return head_maps

#     def get_gazelle_state_dict(self, include_backbone=False):
#         return self.state_dict() if include_backbone else {
#             k: v for k, v in self.state_dict().items() if not k.startswith("backbone")
#         }

#     def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
#         current_state_dict = self.state_dict()
#         keys1 = set(k for k in current_state_dict if include_backbone or not k.startswith("backbone"))
#         keys2 = set(k for k in ckpt_state_dict if include_backbone or not k.startswith("backbone"))
#         for k in keys1 & keys2:
#             current_state_dict[k] = ckpt_state_dict[k]
#         self.load_state_dict(current_state_dict, strict=False)

# def positionalencoding2d(d_model, height, width):
#     if d_model % 4 != 0:
#         raise ValueError("位置编码要求 d_model 为4的倍数")
#     pe = torch.zeros(d_model, height, width)
#     d_model = d_model // 2
#     div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
#     pos_w = torch.arange(0., width).unsqueeze(1)
#     pos_h = torch.arange(0., height).unsqueeze(1)
#     pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     return pe

# def get_gazelle_model(model_name):
#     factory = {
#         "gazelle_cgaf_dinov2_vits14": gazelle_cgaf_dinov2_vits14,
#         "gazelle_cgaf_dinov2_vitb14": gazelle_cgaf_dinov2_vitb14,
#         "gazelle_cgaf_dinov2_vitl14": gazelle_cgaf_dinov2_vitl14,
#         "gazelle_cgaf_dinov2_vitb14_inout": gazelle_cgaf_dinov2_vitb14_inout,
#         "gazelle_cgaf_dinov2_vitl14_inout": gazelle_cgaf_dinov2_vitl14_inout,
#     }
#     assert model_name in factory, "invalid model name"
#     return factory[model_name]()

# def gazelle_cgaf_dinov2_vits14():
#     backbone = DinoV2Backbone('dinov2_vits14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform

# def gazelle_cgaf_dinov2_vitb14():
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform

# def gazelle_cgaf_dinov2_vitl14():
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform

# def gazelle_cgaf_dinov2_vitb14_inout():
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone, inout=True), transform

# def gazelle_cgaf_dinov2_vitl14_inout():
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone, inout=True), transform

# 改2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import torchvision
# from timm.models.vision_transformer import Block
# from einops.layers.torch import Rearrange
# import gazelle.utils as utils
# from gazelle.backbone import DinoV2Backbone

# # === SpatialAttention ===
# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

#     def forward(self, x):
#         x_avg = torch.mean(x, dim=1, keepdim=True)
#         x_max, _ = torch.max(x, dim=1, keepdim=True)
#         x2 = torch.cat([x_avg, x_max], dim=1)
#         sattn = self.sa(x2)
#         return sattn

# # === ChannelAttention ===
# class ChannelAttention(nn.Module):
#     def __init__(self, dim, reduction=8):
#         super(ChannelAttention, self).__init__()
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.ca = nn.Sequential(
#             nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
#         )

#     def forward(self, x):
#         x_gap = self.gap(x)
#         cattn = self.ca(x_gap)
#         return cattn

# # === PixelAttention ===
# class PixelAttention(nn.Module):
#     def __init__(self, dim):
#         super(PixelAttention, self).__init__()
#         self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, pattn1):
#         x = x.unsqueeze(dim=2)
#         pattn1 = pattn1.unsqueeze(dim=2)
#         x2 = torch.cat([x, pattn1], dim=2)
#         x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
#         pattn2 = self.pa2(x2)
#         pattn2 = self.sigmoid(pattn2)
#         return pattn2

# # === CGAFusion 模块 ===
# class CGAFusion(nn.Module):
#     def __init__(self, dim, reduction=8):
#         super(CGAFusion, self).__init__()
#         self.sa = SpatialAttention()
#         self.ca = ChannelAttention(dim, reduction)
#         self.pa = PixelAttention(dim)
#         self.conv = nn.Conv2d(dim, dim, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, y):
#         initial = x + y
#         cattn = self.ca(initial)
#         sattn = self.sa(initial)
#         pattn1 = sattn + cattn
#         pattn2 = self.sigmoid(self.pa(initial, pattn1))
#         result = initial + pattn2 * x + (1 - pattn2) * y
#         result = self.conv(result)
#         return result

# # === ModalityInteraction 模块 ===
# class ModalityInteraction(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.depth_to_image = nn.Conv2d(1, dim, 1)
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.ReLU(),
#             nn.Linear(dim * 4, dim)
#         )
#         self.norm2 = nn.LayerNorm(dim)

#     def forward(self, image_feat, depth_feat):
#         B, C, H, W = image_feat.shape
#         depth_feat = self.depth_to_image(depth_feat)
#         image_flat = image_feat.flatten(2).permute(0, 2, 1)
#         depth_flat = depth_feat.flatten(2).permute(0, 2, 1)
#         q = self.norm1(image_flat)
#         k = self.norm1(depth_flat)
#         v = self.norm1(depth_flat)
#         attn_output, _ = self.attn(q, k, v)
#         fused = image_flat + attn_output
#         fused = fused + self.ffn(self.norm2(fused))
#         out = fused.permute(0, 2, 1).reshape(B, C, H, W)
#         return out, depth_feat

# # === 融合与增强模块 ===
# class FeatureWeightedFusion(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.weight_image = nn.Parameter(torch.tensor(0.5))
#         self.weight_depth = nn.Parameter(torch.tensor(0.5))

#     def forward(self, image_features, depth_features):
#         return image_features * self.weight_image + depth_features * self.weight_depth

# class FeatureCalibration(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv = nn.Conv2d(dim, dim, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, features):
#         return features * self.sigmoid(self.conv(features))

# class FeatureEnhancement(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, features):
#         return features + self.conv2(self.relu(self.conv1(features)))

# # === GazeLLE 主网络 ===
# class GazeLLE(nn.Module):
#     def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
#         super().__init__()
#         self.backbone = backbone
#         self.dim = dim
#         self.in_size = in_size
#         self.out_size = out_size
#         self.inout = inout
#         self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
#         self.linear = nn.Conv2d(backbone.get_dimension(), dim, 1)
#         self.head_token = nn.Embedding(1, dim)
#         if inout:
#             self.inout_token = nn.Embedding(1, dim)
#         self.transformer = nn.Sequential(*[
#             Block(dim=dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
#             for _ in range(num_layers)
#         ])
#         self.heatmap_head = nn.Sequential(
#             nn.ConvTranspose2d(dim, dim, 2, 2),
#             nn.Conv2d(dim, 1, 1, bias=False),
#             nn.Sigmoid()
#         )
#         if inout:
#             self.inout_head = nn.Sequential(
#                 nn.Linear(dim, 128),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(128, 1),
#                 nn.Sigmoid()
#             )
#         self.interaction = ModalityInteraction(dim)
#         self.fusion = FeatureWeightedFusion(dim)
#         self.calibration = FeatureCalibration(dim)
#         self.enhancement = FeatureEnhancement(dim)
#         self.attention_fusion = CGAFusion(dim)
#         self._pos_cache = {}

#     def get_depth_guided_positional_encoding(self, depth_features):
#         B, _, H, W = depth_features.shape
#         key = (H, W)
#         if key not in self._pos_cache:
#             self._pos_cache[key] = positionalencoding2d(self.dim, H, W).to(depth_features.device)
#         return self._pos_cache[key]

#     def forward(self, input):
#         num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
#         x = self.linear(self.backbone(input["images"]))
#         depth_map = input["depth_map"]
#         depth_features = F.interpolate(depth_map, size=x.shape[2:], mode='bilinear', align_corners=False)
#         x, depth_features = self.interaction(x, depth_features)
#         x = self.fusion(x, depth_features)
#         x = self.calibration(x)
#         x = self.enhancement(x)
#         x = self.attention_fusion(x, depth_features)
#         pos_embed = self.get_depth_guided_positional_encoding(depth_features)
#         x = x + pos_embed
#         x = utils.repeat_tensors(x, num_ppl_per_img)
#         head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
#         head_map_embeddings = head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
#         x = x + head_map_embeddings
#         x = x.flatten(start_dim=2).permute(0, 2, 1)
#         if self.inout:
#             x = torch.cat([self.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1), x], dim=1)
#         x = self.transformer(x)
#         if self.inout:
#             inout_tokens = x[:, 0, :]
#             inout_preds = self.inout_head(inout_tokens).squeeze(-1)
#             inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
#             x = x[:, 1:, :]
#         x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
#         x = self.heatmap_head(x).squeeze(1)
#         x = torchvision.transforms.functional.resize(x, self.out_size)
#         heatmap_preds = utils.split_tensors(x, num_ppl_per_img)
#         return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

#     def get_input_head_maps(self, bboxes):
#         head_maps = []
#         for bbox_list in bboxes:
#             img_head_maps = []
#             for bbox in bbox_list:
#                 head_map = torch.zeros((self.featmap_h, self.featmap_w))
#                 if bbox is not None:
#                     xmin, ymin, xmax, ymax = bbox
#                     xmin = round(xmin * self.featmap_w)
#                     ymin = round(ymin * self.featmap_h)
#                     xmax = round(xmax * self.featmap_w)
#                     ymax = round(ymax * self.featmap_h)
#                     head_map[ymin:ymax, xmin:xmax] = 1
#                 img_head_maps.append(head_map)
#             head_maps.append(torch.stack(img_head_maps))
#         return head_maps

#     def get_gazelle_state_dict(self, include_backbone=False):
#         return self.state_dict() if include_backbone else {
#             k: v for k, v in self.state_dict().items() if not k.startswith("backbone")
#         }

#     def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
#         current_state_dict = self.state_dict()
#         keys1 = set(k for k in current_state_dict if include_backbone or not k.startswith("backbone"))
#         keys2 = set(k for k in ckpt_state_dict if include_backbone or not k.startswith("backbone"))
#         for k in keys1 & keys2:
#             current_state_dict[k] = ckpt_state_dict[k]
#         self.load_state_dict(current_state_dict, strict=False)

# # === Positional Encoding ===
# def positionalencoding2d(d_model, height, width):
#     if d_model % 4 != 0:
#         raise ValueError("位置编码要求 d_model 为4的倍数")
#     pe = torch.zeros(d_model, height, width)
#     d_model = d_model // 2
#     div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
#     pos_w = torch.arange(0., width).unsqueeze(1)
#     pos_h = torch.arange(0., height).unsqueeze(1)
#     pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     return pe

# # === 构建模型工厂 ===
# def get_gazelle_model(model_name):
#     factory = {
#         "gazelle_cgaf_dinov2_vits14": gazelle_cgaf_dinov2_vits14,
#         "gazelle_cgaf_dinov2_vitb14": gazelle_cgaf_dinov2_vitb14,
#         "gazelle_cgaf_dinov2_vitl14": gazelle_cgaf_dinov2_vitl14,
#         "gazelle_cgaf_dinov2_vitb14_inout": gazelle_cgaf_dinov2_vitb14_inout,
#         "gazelle_cgaf_dinov2_vitl14_inout": gazelle_cgaf_dinov2_vitl14_inout,
#     }
#     assert model_name in factory, "invalid model name"
#     return factory[model_name]()

# def gazelle_cgaf_dinov2_vits14():
#     backbone = DinoV2Backbone('dinov2_vits14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform

# def gazelle_cgaf_dinov2_vitb14():
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform

# def gazelle_cgaf_dinov2_vitl14():
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform

# def gazelle_cgaf_dinov2_vitb14_inout():
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone, inout=True), transform

# def gazelle_cgaf_dinov2_vitl14_inout():
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone, inout=True), transform



# 修改612
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from einops.layers.torch import Rearrange
# from timm.models.vision_transformer import Block
# import gazelle.utils as utils
# from gazelle.backbone import DinoV2Backbone

# # === HiLo Attention ===
# class HiLo(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#         head_dim = int(dim / num_heads)
#         self.dim = dim
#         self.l_heads = int(num_heads * alpha)
#         self.l_dim = self.l_heads * head_dim
#         self.h_heads = num_heads - self.l_heads
#         self.h_dim = self.h_heads * head_dim
#         self.ws = window_size

#         if self.ws == 1:
#             self.h_heads, self.h_dim = 0, 0
#             self.l_heads, self.l_dim = num_heads, dim

#         self.scale = qk_scale or head_dim ** -0.5

#         if self.l_heads > 0:
#             if self.ws != 1:
#                 self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
#             self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
#             self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
#             self.l_proj = nn.Linear(self.l_dim, self.l_dim)

#         if self.h_heads > 0:
#             self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
#             self.h_proj = nn.Linear(self.h_dim, self.h_dim)

#     def hifi(self, x):
#         B, H, W, C = x.shape
#         h_group, w_group = H // self.ws, W // self.ws
#         total_groups = h_group * w_group
#         x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
#         qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
#         x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)
#         return self.h_proj(x)

#     def lofi(self, x):
#         B, H, W, C = x.shape
#         q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)
#         if self.ws > 1:
#             x_ = self.sr(x.permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)
#             kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
#         else:
#             kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
#         return self.l_proj(x)

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = x.reshape(B, H, W, C)
#         if self.h_heads == 0:
#             return self.lofi(x).reshape(B, N, C)
#         if self.l_heads == 0:
#             return self.hifi(x).reshape(B, N, C)
#         hifi_out = self.hifi(x)
#         lofi_out = self.lofi(x)
#         return torch.cat((hifi_out, lofi_out), dim=-1).reshape(B, N, C)
# # === CGA 注意力模块 ===
# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

#     def forward(self, x):
#         x_avg = torch.mean(x, dim=1, keepdim=True)
#         x_max, _ = torch.max(x, dim=1, keepdim=True)
#         x2 = torch.cat([x_avg, x_max], dim=1)
#         sattn = self.sa(x2)
#         return sattn

# class ChannelAttention(nn.Module):
#     def __init__(self, dim, reduction=8):
#         super(ChannelAttention, self).__init__()
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.ca = nn.Sequential(
#             nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
#         )

#     def forward(self, x):
#         x_gap = self.gap(x)
#         cattn = self.ca(x_gap)
#         return cattn

# class PixelAttention(nn.Module):
#     def __init__(self, dim):
#         super(PixelAttention, self).__init__()
#         self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, pattn1):
#         x = x.unsqueeze(dim=2)
#         pattn1 = pattn1.unsqueeze(dim=2)
#         x2 = torch.cat([x, pattn1], dim=2)
#         x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
#         pattn2 = self.pa2(x2)
#         return self.sigmoid(pattn2)

# class CGAFusion(nn.Module):
#     def __init__(self, dim, reduction=8):
#         super(CGAFusion, self).__init__()
#         self.sa = SpatialAttention()
#         self.ca = ChannelAttention(dim, reduction)
#         self.pa = PixelAttention(dim)
#         self.conv = nn.Conv2d(dim, dim, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, y):
#         initial = x + y
#         cattn = self.ca(initial)
#         sattn = self.sa(initial)
#         pattn1 = sattn + cattn
#         pattn2 = self.sigmoid(self.pa(initial, pattn1))
#         result = initial + pattn2 * x + (1 - pattn2) * y
#         result = self.conv(result)
#         return result

# # === 特征融合增强模块 ===
# class FeatureWeightedFusion(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.weight_image = nn.Parameter(torch.tensor(0.5))
#         self.weight_depth = nn.Parameter(torch.tensor(0.5))

#     def forward(self, image_features, depth_features):
#         return image_features * self.weight_image + depth_features * self.weight_depth

# class FeatureCalibration(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv = nn.Conv2d(dim, dim, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, features):
#         return features * self.sigmoid(self.conv(features))

# class FeatureEnhancement(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, features):
#         return features + self.conv2(self.relu(self.conv1(features)))

# # === ModalityInteraction：集成 HiLo Attention ===
# class ModalityInteraction(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.depth_to_image = nn.Conv2d(1, dim, 1)
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = HiLo(dim=dim, num_heads=8, window_size=2, alpha=0.5)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.ReLU(),
#             nn.Linear(dim * 4, dim)
#         )
#         self.norm2 = nn.LayerNorm(dim)

#     def forward(self, image_feat, depth_feat):
#         B, C, H, W = image_feat.shape
#         depth_feat = self.depth_to_image(depth_feat)
#         image_flat = image_feat.flatten(2).permute(0, 2, 1)  # B, N, C
#         image_flat = self.norm1(image_flat)
#         attn_output = self.attn(image_flat, H, W)
#         fused = image_flat + attn_output
#         fused = fused + self.ffn(self.norm2(fused))
#         out = fused.permute(0, 2, 1).reshape(B, C, H, W)
#         return out, depth_feat
# # === GazeLLE 主网络 ===
# class GazeLLE(nn.Module):
#     def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
#         super().__init__()
#         self.backbone = backbone
#         self.dim = dim
#         self.in_size = in_size
#         self.out_size = out_size
#         self.inout = inout
#         self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
#         self.linear = nn.Conv2d(backbone.get_dimension(), dim, 1)
#         self.head_token = nn.Embedding(1, dim)
#         if inout:
#             self.inout_token = nn.Embedding(1, dim)
#         self.transformer = nn.Sequential(*[
#             Block(dim=dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
#             for _ in range(num_layers)
#         ])
#         self.heatmap_head = nn.Sequential(
#             nn.ConvTranspose2d(dim, dim, 2, 2),
#             nn.Conv2d(dim, 1, 1, bias=False),
#             nn.Sigmoid()
#         )
#         if inout:
#             self.inout_head = nn.Sequential(
#                 nn.Linear(dim, 128),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(128, 1),
#                 nn.Sigmoid()
#             )
#         self.interaction = ModalityInteraction(dim)
#         self.fusion = FeatureWeightedFusion(dim)
#         self.calibration = FeatureCalibration(dim)
#         self.enhancement = FeatureEnhancement(dim)
#         self.attention_fusion = CGAFusion(dim)
#         self._pos_cache = {}

#     def get_depth_guided_positional_encoding(self, depth_features):
#         B, _, H, W = depth_features.shape
#         key = (H, W)
#         if key not in self._pos_cache:
#             self._pos_cache[key] = positionalencoding2d(self.dim, H, W).to(depth_features.device)
#         return self._pos_cache[key]

#     def forward(self, input):
#         num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
#         x = self.linear(self.backbone(input["images"]))
#         depth_map = input["depth_map"]
#         depth_features = F.interpolate(depth_map, size=x.shape[2:], mode='bilinear', align_corners=False)
#         x, depth_features = self.interaction(x, depth_features)
#         x = self.fusion(x, depth_features)
#         x = self.calibration(x)
#         x = self.enhancement(x)
#         x = self.attention_fusion(x, depth_features)
#         pos_embed = self.get_depth_guided_positional_encoding(depth_features)
#         x = x + pos_embed
#         x = utils.repeat_tensors(x, num_ppl_per_img)
#         head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
#         head_map_embeddings = head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
#         x = x + head_map_embeddings
#         x = x.flatten(start_dim=2).permute(0, 2, 1)
#         if self.inout:
#             x = torch.cat([self.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1), x], dim=1)
#         x = self.transformer(x)
#         if self.inout:
#             inout_tokens = x[:, 0, :]
#             inout_preds = self.inout_head(inout_tokens).squeeze(-1)
#             inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
#             x = x[:, 1:, :]
#         x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
#         x = self.heatmap_head(x).squeeze(1)
#         x = torchvision.transforms.functional.resize(x, self.out_size)
#         heatmap_preds = utils.split_tensors(x, num_ppl_per_img)
#         return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

#     def get_input_head_maps(self, bboxes):
#         head_maps = []
#         for bbox_list in bboxes:
#             img_head_maps = []
#             for bbox in bbox_list:
#                 head_map = torch.zeros((self.featmap_h, self.featmap_w))
#                 if bbox is not None:
#                     xmin, ymin, xmax, ymax = bbox
#                     xmin = round(xmin * self.featmap_w)
#                     ymin = round(ymin * self.featmap_h)
#                     xmax = round(xmax * self.featmap_w)
#                     ymax = round(ymax * self.featmap_h)
#                     head_map[ymin:ymax, xmin:xmax] = 1
#                 img_head_maps.append(head_map)
#             head_maps.append(torch.stack(img_head_maps))
#         return head_maps

#     def get_gazelle_state_dict(self, include_backbone=False):
#         return self.state_dict() if include_backbone else {
#             k: v for k, v in self.state_dict().items() if not k.startswith("backbone")
#         }

#     def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
#         current_state_dict = self.state_dict()
#         keys1 = set(k for k in current_state_dict if include_backbone or not k.startswith("backbone"))
#         keys2 = set(k for k in ckpt_state_dict if include_backbone or not k.startswith("backbone"))
#         for k in keys1 & keys2:
#             current_state_dict[k] = ckpt_state_dict[k]
#         self.load_state_dict(current_state_dict, strict=False)

# # === Positional Encoding ===
# def positionalencoding2d(d_model, height, width):
#     if d_model % 4 != 0:
#         raise ValueError("位置编码要求 d_model 为4的倍数")
#     pe = torch.zeros(d_model, height, width)
#     d_model = d_model // 2
#     div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
#     pos_w = torch.arange(0., width).unsqueeze(1)
#     pos_h = torch.arange(0., height).unsqueeze(1)
#     pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     return pe

# # === 模型工厂函数 ===
# def get_gazelle_model(model_name):
#     factory = {
#         "gazelle_cgaf_dinov2_vits14": gazelle_cgaf_dinov2_vits14,
#         "gazelle_cgaf_dinov2_vitb14": gazelle_cgaf_dinov2_vitb14,
#         "gazelle_cgaf_dinov2_vitl14": gazelle_cgaf_dinov2_vitl14,
#         "gazelle_cgaf_dinov2_vitb14_inout": gazelle_cgaf_dinov2_vitb14_inout,
#         "gazelle_cgaf_dinov2_vitl14_inout": gazelle_cgaf_dinov2_vitl14_inout,
#     }
#     assert model_name in factory, "invalid model name"
#     return factory[model_name]()

# def gazelle_cgaf_dinov2_vits14():
#     backbone = DinoV2Backbone('dinov2_vits14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform

# def gazelle_cgaf_dinov2_vitb14():
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform

# def gazelle_cgaf_dinov2_vitl14():
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform

# def gazelle_cgaf_dinov2_vitb14_inout():
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone, inout=True), transform

# def gazelle_cgaf_dinov2_vitl14_inout():
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone, inout=True), transform



# 修改613
# 原来的
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops.layers.torch import Rearrange
# from timm.models.vision_transformer import Block
# import math
# import torchvision
# import torch.utils.checkpoint as checkpoint
# import gazelle.utils as utils
# from gazelle.backbone import DinoV2Backbone

# # === HiLo Attention ===
# class HiLo(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#         head_dim = int(dim / num_heads)
#         self.dim = dim
#         self.l_heads = int(num_heads * alpha)
#         self.l_dim = self.l_heads * head_dim
#         self.h_heads = num_heads - self.l_heads
#         self.h_dim = self.h_heads * head_dim
#         self.ws = window_size

#         if self.ws == 1:
#             self.h_heads, self.h_dim = 0, 0
#             self.l_heads, self.l_dim = num_heads, dim

#         self.scale = qk_scale or head_dim ** -0.5

#         # 低频头的层定义
#         if self.l_heads > 0:
#             if self.ws != 1:
#                 self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
#             self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
#             self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
#             self.l_proj = nn.Linear(self.l_dim, self.l_dim)

#         # 高频头的层定义
#         if self.h_heads > 0:
#             self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
#             self.h_proj = nn.Linear(self.h_dim, self.h_dim)

#     def compute_qkv(self, x, heads, dim, is_hifi=True):
#         B, H, W, C = x.shape
#         if is_hifi:
#             # 高频部分的QKV计算
#             x = x.reshape(B, H // self.ws, self.ws, W // self.ws, self.ws, C).transpose(2, 3)
#             qkv = self.h_qkv(x).reshape(B, H // self.ws * W // self.ws, heads, dim // heads).permute(0, 2, 1, 3)
#         else:
#             # 低频部分的QKV计算
#             qkv = self.l_qkv(x).reshape(B, H * W, heads, dim // heads).permute(0, 2, 1, 3)

#         k, v = qkv[1], qkv[2]
#         return k, v

#     def hifi(self, x):
#         B, H, W, C = x.shape
#         k, v = self.compute_qkv(x, self.h_heads, self.h_dim, is_hifi=True)
#         attn = (k @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = (attn @ v).transpose(2, 3).reshape(B, H // self.ws, W // self.ws, self.ws, self.ws, self.h_dim)
#         x = attn.transpose(2, 3).reshape(B, H // self.ws * self.ws, W // self.ws * self.ws, self.h_dim)
#         return self.h_proj(x)

#     def lofi(self, x):
#         B, H, W, C = x.shape
#         k, v = self.compute_qkv(x, self.l_heads, self.l_dim, is_hifi=False)
#         attn = (k @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
#         return self.l_proj(x)

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = x.reshape(B, H, W, C)
#         if self.h_heads == 0:
#             return checkpoint.checkpoint(self.lofi, x).reshape(B, N, C)
#         if self.l_heads == 0:
#             return checkpoint.checkpoint(self.hifi, x).reshape(B, N, C)
#         hifi_out = checkpoint.checkpoint(self.hifi, x)
#         lofi_out = checkpoint.checkpoint(self.lofi, x)
#         return torch.cat((hifi_out, lofi_out), dim=-1).reshape(B, N, C)


# # === Pixel Attention ===
# class PixelAttention(nn.Module):
#     def __init__(self, dim):
#         super(PixelAttention, self).__init__()
#         self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)

#     def forward(self, x, pattn1):
#         x = x.unsqueeze(dim=2)
#         pattn1 = pattn1.unsqueeze(dim=2)
#         x2 = torch.cat([x, pattn1], dim=2)
#         x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
#         pattn2 = self.pa2(x2)
#         return pattn2  # 移除sigmoid


# # === CGA Attention ===
# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

#     def forward(self, x):
#         x_avg = torch.mean(x, dim=1, keepdim=True)
#         x_max, _ = torch.max(x, dim=1, keepdim=True)
#         x2 = torch.cat([x_avg, x_max], dim=1)
#         sattn = self.sa(x2)
#         return sattn

# class ChannelAttention(nn.Module):
#     def __init__(self, dim, reduction=8):
#         super(ChannelAttention, self).__init__()
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.ca = nn.Sequential(
#             nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
#         )

#     def forward(self, x):
#         x_gap = self.gap(x)
#         cattn = self.ca(x_gap)
#         return cattn

# class CGAFusion(nn.Module):
#     def __init__(self, dim, reduction=8):
#         super(CGAFusion, self).__init__()
#         self.sa = SpatialAttention()
#         self.ca = ChannelAttention(dim, reduction)
#         self.pa = PixelAttention(dim)
#         self.conv = nn.Conv2d(dim, dim, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, y):
#         initial = x + y
#         cattn = self.ca(initial)
#         sattn = self.sa(initial)
#         pattn1 = sattn + cattn
#         pattn2 = self.sigmoid(self.pa(initial, pattn1))
#         result = initial + pattn2 * x + (1 - pattn2) * y
#         result = self.conv(result)
#         return result


# # === GazeLLE ===
# class GazeLLE(nn.Module):
#     def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
#         super().__init__()
#         self.inout = inout  # 外部控制inout
#         self.num_layers = num_layers  # 外部配置num_layers
#         self.backbone = backbone
#         self.dim = dim
#         self.in_size = in_size
#         self.out_size = out_size
#         self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
#         self.linear = nn.Conv2d(backbone.get_dimension(), dim, 1)
#         self.head_token = nn.Embedding(1, dim)
#         if inout:
#             self.inout_token = nn.Embedding(1, dim)
#         self.transformer = nn.Sequential(*[
#             Block(dim=dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
#             for _ in range(self.num_layers)
#         ])
#         self.heatmap_head = nn.Sequential(
#             nn.ConvTranspose2d(dim, dim, 2, 2),
#             nn.Conv2d(dim, 1, 1, bias=False),
#             nn.Sigmoid()
#         )
#         self.attention_fusion = CGAFusion(dim)  # 使用CGA融合

#     def forward(self, input):
#         num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
#         x = self.linear(self.backbone(input["images"]))
#         depth_map = input["depth_map"]
#         depth_features = F.interpolate(depth_map, size=x.shape[2:], mode='bilinear', align_corners=False)
#         x, depth_features = self.interaction(x, depth_features)
#         x = self.fusion(x, depth_features)
#         x = self.calibration(x)
#         x = self.enhancement(x)
#         x = self.attention_fusion(x, depth_features)  # 使用CGA融合
#         pos_embed = self.get_depth_guided_positional_encoding(depth_features)
#         x = x + pos_embed
#         x = utils.repeat_tensors(x, num_ppl_per_img)
#         head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
#         head_map_embeddings = head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
#         x = x + head_map_embeddings
#         x = x.flatten(start_dim=2).permute(0, 2, 1)
#         if self.inout:
#             x = torch.cat([self.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1), x], dim=1)
#         x = self.transformer(x)
#         if self.inout:
#             inout_tokens = x[:, 0, :]
#             inout_preds = self.inout_head(inout_tokens).squeeze(-1)
#             inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
#             x = x[:, 1:, :]
#         x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
#         x = self.heatmap_head(x).squeeze(1)
#         x = torchvision.transforms.functional.resize(x, self.out_size)
#         heatmap_preds = utils.split_tensors(x, num_ppl_per_img)
#         return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

#     def get_depth_guided_positional_encoding(self, depth_features):
#         B, _, H, W = depth_features.shape
#         key = (H, W)
#         if key not in self._pos_cache:
#             self._pos_cache[key] = positionalencoding2d(self.dim, H, W).to(depth_features.device)
#         return self._pos_cache[key]


# # === Positional Encoding ===
# def positionalencoding2d(d_model, height, width):
#     if d_model % 4 != 0:
#         raise ValueError("位置编码要求 d_model 为4的倍数")
#     pe = torch.zeros(d_model, height, width)
#     d_model = d_model // 2
#     div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
#     pos_w = torch.arange(0., width).unsqueeze(1)
#     pos_h = torch.arange(0., height).unsqueeze(1)
#     pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     return pe


# # === 模型工厂函数 ===
# def get_gazelle_model(model_name):
#     factory = {
#         "gazelle_cgaf_dinov2_vits14": gazelle_cgaf_dinov2_vits14,
#         "gazelle_cgaf_dinov2_vitb14": gazelle_cgaf_dinov2_vitb14,
#         "gazelle_cgaf_dinov2_vitl14": gazelle_cgaf_dinov2_vitl14,
#         "gazelle_cgaf_dinov2_vitb14_inout": gazelle_cgaf_dinov2_vitb14_inout,
#         "gazelle_cgaf_dinov2_vitl14_inout": gazelle_cgaf_dinov2_vitl14_inout,
#     }
#     assert model_name in factory, "invalid model name"
#     return factory[model_name]()

# def gazelle_cgaf_dinov2_vits14():
#     backbone = DinoV2Backbone('dinov2_vits14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform

# def gazelle_cgaf_dinov2_vitb14():
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform

# def gazelle_cgaf_dinov2_vitl14():
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone), transform

# def gazelle_cgaf_dinov2_vitb14_inout():
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone, inout=True), transform

# def gazelle_cgaf_dinov2_vitl14_inout():
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     return GazeLLE(backbone, inout=True), transform


# 9.28
# gazelle/model.py

import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import math

import gazelle.utils as utils
from gazelle.backbone import DinoV2Backbone, DinoV3Backbone


class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64), device=None):
        """
        GazeLLE 模型主类：用于 gaze heatmap + in/out 预测
        """
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 将 backbone 输出映射到 transformer 输入维度
        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)

        # 用于 head 位置增强的可学习 token
        self.head_token = nn.Embedding(1, self.dim)

        # 注册空间位置信息编码
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w))

        # 可选 in/out token
        if self.inout:
            self.inout_token = nn.Embedding(1, self.dim)

        # 构建多层 Transformer block
        self.transformer = nn.Sequential(*[
            Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
            for i in range(num_layers)
        ])

        # 热图预测头
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # in/out 分类头
        if self.inout:
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

    def forward(self, input):
        """
        input["images"]: Tensor[B, 3, H, W]
        input["bboxes"]: List[List[Tuple(xmin, ymin, xmax, ymax)]]
        """
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]

        # 提取视觉特征
        x = self.backbone.forward(input["images"].to(self.device))
        x = self.linear(x)
        x = x + self.pos_embed  # 添加位置编码

        # 每人一份复制
        x = utils.repeat_tensors(x, num_ppl_per_img)

        # 获取头部位置 mask，并通过 head token 引导注意力
        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
        head_map_embeddings = head_maps.unsqueeze(1) * self.head_token.weight.view(1, self.dim, 1, 1)
        x = x + head_map_embeddings

        # 展平维度，输入 transformer
        x = x.flatten(start_dim=2).permute(0, 2, 1)

        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1), x], dim=1)

        x = self.transformer(x)

        # inout 判别输出
        if self.inout:
            inout_tokens = x[:, 0, :]
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :]  # 去除 inout token

        # 恢复空间维度
        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
        x = self.heatmap_head(x).squeeze(1)
        x = torchvision.transforms.functional.resize(x, self.out_size)

        heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

        return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

    def get_input_head_maps(self, bboxes):
        """
        将 head bbox 转换为 0/1 mask
        """
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None:
                    img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w, device=self.device))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    width, height = self.featmap_w, self.featmap_h
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(1, xmax)
                    ymax = min(1, ymax)
                    if xmin >= xmax or ymin >= ymax:
                        img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w, device=self.device))
                        continue
                    xmin = torch.round(torch.tensor([xmin], device=self.device).float() * width).int()
                    ymin = torch.round(torch.tensor([ymin], device=self.device).float() * height).int()
                    xmax = torch.round(torch.tensor([xmax], device=self.device).float() * width).int()
                    ymax = torch.round(torch.tensor([ymax], device=self.device).float() * height).int()
                    head_map = torch.zeros((height, width), device=self.device)
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

        if keys2 - keys1:
            print("WARNING: 未使用的参数键：", keys2 - keys1)
        if keys1 - keys2:
            print("WARNING: 缺失的参数键：", keys1 - keys2)

        for k in keys1 & keys2:
            current_state_dict[k] = ckpt_state_dict[k]

        self.load_state_dict(current_state_dict, strict=False)


def positionalencoding2d(d_model, height, width):
    """2D sin/cos 位置编码"""
    if d_model % 4 != 0:
        raise ValueError("d_model 必须为 4 的倍数")
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


# 构造模型工厂
def get_gazelle_model(model_name, device=None):
    factory = {
        # ---- DINOv2 ----
        "gazelle_dinov2_vitb14": gazelle_dinov2_vitb14,
        "gazelle_dinov2_vitl14": gazelle_dinov2_vitl14,
        "gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout,
        "gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout,
        # ---- DINOv3 ----
        "gazelle_dinov3_vitb16": gazelle_dinov3_vitb16,
        "gazelle_dinov3_vitl16": gazelle_dinov3_vitl16,
        "gazelle_dinov3_vits16": gazelle_dinov3_vits16,
        "gazelle_dinov3_vith16": gazelle_dinov3_vith16,
        "gazelle_dinov3_vit7b16": gazelle_dinov3_vit7b16,
        "gazelle_dinov3_vitb16_inout": gazelle_dinov3_vitb16_inout,
        "gazelle_dinov3_vitl16_inout": gazelle_dinov3_vitl16_inout,
    }
    assert model_name in factory.keys(), "无效的模型名称"
    return factory[model_name](device=device)


# ---- DINOv2 ----
def gazelle_dinov2_vitb14(device=None):
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, device=device)
    return model, transform

def gazelle_dinov2_vitl14(device=None):
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, device=device)
    return model, transform

def gazelle_dinov2_vitb14_inout(device=None):
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True, device=device)
    return model, transform

def gazelle_dinov2_vitl14_inout(device=None):
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True, device=device)
    return model, transform


# ---- DINOv3 ----
def gazelle_dinov3_vits16(device=None):
    backbone = DinoV3Backbone('vit_small_patch16_dinov3')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, device=device)
    return model, transform

def gazelle_dinov3_vitb16(device=None):
    backbone = DinoV3Backbone('vit_base_patch16_dinov3')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, device=device)
    return model, transform

def gazelle_dinov3_vitl16(device=None):
    backbone = DinoV3Backbone('vit_large_patch16_dinov3')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, device=device)
    return model, transform

def gazelle_dinov3_vith16(device=None):
    backbone = DinoV3Backbone('vit_huge_plus_patch16_dinov3')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, device=device)
    return model, transform

def gazelle_dinov3_vit7b16(device=None):
    backbone = DinoV3Backbone('vit_7b_patch16_dinov3')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, device=device)
    return model, transform

def gazelle_dinov3_vitb16_inout(device=None):
    backbone = DinoV3Backbone('vit_base_patch16_dinov3')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True, device=device)
    return model, transform

def gazelle_dinov3_vitl16_inout(device=None):
    backbone = DinoV3Backbone('vit_large_patch16_dinov3')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True, device=device)
    return model, transform
