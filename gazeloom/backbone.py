# from abc import ABC, abstractmethod
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# # from dinov2.models.vision_transformer import vit_small, vit_base, vit_large
# from simdinov2.models.vision_transformer import vit_base, vit_small, vit_large


# # 定义骨干网络的抽象基类
# class Backbone(nn.Module, ABC):
#     def __init__(self):
#         super(Backbone, self).__init__()

#     @abstractmethod
#     def forward(self, x):
#         pass

#     @abstractmethod
#     def get_dimension(self):
#         pass

#     @abstractmethod
#     def get_out_size(self, in_size):
#         pass

#     def get_transform(self, in_size=(224, 224)):
#         return transforms.Compose([
#             transforms.Resize(in_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225]),
#         ])

# # 定义DINOv2骨干网络
# class DinoV2Backbone(Backbone):
#     def __init__(self, model_name):
#         super(DinoV2Backbone, self).__init__()

#         # 根据model_name选择对应ViT模型结构
#         if model_name == 'dinov2_vits14':
#             self.model = vit_small(patch_size=14)
#             checkpoint_path = 'gazelle/dinov2_vits14_reg4_pretrain.pth'
#         elif model_name == 'dinov2_vitb14':
#             self.model = vit_base(patch_size=16)
#             checkpoint_path = 'gazelle/vitb16_reg4_SimDNIOv2_ep100.pth'
#         elif model_name == 'dinov2_vitl14':
#             self.model = vit_large(patch_size=14)
#             checkpoint_path = 'gazelle/dinov2_vitl14_reg4_pretrain.pth'
#         else:
#             raise ValueError(f"Unsupported model_name: {model_name}")

#         # # 加载预训练权重
#         # checkpoint = torch.load(checkpoint_path, map_location='cpu')
#         # checkpoint = checkpoint['teacher']  # 指定加载教师模型的权重
#         # checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # 去除 DataParallel 的前缀
#         checkpoint = torch.load(checkpoint_path, map_location='cpu')

#         print(f"加载的预训练模型包含的键: {list(checkpoint.keys())}")
#         # 如果是 'teacher' 模型权重
#         # checkpoint = checkpoint['teacher']  
#         # print(f"从 'teacher' 模型中提取出的权重键: {list(checkpoint.keys())}")

#         # checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # 去除 DataParallel 的前缀
#         # print(f"去除 'module.' 后的键: {list(checkpoint.keys())}")
#         # 插值处理pos_embed
#         if 'pos_embed' in checkpoint:
#             pos_embed_pretrained = checkpoint['pos_embed']
#             pos_embed_current = self.model.pos_embed

#             if pos_embed_pretrained.shape != pos_embed_current.shape:
#                 print(f"[Backbone] Resizing pos_embed from {pos_embed_pretrained.shape} to {pos_embed_current.shape}")

#                 cls_token = pos_embed_pretrained[:, 0:1, :]
#                 patch_token = pos_embed_pretrained[:, 1:, :]

#                 num_patches_pretrained = patch_token.shape[1]
#                 num_patches_current = pos_embed_current.shape[1] - 1

#                 H_pre = W_pre = int(num_patches_pretrained ** 0.5)
#                 H_cur = W_cur = int(num_patches_current ** 0.5)

#                 patch_token = patch_token.reshape(1, H_pre, W_pre, -1).permute(0, 3, 1, 2)
#                 patch_token = F.interpolate(patch_token, size=(H_cur, W_cur), mode='bilinear', align_corners=False)
#                 patch_token = patch_token.permute(0, 2, 3, 1).reshape(1, H_cur * W_cur, -1)

#                 checkpoint['pos_embed'] = torch.cat((cls_token, patch_token), dim=1)

#         missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
#         if missing_keys:
#             print(f"[Backbone] Warning: Missing keys when loading checkpoint: {missing_keys}")
#         if unexpected_keys:
#             print(f"[Backbone] Warning: Unexpected keys when loading checkpoint: {unexpected_keys}")

#     def forward(self, x):
#         b, c, h, w = x.shape
#         out_h, out_w = self.get_out_size((h, w))
#         x = self.model.forward_features(x)['x_norm_patchtokens']
#         x = x.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2)
#         return x

#     def get_dimension(self):
#         return self.model.embed_dim

#     def get_out_size(self, in_size):
#         h, w = in_size
#         return (h // self.model.patch_size, w // self.model.patch_size)
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# from dinov2.models.vision_transformer import vit_small, vit_base, vit_large
from simdinov2.models.vision_transformer import vit_base, vit_small, vit_large


# 定义骨干网络的抽象基类
class Backbone(nn.Module, ABC):
    def __init__(self):
        super(Backbone, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_dimension(self):
        pass

    @abstractmethod
    def get_out_size(self, in_size):
        pass

    def get_transform(self, in_size=(224, 224)):
        return transforms.Compose([
            transforms.Resize(in_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

# 定义DINOv2骨干网络
class DinoV2Backbone(Backbone):
    def __init__(self, model_name):
        super(DinoV2Backbone, self).__init__()

        # 根据model_name选择对应ViT模型结构
        if model_name == 'dinov2_vits14':
            self.model = vit_small(patch_size=14)
            checkpoint_path = 'gazelle/dinov2_vits14_reg4_pretrain.pth'
        elif model_name == 'dinov2_vitb14':
            self.model = vit_base(patch_size=16)
            # checkpoint_path = 'gazelle/vitb16_reg4_SimDNIOv2_ep100.pth'
            checkpoint_path = 'gazelle/modified_weights.pth'
            
        elif model_name == 'dinov2_vitl14':
            self.model = vit_large(patch_size=14)
            checkpoint_path = 'gazelle/dinov2_vitl14_reg4_pretrain.pth'
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # 加载预训练权重
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint = checkpoint['teacher']  # 提取 'teacher' 字段中的权重

        print(f"加载的预训练模型包含的键: {list(checkpoint.keys())}")

        # 去掉 'backbone.' 前缀以匹配模型需要的权重名
        checkpoint = {k.replace('backbone.', ''): v for k, v in checkpoint.items()}

        print(f"[Backbone] 去除 'backbone.' 后的权重键: {list(checkpoint.keys())}")

        # 获取模型需要的所有键
        model_keys = list(self.model.state_dict().keys())
        print(f"[Backbone] 模型需要的权重键: {model_keys}")

        # 检查缺失和多余的权重
        missing_keys = [key for key in model_keys if key not in checkpoint]
        unexpected_keys = [key for key in checkpoint if key not in model_keys]
        matching_keys = [key for key in model_keys if key in checkpoint]

        print(f"[Backbone] 正常匹配的权重键: {matching_keys}")

        if missing_keys:
            print(f"[Backbone] Missing keys (模型缺失的权重): {missing_keys}")
        if unexpected_keys:
            print(f"[Backbone] Unexpected keys (预训练权重中不需要的权重): {unexpected_keys}")

        # 如果需要处理 pos_embed
        if 'pos_embed' in checkpoint:
            pos_embed_pretrained = checkpoint['pos_embed']
            pos_embed_current = self.model.pos_embed

            print(f"[Backbone] Pos_embed预训练形状: {pos_embed_pretrained.shape}")
            print(f"[Backbone] 当前pos_embed形状: {pos_embed_current.shape}")

            if pos_embed_pretrained.shape != pos_embed_current.shape:
                print(f"[Backbone] Resizing pos_embed from {pos_embed_pretrained.shape} to {pos_embed_current.shape}")

                cls_token = pos_embed_pretrained[:, 0:1, :]
                patch_token = pos_embed_pretrained[:, 1:, :]

                num_patches_pretrained = patch_token.shape[1]
                num_patches_current = pos_embed_current.shape[1] - 1

                H_pre = W_pre = int(num_patches_pretrained ** 0.5)
                H_cur = W_cur = int(num_patches_current ** 0.5)

                patch_token = patch_token.reshape(1, H_pre, W_pre, -1).permute(0, 3, 1, 2)
                patch_token = F.interpolate(patch_token, size=(H_cur, W_cur), mode='bilinear', align_corners=False)
                patch_token = patch_token.permute(0, 2, 3, 1).reshape(1, H_cur * W_cur, -1)

                checkpoint['pos_embed'] = torch.cat((cls_token, patch_token), dim=1)


    def forward(self, x):
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        x = self.model.forward_features(x)['x_norm_patchtokens']
        x = x.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2)
        return x

    def get_dimension(self):
        return self.model.embed_dim

    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.model.patch_size, w // self.model.patch_size)
