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

# 加载预训练权重
checkpoint_path = r'gazelle/vitb16_reg4_SimDNIOv2_ep100.pth'
state_dict = torch.load(checkpoint_path, map_location='cpu')
state_dict = state_dict['teacher']['backbone']  # 提取教师模型的权重
state_dict = {k.replace('teacher.backbone.', ''): v for k, v in state_dict.items()}  # 去除命名空间

# 初始化模型
model = vit_small(img_size=518, patch_size=16, init_values=1.0, block_chunks=0)

# 加载权重
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {missing_keys}")
print(f"Unexpected keys: {unexpected_keys}")
