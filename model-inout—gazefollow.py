import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import math

import gazelle.utils as utils
from gazelle.backbone import DinoV2Backbone, DinoV3Backbone


class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3,
                 in_size=(448, 448), out_size=(64, 64), device=None):
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
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 将 backbone 输出映射到 transformer 输入维度
        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)

        # 用于 head 位置增强的可学习 token
        self.head_token = nn.Embedding(1, self.dim)

        # 注册空间位置信息编码
        self.register_buffer("pos_embed",
                             positionalencoding2d(self.dim, self.featmap_h, self.featmap_w))

        # 可选 in/out token
        if self.inout:
            self.inout_token = nn.Embedding(1, self.dim)

        # 构建多层 Transformer block
        self.transformer = nn.Sequential(*[
            Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
            for _ in range(num_layers)
        ])

        # 热图预测头（输出概率）
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # in/out 分类头（去掉 Sigmoid，直接输出 logits）
        if self.inout:
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1)
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
        x = x + self.pos_embed

        # 每人一份复制
        x = utils.repeat_tensors(x, num_ppl_per_img)

        # 获取头部位置 mask，并通过 head token 引导注意力
        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
        head_map_embeddings = head_maps.unsqueeze(1) * self.head_token.weight.view(1, self.dim, 1, 1)
        x = x + head_map_embeddings

        # 展平维度
        x = x.flatten(start_dim=2).permute(0, 2, 1)

        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1), x], dim=1)

        x = self.transformer(x)

        if self.inout:
            inout_tokens = x[:, 0, :]
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)  # logits
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :]

        # 恢复空间维度
        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
        x = self.heatmap_head(x).squeeze(1)
        x = torchvision.transforms.functional.resize(x, self.out_size)

        heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

        return {"heatmap": heatmap_preds,
                "inout": inout_preds if self.inout else None}

    def get_input_head_maps(self, bboxes):
        """将 head bbox 转换为 0/1 mask"""
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
