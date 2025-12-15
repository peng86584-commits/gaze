

# import torch  # 导入PyTorch主库
# import torch.nn as nn  # 导入PyTorch的神经网络模块
# import torchvision  # 导入PyTorch视觉工具库
# from timm.models.vision_transformer import Block  # 从timm库中导入Transformer的基础Block
# import math  # 导入数学库

# import gazelle.utils as utils  # 导入项目中的工具函数模块
# from gazelle.backbone import DinoV2Backbone  # 导入项目中的DinoV2骨干网络模块
# import torch.nn.functional as F

# # 定义GazeLLE模型类
# class GazeLLE(nn.Module):  # 继承自PyTorch的nn.Module
#     def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):  # 初始化函数
#         super().__init__()  # 调用父类nn.Module的初始化函数
#         self.backbone = backbone  # 设置骨干网络
#         self.dim = dim  # 设置Transformer特征维度
#         self.num_layers = num_layers  # Transformer堆叠层数
#         self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)  # 获取特征图的高宽尺寸
#         self.in_size = in_size  # 记录输入图像大小
#         self.out_size = out_size  # 记录输出热图大小
#         self.inout = inout  # 是否启用in/out判别任务（场景内/外）
#         self.relu = nn.ReLU(inplace=True)  # 在这里定义 ReLU 激活函数

#         # 使用1x1卷积将骨干网络输出的特征图通道数调整为self.dim。
#         # 输入特征图的形状为B,C,H,W，其中C是骨干网络输出的通道数。
#         # 输出特征图的形状为B,self.dim,H,W，即通道数被调整为self.dim。
#         # 1: 卷积核大小为1x1，表示在空间维度上不进行卷积操作，只对通道进行线性变换。
#         # 实现过程：1x1卷积通过学习一组权重矩阵，将输入特征图的每个位置的多通道信息线性组合成目标通道数。
#         # 1x1卷积是一种高效调整通道数的方法，它不会引入额外的空间计算负担，同时可以通过学习到的权重矩阵捕捉通道间的相关性。
#         # 这在不改变空间分辨率的情况下，调整特征图的通道维度，使其与后续模块（如Transformer或注意力机制）的输入要求相匹配。
        
#         # backbone.get_dimension() 获取骨干网络输出的通道数，作为输入通道数。
#         # self.dim 是目标通道数，作为输出通道数。
#         # 参数学习：1×1卷积层的权重矩阵大小为 [Cout, Cin, 1, 1]。
#         # 在训练过程中，这些权重通过反向传播算法进行更新，以最小化模型的损失函数。
#         # 模型会学习如何将输入特征图的通道信息有效转换为目标通道数的表示。
#         # 梯度更新：在反向传播过程中，计算损失函数对输出特征图的梯度，然后通过链式法则计算对卷积核的梯度，从而更新卷积核的权重。
        
#         # 通道数调整：1×1卷积主要用于调整特征图的通道数。它不会改变特征图的空间维度（即高度和宽度），
#         # 但可以将输入特征图的通道数从骨干网络的输出维度（由backbone.get_dimension()获取）调整为模型所需的维度self.dim。
#         # 特征融合：1×1卷积通过线性组合输入特征图的通道信息，能够融合多通道特征，生成新的特征表示，从而为后续模块提供更合适的输入。
#         # 输出特征图：输出特征图形状为 [B, Cout, H, W]，其中 Cout是目标通道数（即self.dim）。每个位置的输出值是输入特征图对应位置的通道信息经过线性变换后的结果。
#         # 
#         self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1) # 用1x1卷积调整特征通道数，骨干网络输出通道数调整为 dim
        
#         # 定义一个可学习的嵌入向量，用于表示头部位置的特殊Token。
#         # 引入特殊Token（如[CLS] Token）来聚合全局信息或表示特定位置（如头部）的信息。
#         # 1: 表示嵌入层的词汇表大小，这里只包含一个Token（头部位置Token）
#         # self.dim: 嵌入向量的维度，与模型的隐藏层维度一致。
#         # 嵌入层为每个可能的输入索引（这里是0）学习一个对应的向量。在训练过程中，这个向量会通过反向传播不断优化，以捕捉头部位置的特征信息。
      
#         self.head_token = nn.Embedding(1, self.dim)  # 定义头部位置Token的嵌入
   
#         self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w).squeeze(dim=0).squeeze(dim=0))  # 注册位置编码，不作为可学习参数
#         # self.register_buffer("pos_embed", ...)：将张量注册到模型的缓冲区中，
#         # 不会被优化器更新（即不参与梯度计算和参数学习），会随模型一起保存/加载（存在于state_dict中）
#         # positionalencoding2d(self.dim, self.featmap_h, self.featmap_w)：生成二维位置编码，
#         # 输出：形状为 (1, dim, featmap_h, featmap_w) 的位置编码张量。
#         # .squeeze(dim=0).squeeze(dim=0)：移除多余维度，将位置编码从 (1, dim, h, w) 转换为 (dim, h, w)。
        
#         if self.inout:  # 如果需要in/out输出
#             self.inout_token = nn.Embedding(1, self.dim)  # 定义in/out Token的嵌入
        
#         # 叠了多个 Block，每个 Block 都是 Transformer 中的基础层。构建一个由多个 Transformer 块堆叠而成的深度 Transformer 模型。  
#         # self.transformer = nn.Sequential(*[...])：按顺序堆叠多个神经网络层。将多个 Transformer 块串联起来，形成完整的 Transformer 模块。
#         # 输入：特征序列（形状为 (B, N, D)，其中 B 是批量大小，N 是序列长度，D 是特征维度）。
#         # 输出：经过多层 Transformer 块处理后的特征序列。
#         # 前向传播时，输入数据会依次通过所有堆叠的模块，自动管理模块的参数和梯度。
#         # 多层感知机（MLP）：mlp_ratio=4 表示 MLP 的隐藏层维度是输入维度的 4 倍。
#         # DropPath：一种正则化技术，drop_path=0.1 表示以 10% 的概率随机丢弃一些路径。
        
#         self.transformer = nn.Sequential(*[  # 定义Transformer块堆叠
#             Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)  # 每一层使用8头注意力、4倍MLP扩展
#             for i in range(num_layers)  # 堆叠 num_layers 个 Transformer 块，形成深度模型。
#         ])

#         # 1. 转置卷积层（ConvTranspose2d）：用于上采样操作，将输入特征图的尺寸扩大一倍。
#         # 这有助于恢复特征图的空间分辨率，使其更接近原始输入图像的尺寸，从而提高热图预测的空间精度。
#         # 2. 标准卷积层（Conv2d）：将特征图的通道数转换为单通道。这一步骤将多通道的特征信息整合到一个通道中，形成热图，
#         # 其中每个像素值表示对应位置的目标存在概率或响应强度
#         # 3. 在 nn.Sequential 中，输入数据会按照定义的顺序流过每一层，每一层的输出作为下一层的输入，直到最后得到结果。
#         # 这种方式不需要单独定义 forward 方法
#         # 当输入特征图经过热图预测分支时，首先通过转置卷积层进行上采样，恢复部分空间信息，使特征图的尺寸更接近原始输入图像。
#         # 接着，标准卷积层将多通道的特征整合为单通道的热图，每个像素值表示对应位置的目标响应强度。
#         # 最后，Sigmoid激活函数将这些响应强度转换为概率值，输出热图中每个像素的概率表示目标在该位置存在的可能性。
        
#         self.heatmap_head = nn.Sequential(  # 定义热图预测分支
#             # ConvTranspose2d：一个转置卷积层，用于上采样（使热图的尺寸变大）。
#             nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),  # 上采样一倍
#             # Conv2d：一个标准卷积层，将特征图转化为单通道的热图（给、灰度图）。
#             nn.Conv2d(dim, 1, kernel_size=1, bias=False),  # 变换成单通道
#             nn.Sigmoid()  # 用Sigmoid将输出限制到0-1之间，表示概率。
#         )

#         if self.inout:  # 如果需要in/out输出
#             self.inout_head = nn.Sequential(  # 定义in/out预测头
#                 nn.Linear(self.dim, 128),  # 线性层降维
#                 nn.ReLU(),  # 激活函数，增加非线性。
#                 nn.Dropout(0.1),  # 随机丢弃10%的神经元，防止过拟合。
#                 nn.Linear(128, 1),  # 再映射到1维
#                 nn.Sigmoid()  # Sigmoid输出0-1概率
#             )

#     def forward(self, input):  # 前向传播定义
#         # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # 通过输入的边界框信息（bboxes）计算每张图片中的人数。
#         # input["bboxes"]：输入的数据结构，包含每张图像的边界框信息。
#         # 每张图像可能有多个边界框，每个边界框对应一个检测到的物体（在这里可能是一个人）
#         # len(bbox_list)：计算每张图像的边界框个数，实际上就是统计每张图像中的物体数量。
        
#         num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]  # 记录每张图中的人数（bbox个数）
        
#         # input["images"]：输入的是一批图像。假设输入是形状为 [batch_size, channels, height, width] 的4维tensor。
#         # batch_size 表示图像的数量，channels 是图像的通道数（RGB是3通道），height 和 width 分别是图像的高度和宽度。
#         # x：self.backbone.forward 返回的特征图。假设它是一个形状为 [batch_size, channels_out, height_out, width_out] 的tensor，
#         # 其中 channels_out 是骨干网络输出的通道数，height_out 和 width_out 是输出特征图的高度和宽度。
        
#         x = self.backbone.forward(input["images"])  # 提取输入图片的特征图
#         # self.linear：这里的linear通常是一个1x1卷积层（实际上是线性层），其作用是调整输出特征图的通道数。
#         x = self.linear(x)  # 将特征通道数调整为dim，一个 1x1 的卷积层，将骨干网络输出的通道数调整为 dim，确保与后续网络兼容。
#         # 将卷积层移动到 GPU
#         # self.depth_conv = nn.Conv2d(1, self.dim, kernel_size=1).to(device)
#         # self.depth_conv = nn.Conv2d(1, self.dim, kernel_size=1)
        
#         self.depth_conv = nn.Conv2d(1, self.dim, kernel_size=1).cuda()  # 将卷积层初始化为 GPU 版本

#          # 获取并处理深度图
#         # depth_map = input["depth_map"].to(self.depth_conv.weight.device)  # 确保深度图在 GPU 上
        
#         depth_map = input["depth_map"]  # 形状为 [batch_size, 1, height, width]
#         depth_features = self.depth_conv(depth_map)  # 将深度图转化为与特征图相同维度的特征
#           # 确保深度图特征和图像特征图在空间维度上匹配
#         if depth_features.shape[2:] != x.shape[2:]:
#             # 调整深度图特征的尺寸，使其与图像特征图的尺寸一致
#             depth_features = F.interpolate(depth_features, size=x.shape[2:], mode='bilinear', align_corners=False)
            
#         # # 增加
#         # # 在通道维度上拼接图像特征和深度特征
#         # x = torch.cat([x, depth_features], dim=1)  # 拼接特征

#         # # 定义融合卷积层，将拼接后的通道数转换为指定维度
#         # fusion_dim = self.dim  # 可以根据需要调整融合后的通道数
#         # self.fusion_conv = nn.Conv2d(x.shape[1], fusion_dim, kernel_size=3, padding=1).cuda()
#         # x = self.fusion_conv(x)

#         # # 可选：添加批归一化和激活函数
#         # self.fusion_bn = nn.BatchNorm2d(fusion_dim).cuda()
#         # x = self.fusion_bn(x)
#         # x = self.relu(x)
#         # # # 
        
#         # 融合深度信息和图像特征图
#         # x = x + depth_features  # 加法融合深度信息与图像特征图
        
#         # 尝试2
#         # 定义多模态特征交互模块
#         class ModalityInteraction(nn.Module):
#             def __init__(self, dim):
#                super(ModalityInteraction, self).__init__()
#                self.image_to_depth = nn.Conv2d(dim, dim, kernel_size=1)
#                self.depth_to_image = nn.Conv2d(dim, dim, kernel_size=1)
#                self.softmax = nn.Softmax(dim=1)

#             def forward(self, image_features, depth_features):
#         # 图像特征对深度特征的注意力
#                depth_attn = self.softmax(self.image_to_depth(image_features))
#         # 深度特征对图像特征的注意力
#                image_attn = self.softmax(self.depth_to_image(depth_features))
#         # 特征交互
#                image_features = image_features + depth_features * image_attn
#                depth_features = depth_features + image_features * depth_attn
#                return image_features, depth_features

#         # 使用特征交互模块
#         self.interaction = ModalityInteraction(self.dim).cuda() 
#         x, depth_features = self.interaction(x, depth_features)

#         # 直接相加融合深度信息与图像特征图
#         x = x + depth_features  # 加法融合深度信息与图像特征图
        
    
                
#         # 位置编码（Positional Encoding）：由于卷积层和线性层通常不会保留图像中的空间位置信息，
#         # 尤其是当特征图变得越来越小的时候。为了让模型知道每个位置的信息（例如，图像中的某个区域的特征），引入了位置编码。
#         # self.pos_embed：这是一个与输入特征图大小相同的tensor，包含了每个位置的编码信息。
#         # 通常是通过一些学习得到的，或者使用固定的方式（例如，正弦余弦函数）生成。
#         x = x + self.pos_embed  # 加入位置编码信息
        
#         # utils.repeat_tensors(x, num_ppl_per_img)：这个函数的作用是根据每张图像的人数（即边界框的数量），将特征图进行复制，
#         # 确保每个人物都有对应的特征。 每个边界框（即每个人物）都需要独立的特征，以便模型能够为每个人物生成不同的输出。
#         # 举个例子，如果图像1有2个人，图像2有1个人，x会分别复制2次和1次，确保每个人物都有相应的特征图。
        
#         x = utils.repeat_tensors(x, num_ppl_per_img)  # 根据人数将特征复制（每个人头一份特征）
        
#         # 提取边界框信息：从输入数据中提取 bboxes（边界框）信息，这些边界框定义了图像中目标对象的位置。
#         # self.get_input_head_maps(input["bboxes"])：这个函数的作用是根据边界框生成头部掩码。
#         # 生成头部掩码调：用 self.get_input_head_maps 方法，根据边界框信息生成头部掩码（head maps）。这些掩码用于标记图像中头部区域的位置。
#         # 每个边界框对应一个掩码，标记出物体的头部位置。该函数的输出是每张图像中每个人物的掩码。
#         # torch.cat(..., dim=0)：将所有图像中的头部掩码拼接成一个大的tensor，
#         # 拼接掩码：使用 torch.cat 将生成的头部掩码在第 0 维（通常是批量维度）进行拼接，形成一个完整的张量。
#         # 维度是 [total_heads, height, width]，其中 total_heads 是所有图像和所有人物的总数。
#         # head_maps.to(x.device)：将生成的头部掩码移动到和特征图相同的设备（例如GPU）上，确保它们在同一计算资源上进行操作
#         head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)  # 生成并拼接头部掩码
        
#         # head_maps.unsqueeze(dim=1)：增加一个维度，使得掩码的形状从 [total_heads, height, width] 
#         # 变为 [total_heads, 1, height, width]，这使得它可以与特征图在维度上对齐。
#         # self.head_token.weight.unsqueeze(-1).unsqueeze(-1)：
#         # self.head_token.weight 是一个可训练的参数，表示每个人物头部的特征信息。通过unsqueeze方法，
#         # 将它扩展为一个与掩码大小匹配的形状（即 [num_heads, channels, height, width]），使得它可以与头部掩码相乘
#         # head_map_embeddings = head_maps * self.head_token.weight：这是将头部掩码信息和头部token特征进行加权融合。
#         head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)  # 将头部掩码编码到特征中
#         x = x + head_map_embeddings  # 特征图中融合头部位置信息
#         # flatten(start_dim=2)：将特征图展平成序列，从第2维开始展平（即将height和width展平成一维）。
#         # 展平后，x 的形状变为 [batch_size, height * width, channels]。
#         # permute(0, 2, 1)：调整维度顺序，将 [batch_size, height * width, channels] 转换为 [batch_size, channels, height * width]，这是Transformer模块所需的输入格式。
#         x = x.flatten(start_dim=2).permute(0, 2, 1)  # 将空间维度展平成序列：b (h*w) c -> b c h w

#         if self.inout:  # 如果需要in/out预测
#         # torch.cat(..., dim=1)：将in/out token添加到序列的开头。这个token用于让模型学习物体是否进入或退出场景。
#             x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)  # 添加in/out专用token
        
#         x = self.transformer(x)  # 将序列送入Transformer模块

#         if self.inout:  # 如果需要in/out预测
#             inout_tokens = x[:, 0, :]  # 取出第一个token作为in/out特征
#             inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)  # 进行in/out预测
#             inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)  # 将预测结果按照每张图分开
#             x = x[:, 1:, :]  # 去掉in/out token，只保留场景tokens
#         x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)  # 将序列恢复成特征图 b (h*w) c -> b c h w
#         x = self.heatmap_head(x).squeeze(dim=1)  # 生成单通道热图
#         x = torchvision.transforms.functional.resize(x, self.out_size)  # 将热图调整到目标尺寸
        
#         # utils.split_tensors(x, num_ppl_per_img)：将生成的热图分开，按每张图像中的人数分割，使得每个人物都有独立的热图。

#         heatmap_preds = utils.split_tensors(x, num_ppl_per_img)  # 将热图分开成按图像组织的list

#         return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}  # 返回热图和in/out预测结果

#     def get_input_head_maps(self, bboxes):  # 生成头部掩码图
#         head_maps = []  # 初始化头部掩码列表
#         for bbox_list in bboxes:  # 遍历每张图的头部框
#             img_head_maps = []  # 当前图片的头部掩码列表
#             for bbox in bbox_list:  # 遍历每个人的头部框
#                 if bbox is None:  # 如果没有头部框
#      # 遍历图像中的每个边界框（bbox），如果 bbox 为 None，则说明该图像没有有效的头部框，生成一个全零的掩码。
#      # torch.zeros(self.featmap_h, self.featmap_w) 创建一个全零的张量，大小为 featmap_h 和 featmap_w，对应特征图的高度和宽度。
#                     img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))  # 填充全零图
#                 else:
#     # 如果 bbox 存在，xmin, ymin, xmax, ymax 分别是头部框的左上角和右下角坐标（通常是归一化的坐标，范围在 [0, 1]）。
#     # 将这些归一化坐标转换为实际的像素坐标，通过将坐标乘以特征图的宽度和高度，然后四舍五入为整数
#                     xmin, ymin, xmax, ymax = bbox  # 提取框的坐标
#                     width, height = self.featmap_w, self.featmap_h  # 特征图尺寸
#                     xmin = round(xmin * width)  # 归一化到特征图尺寸
#                     ymin = round(ymin * height)
#                     xmax = round(xmax * width)
#                     ymax = round(ymax * height)
#     # 创建一个全零的掩码 head_map，大小为特征图的高度和宽度。
#     # 然后，将头部框的区域（通过 [ymin:ymax, xmin:xmax] 选择区域）置为 1，表示该区域是头部所在的区域.
#                     head_map = torch.zeros((height, width))  # 初始化全零掩码
#                     head_map[ymin:ymax, xmin:xmax] = 1  # 将bbox区域置1
#                     img_head_maps.append(head_map)  # 保存头部掩码
#     # 使用 torch.stack(img_head_maps) 将当前图像的多个头部掩码堆叠成一个张量，保存在 head_maps 列表中。
#     # head_maps 列表中会包含每张图像的头部掩码。
#             head_maps.append(torch.stack(img_head_maps))  # 合并所有人头部掩码
#         return head_maps  # 返回最终头部掩码列表

#     def get_gazelle_state_dict(self, include_backbone=False):  # 导出模型权重
#         if include_backbone:  # 是否包含骨干网络
#             return self.state_dict()
#         else:
#             return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone")}  # 排除掉骨干网络的参数

#     def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):  # 加载已有的权重
#         current_state_dict = self.state_dict()  # 当前模型参数
#         keys1 = current_state_dict.keys()
#         keys2 = ckpt_state_dict.keys()

#         if not include_backbone:  # 如果不包含backbone
#             keys1 = set([k for k in keys1 if not k.startswith("backbone")])
#             keys2 = set([k for k in keys2 if not k.startswith("backbone")])
#         else:
#             keys1 = set(keys1)
#             keys2 = set(keys2)

#         if len(keys2 - keys1) > 0:  # 检查多余参数
#             print("WARNING unused keys in provided state dict: ", keys2 - keys1)
#         if len(keys1 - keys2) > 0:  # 检查缺失参数
#             print("WARNING provided state dict does not have values for keys: ", keys1 - keys2)

#         for k in list(keys1 & keys2):  # 加载公共部分
#             current_state_dict[k] = ckpt_state_dict[k]
        
#         self.load_state_dict(current_state_dict, strict=False)  # 非严格模式加载参数


# # 定义2D位置编码函数
# def positionalencoding2d(d_model, height, width):
#     if d_model % 4 != 0:  # 检查d_model是否是4的倍数
#         raise ValueError("位置编码要求d_model为4的倍数")
#     pe = torch.zeros(d_model, height, width)  # 初始化位置编码张量
#     d_model = int(d_model / 2)  # 每一方向占用d_model的一半
#     div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))  # 计算缩放因子
#     pos_w = torch.arange(0., width).unsqueeze(1)  # 列方向位置
#     pos_h = torch.arange(0., height).unsqueeze(1)  # 行方向位置
#     pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)  # 宽度方向sin编码
#     pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)  # 宽度方向cos编码
#     pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)  # 高度方向sin编码
#     pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)  # 高度方向cos编码
#     return pe  # 返回位置编码


# # 定义GazeLLE模型工厂函数
# def get_gazelle_model(model_name):
#     factory = {  # 定义模型名字到具体实例函数的映射
#         "gazelle_dinov2_vits14": gazelle_dinov2_vits14,
#         "gazelle_dinov2_vitb14": gazelle_dinov2_vitb14,
#         "gazelle_dinov2_vitl14": gazelle_dinov2_vitl14,
#         "gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout,
#         "gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout,
#     }
#     assert model_name in factory.keys(), "invalid model name"  # 检查输入模型名是否有效
#     return factory[model_name]()  # 返回实例化后的模型和transform

# # 定义具体的模型构建函数
# def gazelle_dinov2_vits14():
#     backbone = DinoV2Backbone('dinov2_vits14')  # 使用ViT-S14骨干
#     transform = backbone.get_transform((448, 448))  # 获取输入预处理
#     model = GazeLLE(backbone)  # 创建模型
#     return model, transform

# def gazelle_dinov2_vitb14():
#     backbone = DinoV2Backbone('dinov2_vitb14')  # 使用ViT-B14骨干
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone)
#     return model, transform

# def gazelle_dinov2_vitl14():
#     backbone = DinoV2Backbone('dinov2_vitl14')  # 使用ViT-L14骨干
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone)
#     return model, transform

# def gazelle_dinov2_vitb14_inout():
#     backbone = DinoV2Backbone('dinov2_vitb14')  # 使用ViT-B14骨干，带in/out输出
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone, inout=True)
#     return model, transform

# def gazelle_dinov2_vitl14_inout():
#     backbone = DinoV2Backbone('dinov2_vitl14')  # 使用ViT-L14骨干，带in/out输出
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone, inout=True)
#     return model, transform

#改变11
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.vision_transformer import Block
# import math
# import gazelle.utils as utils
# from gazelle.backbone import DinoV2Backbone
# import torchvision


# # 定义特征加权融合模块
# class FeatureWeightedFusion(nn.Module):
#     def __init__(self, dim):
#         super(FeatureWeightedFusion, self).__init__()
#         self.weight_image = nn.Parameter(torch.tensor(0.5))
#         self.weight_depth = nn.Parameter(torch.tensor(0.5))

#     def forward(self, image_features, depth_features):
#         weighted_image = image_features * self.weight_image
#         weighted_depth = depth_features * self.weight_depth
#         return weighted_image + weighted_depth


# # 定义特征校准模块
# class FeatureCalibration(nn.Module):
#     def __init__(self, dim):
#         super(FeatureCalibration, self).__init__()
#         self.conv = nn.Conv2d(dim, dim, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, features):
#         calibrated = self.sigmoid(self.conv(features))
#         return features * calibrated


# # 定义特征增强模块
# class FeatureEnhancement(nn.Module):
#     def __init__(self, dim):
#         super(FeatureEnhancement, self).__init__()
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, features):
#         enhanced = self.relu(self.conv1(features))
#         enhanced = self.conv2(enhanced)
#         return features + enhanced


# # 定义注意力引导的特征融合模块
# class AttentionFusion(nn.Module):
#     def __init__(self, dim):
#         super(AttentionFusion, self).__init__()
#         self.attention = nn.Sequential(
#             nn.Conv2d(dim, 1, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, image_features, depth_features):
#         attention = self.attention(image_features)
#         fused_features = image_features * attention + depth_features * (1 - attention)
#         return fused_features


# # 定义多模态特征交互模块
# class ModalityInteraction(nn.Module):
#     def __init__(self, dim):
#         super(ModalityInteraction, self).__init__()
#         self.image_to_depth = nn.Conv2d(dim, dim, kernel_size=1)
#         self.depth_to_image = nn.Conv2d(dim, dim, kernel_size=1)
#         self.softmax = nn.Softmax(dim=1)
#         self.image_attn = nn.MultiheadAttention(dim, num_heads=8)
#         self.depth_attn = nn.MultiheadAttention(dim, num_heads=8)

#     def forward(self, image_features, depth_features):
#         depth_attn = self.softmax(self.image_to_depth(image_features))
#         image_attn = self.softmax(self.depth_to_image(depth_features))
#         image_features = image_features + depth_features * image_attn
#         depth_features = depth_features + image_features * depth_attn

#         batch_size, embedding_dim, height, width = image_features.shape
#         image_features = image_features.reshape(batch_size, embedding_dim, -1).permute(0, 2, 1)
#         depth_features = depth_features.reshape(batch_size, embedding_dim, -1).permute(0, 2, 1)

#         image_features = self.image_attn(image_features, image_features, image_features)[0]
#         depth_features = self.depth_attn(depth_features, depth_features, depth_features)[0]

#         image_features = image_features.permute(0, 2, 1).reshape(batch_size, embedding_dim, height, width)
#         depth_features = depth_features.permute(0, 2, 1).reshape(batch_size, embedding_dim, height, width)

#         return image_features, depth_features


# class GazeLLE(nn.Module):
#     def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
#         super().__init__()
#         self.backbone = backbone
#         self.dim = dim
#         self.num_layers = num_layers
#         self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
#         self.in_size = in_size
#         self.out_size = out_size
#         self.inout = inout
        
#         self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
#         self.head_token = nn.Embedding(1, self.dim)
#         self.relu = nn.ReLU(inplace=True)
#         self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w).squeeze())
        
#         if self.inout:
#             self.inout_token = nn.Embedding(1, self.dim)
        
#         # 使用Dropout防止过拟合
#         self.dropout = nn.Dropout(0.1)
        
#         self.transformer = nn.Sequential(*[
#             Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
#             for i in range(num_layers)
#         ])

#         self.heatmap_head = nn.Sequential(
#             nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
#             nn.Conv2d(dim, 1, kernel_size=1, bias=False),
#             nn.Sigmoid()
#         )

#         if self.inout:
#             self.inout_head = nn.Sequential(
#                 nn.Linear(self.dim, 128),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(128, 1),
#                 nn.Sigmoid()
#             )

#         self.interaction = ModalityInteraction(self.dim)
#         self.depth_conv = nn.Conv2d(1, self.dim, kernel_size=1)
#         self.fusion = FeatureWeightedFusion(self.dim)
#         self.calibration = FeatureCalibration(self.dim)
#         self.enhancement = FeatureEnhancement(self.dim)
#         self.attention_fusion = AttentionFusion(self.dim)

#     def forward(self, input):
#         num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        
#         # 图像特征提取
#         x = self.backbone.forward(input["images"])
#         x = self.linear(x)
#         x = self.dropout(x)

#         # 深度特征提取
#         depth_map = input["depth_map"]
#         if depth_map.shape[1] != 1:
#             raise ValueError("Depth map must have 1 channel")
#         depth_features = self.depth_conv(depth_map)
#         depth_features = self.dropout(depth_features)

#         # 特征对齐
#         if depth_features.shape[2:] != x.shape[2:]:
#             depth_features = F.interpolate(depth_features, size=x.shape[2:], mode='bilinear', align_corners=False)
        
#         # 特征交互
#         x, depth_features = self.interaction(x, depth_features)

#         # 特征融合
#         x = self.fusion(x, depth_features)
#         x = self.calibration(x)
#         x = self.enhancement(x)
#         x = self.attention_fusion(x, depth_features)

#         # 位置编码
#         x = x + self.pos_embed
        
#         # 重复张量
#         x = utils.repeat_tensors(x, num_ppl_per_img)
        
#         # 头部特征图
#         head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
#         head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
#         x = x + head_map_embeddings
#         x = x.flatten(start_dim=2).permute(0, 2, 1)

#         # 处理进出预测
#         if self.inout:
#             x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)
        
#         # Transformer编码
#         x = self.transformer(x)

#         # 生成预测结果
#         if self.inout:
#             inout_tokens = x[:, 0, :]
#             inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
#             inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
#             x = x[:, 1:, :]

#         # 热图预测
#         x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
#         x = self.heatmap_head(x).squeeze(dim=1)
#         x = torchvision.transforms.functional.resize(x, self.out_size)
        
#         heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

#         return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

#     def get_input_head_maps(self, bboxes):
#         head_maps = []
#         for bbox_list in bboxes:
#             img_head_maps = []
#             for bbox in bbox_list:
#                 if bbox is None:
#                     img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))
#                 else:
#                     xmin, ymin, xmax, ymax = bbox
#                     width, height = self.featmap_w, self.featmap_h
#                     xmin = round(xmin * width)
#                     ymin = round(ymin * height)
#                     xmax = round(xmax * width)
#                     ymax = round(ymax * height)
#                     head_map = torch.zeros((height, width))
#                     head_map[ymin:ymax, xmin:xmax] = 1
#                     img_head_maps.append(head_map)
#             head_maps.append(torch.stack(img_head_maps))
#         return head_maps

#     def get_gazelle_state_dict(self, include_backbone=False):
#         if include_backbone:
#             return self.state_dict()
#         else:
#             return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone")}

#     def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
#         current_state_dict = self.state_dict()
#         keys1 = current_state_dict.keys()
#         keys2 = ckpt_state_dict.keys()

#         if not include_backbone:
#             keys1 = set([k for k in keys1 if not k.startswith("backbone")])
#             keys2 = set([k for k in keys2 if not k.startswith("backbone")])
#         else:
#             keys1 = set(keys1)
#             keys2 = set(keys2)

#         if len(keys2 - keys1) > 0:
#             print("WARNING unused keys in provided state dict: ", keys2 - keys1)
#         if len(keys1 - keys2) > 0:
#             print("WARNING provided state dict does not have values for keys: ", keys1 - keys2)

#         for k in list(keys1 & keys2):
#             current_state_dict[k] = ckpt_state_dict[k]
        
#         self.load_state_dict(current_state_dict, strict=False)


# def positionalencoding2d(d_model, height, width):
#     if d_model % 4 != 0:
#         raise ValueError("位置编码要求d_model为4的倍数")
#     pe = torch.zeros(d_model, height, width)
#     d_model = int(d_model / 2)
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
#     assert model_name in factory.keys(), "invalid model name"
#     return factory[model_name]()


# def gazelle_dinov2_vits14():
#     backbone = DinoV2Backbone('dinov2_vits14')
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone)
#     return model, transform

# def gazelle_dinov2_vitb14():
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone)
#     return model, transform

# def gazelle_dinov2_vitl14():
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone)
#     return model, transform

# def gazelle_dinov2_vitb14_inout():
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone, inout=True)
#     return model, transform

# def gazelle_dinov2_vitl14_inout():
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone, inout=True)
#     return model, transform

# 最初的，直接融合
# import torch  # 导入PyTorch主库
# import torch.nn as nn  # 导入PyTorch的神经网络模块
# import torchvision  # 导入PyTorch视觉工具库
# from timm.models.vision_transformer import Block  # 从timm库中导入Transformer的基础Block
# import math  # 导入数学库

# import gazelle.utils as utils  # 导入项目中的工具函数模块
# from gazelle.backbone import DinoV2Backbone  # 导入项目中的DinoV2骨干网络模块
# import torch.nn.functional as F

# # 定义GazeLLE模型类
# class GazeLLE(nn.Module):  # 继承自PyTorch的nn.Module
#     def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):  # 初始化函数
#         super().__init__()  # 调用父类nn.Module的初始化函数
#         self.backbone = backbone  # 设置骨干网络
#         self.dim = dim  # 设置Transformer特征维度
#         self.num_layers = num_layers  # Transformer堆叠层数
#         self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)  # 获取特征图的高宽尺寸
#         self.in_size = in_size  # 记录输入图像大小
#         self.out_size = out_size  # 记录输出热图大小
#         self.inout = inout  # 是否启用in/out判别任务（场景内/外）
#         self.relu = nn.ReLU(inplace=True)  # 在这里定义 ReLU 激活函数

#         # 使用1x1卷积将骨干网络输出的特征图通道数调整为self.dim。
#         # 输入特征图的形状为B,C,H,W，其中C是骨干网络输出的通道数。
#         # 输出特征图的形状为B,self.dim,H,W，即通道数被调整为self.dim。
#         # 1: 卷积核大小为1x1，表示在空间维度上不进行卷积操作，只对通道进行线性变换。
#         # 实现过程：1x1卷积通过学习一组权重矩阵，将输入特征图的每个位置的多通道信息线性组合成目标通道数。
#         # 1x1卷积是一种高效调整通道数的方法，它不会引入额外的空间计算负担，同时可以通过学习到的权重矩阵捕捉通道间的相关性。
#         # 这在不改变空间分辨率的情况下，调整特征图的通道维度，使其与后续模块（如Transformer或注意力机制）的输入要求相匹配。
        
#         # backbone.get_dimension() 获取骨干网络输出的通道数，作为输入通道数。
#         # self.dim 是目标通道数，作为输出通道数。
#         # 参数学习：1×1卷积层的权重矩阵大小为 [Cout, Cin, 1, 1]。
#         # 在训练过程中，这些权重通过反向传播算法进行更新，以最小化模型的损失函数。
#         # 模型会学习如何将输入特征图的通道信息有效转换为目标通道数的表示。
#         # 梯度更新：在反向传播过程中，计算损失函数对输出特征图的梯度，然后通过链式法则计算对卷积核的梯度，从而更新卷积核的权重。
        
#         # 通道数调整：1×1卷积主要用于调整特征图的通道数。它不会改变特征图的空间维度（即高度和宽度），
#         # 但可以将输入特征图的通道数从骨干网络的输出维度（由backbone.get_dimension()获取）调整为模型所需的维度self.dim。
#         # 特征融合：1×1卷积通过线性组合输入特征图的通道信息，能够融合多通道特征，生成新的特征表示，从而为后续模块提供更合适的输入。
#         # 输出特征图：输出特征图形状为 [B, Cout, H, W]，其中 Cout是目标通道数（即self.dim）。每个位置的输出值是输入特征图对应位置的通道信息经过线性变换后的结果。
#         # 
#         self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1) # 用1x1卷积调整特征通道数，骨干网络输出通道数调整为 dim
        
#         # 定义一个可学习的嵌入向量，用于表示头部位置的特殊Token。
#         # 引入特殊Token（如[CLS] Token）来聚合全局信息或表示特定位置（如头部）的信息。
#         # 1: 表示嵌入层的词汇表大小，这里只包含一个Token（头部位置Token）
#         # self.dim: 嵌入向量的维度，与模型的隐藏层维度一致。
#         # 嵌入层为每个可能的输入索引（这里是0）学习一个对应的向量。在训练过程中，这个向量会通过反向传播不断优化，以捕捉头部位置的特征信息。
      
#         self.head_token = nn.Embedding(1, self.dim)  # 定义头部位置Token的嵌入
   
#         self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w).squeeze(dim=0).squeeze(dim=0))  # 注册位置编码，不作为可学习参数
#         # self.register_buffer("pos_embed", ...)：将张量注册到模型的缓冲区中，
#         # 不会被优化器更新（即不参与梯度计算和参数学习），会随模型一起保存/加载（存在于state_dict中）
#         # positionalencoding2d(self.dim, self.featmap_h, self.featmap_w)：生成二维位置编码，
#         # 输出：形状为 (1, dim, featmap_h, featmap_w) 的位置编码张量。
#         # .squeeze(dim=0).squeeze(dim=0)：移除多余维度，将位置编码从 (1, dim, h, w) 转换为 (dim, h, w)。
        
#         if self.inout:  # 如果需要in/out输出
#             self.inout_token = nn.Embedding(1, self.dim)  # 定义in/out Token的嵌入
        
#         # 叠了多个 Block，每个 Block 都是 Transformer 中的基础层。构建一个由多个 Transformer 块堆叠而成的深度 Transformer 模型。  
#         # self.transformer = nn.Sequential(*[...])：按顺序堆叠多个神经网络层。将多个 Transformer 块串联起来，形成完整的 Transformer 模块。
#         # 输入：特征序列（形状为 (B, N, D)，其中 B 是批量大小，N 是序列长度，D 是特征维度）。
#         # 输出：经过多层 Transformer 块处理后的特征序列。
#         # 前向传播时，输入数据会依次通过所有堆叠的模块，自动管理模块的参数和梯度。
#         # 多层感知机（MLP）：mlp_ratio=4 表示 MLP 的隐藏层维度是输入维度的 4 倍。
#         # DropPath：一种正则化技术，drop_path=0.1 表示以 10% 的概率随机丢弃一些路径。
        
#         self.transformer = nn.Sequential(*[  # 定义Transformer块堆叠
#             Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)  # 每一层使用8头注意力、4倍MLP扩展
#             for i in range(num_layers)  # 堆叠 num_layers 个 Transformer 块，形成深度模型。
#         ])

#         # 1. 转置卷积层（ConvTranspose2d）：用于上采样操作，将输入特征图的尺寸扩大一倍。
#         # 这有助于恢复特征图的空间分辨率，使其更接近原始输入图像的尺寸，从而提高热图预测的空间精度。
#         # 2. 标准卷积层（Conv2d）：将特征图的通道数转换为单通道。这一步骤将多通道的特征信息整合到一个通道中，形成热图，
#         # 其中每个像素值表示对应位置的目标存在概率或响应强度
#         # 3. 在 nn.Sequential 中，输入数据会按照定义的顺序流过每一层，每一层的输出作为下一层的输入，直到最后得到结果。
#         # 这种方式不需要单独定义 forward 方法
#         # 当输入特征图经过热图预测分支时，首先通过转置卷积层进行上采样，恢复部分空间信息，使特征图的尺寸更接近原始输入图像。
#         # 接着，标准卷积层将多通道的特征整合为单通道的热图，每个像素值表示对应位置的目标响应强度。
#         # 最后，Sigmoid激活函数将这些响应强度转换为概率值，输出热图中每个像素的概率表示目标在该位置存在的可能性。
        
#         self.heatmap_head = nn.Sequential(  # 定义热图预测分支
#             # ConvTranspose2d：一个转置卷积层，用于上采样（使热图的尺寸变大）。
#             nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),  # 上采样一倍
#             # Conv2d：一个标准卷积层，将特征图转化为单通道的热图（给、灰度图）。
#             nn.Conv2d(dim, 1, kernel_size=1, bias=False),  # 变换成单通道
#             nn.Sigmoid()  # 用Sigmoid将输出限制到0-1之间，表示概率。
#         )

#         if self.inout:  # 如果需要in/out输出
#             self.inout_head = nn.Sequential(  # 定义in/out预测头
#                 nn.Linear(self.dim, 128),  # 线性层降维
#                 nn.ReLU(),  # 激活函数，增加非线性。
#                 nn.Dropout(0.1),  # 随机丢弃10%的神经元，防止过拟合。
#                 nn.Linear(128, 1),  # 再映射到1维
#                 nn.Sigmoid()  # Sigmoid输出0-1概率
#             )

#     def forward(self, input):  # 前向传播定义
#         # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # 通过输入的边界框信息（bboxes）计算每张图片中的人数。
#         # input["bboxes"]：输入的数据结构，包含每张图像的边界框信息。
#         # 每张图像可能有多个边界框，每个边界框对应一个检测到的物体（在这里可能是一个人）
#         # len(bbox_list)：计算每张图像的边界框个数，实际上就是统计每张图像中的物体数量。
        
#         num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]  # 记录每张图中的人数（bbox个数）
        
#         # input["images"]：输入的是一批图像。假设输入是形状为 [batch_size, channels, height, width] 的4维tensor。
#         # batch_size 表示图像的数量，channels 是图像的通道数（RGB是3通道），height 和 width 分别是图像的高度和宽度。
#         # x：self.backbone.forward 返回的特征图。假设它是一个形状为 [batch_size, channels_out, height_out, width_out] 的tensor，
#         # 其中 channels_out 是骨干网络输出的通道数，height_out 和 width_out 是输出特征图的高度和宽度。
        
#         x = self.backbone.forward(input["images"])  # 提取输入图片的特征图
#         # self.linear：这里的linear通常是一个1x1卷积层（实际上是线性层），其作用是调整输出特征图的通道数。
#         x = self.linear(x)  # 将特征通道数调整为dim，一个 1x1 的卷积层，将骨干网络输出的通道数调整为 dim，确保与后续网络兼容。
#         # 将卷积层移动到 GPU
#         # self.depth_conv = nn.Conv2d(1, self.dim, kernel_size=1).to(device)
#         # self.depth_conv = nn.Conv2d(1, self.dim, kernel_size=1)
        
#         self.depth_conv = nn.Conv2d(1, self.dim, kernel_size=1).cuda()  # 将卷积层初始化为 GPU 版本

#          # 获取并处理深度图
#         # depth_map = input["depth_map"].to(self.depth_conv.weight.device)  # 确保深度图在 GPU 上
        
#         depth_map = input["depth_map"]  # 形状为 [batch_size, 1, height, width]
#         depth_features = self.depth_conv(depth_map)  # 将深度图转化为与特征图相同维度的特征
#           # 确保深度图特征和图像特征图在空间维度上匹配
#         if depth_features.shape[2:] != x.shape[2:]:
#             # 调整深度图特征的尺寸，使其与图像特征图的尺寸一致
#             depth_features = F.interpolate(depth_features, size=x.shape[2:], mode='bilinear', align_corners=False)
            
#         # # 增加
#         # # 在通道维度上拼接图像特征和深度特征
#         # x = torch.cat([x, depth_features], dim=1)  # 拼接特征

#         # # 定义融合卷积层，将拼接后的通道数转换为指定维度
#         # fusion_dim = self.dim  # 可以根据需要调整融合后的通道数
#         # self.fusion_conv = nn.Conv2d(x.shape[1], fusion_dim, kernel_size=3, padding=1).cuda()
#         # x = self.fusion_conv(x)

#         # # 可选：添加批归一化和激活函数
#         # self.fusion_bn = nn.BatchNorm2d(fusion_dim).cuda()
#         # x = self.fusion_bn(x)
#         # x = self.relu(x)
#         # # # 
        
#         # 融合深度信息和图像特征图
#         # x = x + depth_features  # 加法融合深度信息与图像特征图
        
#         # 尝试2
#         # 定义多模态特征交互模块
#         class ModalityInteraction(nn.Module):
#             def __init__(self, dim):
#                super(ModalityInteraction, self).__init__()
#                self.image_to_depth = nn.Conv2d(dim, dim, kernel_size=1)
#                self.depth_to_image = nn.Conv2d(dim, dim, kernel_size=1)
#                self.softmax = nn.Softmax(dim=1)

#             def forward(self, image_features, depth_features):
#         # 图像特征对深度特征的注意力
#                depth_attn = self.softmax(self.image_to_depth(image_features))
#         # 深度特征对图像特征的注意力
#                image_attn = self.softmax(self.depth_to_image(depth_features))
#         # 特征交互
#                image_features = image_features + depth_features * image_attn
#                depth_features = depth_features + image_features * depth_attn
#                return image_features, depth_features

#         # 使用特征交互模块
#         self.interaction = ModalityInteraction(self.dim).cuda() 
#         x, depth_features = self.interaction(x, depth_features)

#         # 直接相加融合深度信息与图像特征图
#         x = x + depth_features  # 加法融合深度信息与图像特征图
        
    
                
#         # 位置编码（Positional Encoding）：由于卷积层和线性层通常不会保留图像中的空间位置信息，
#         # 尤其是当特征图变得越来越小的时候。为了让模型知道每个位置的信息（例如，图像中的某个区域的特征），引入了位置编码。
#         # self.pos_embed：这是一个与输入特征图大小相同的tensor，包含了每个位置的编码信息。
#         # 通常是通过一些学习得到的，或者使用固定的方式（例如，正弦余弦函数）生成。
#         x = x + self.pos_embed  # 加入位置编码信息
        
#         # utils.repeat_tensors(x, num_ppl_per_img)：这个函数的作用是根据每张图像的人数（即边界框的数量），将特征图进行复制，
#         # 确保每个人物都有对应的特征。 每个边界框（即每个人物）都需要独立的特征，以便模型能够为每个人物生成不同的输出。
#         # 举个例子，如果图像1有2个人，图像2有1个人，x会分别复制2次和1次，确保每个人物都有相应的特征图。
        
#         x = utils.repeat_tensors(x, num_ppl_per_img)  # 根据人数将特征复制（每个人头一份特征）
        
#         # 提取边界框信息：从输入数据中提取 bboxes（边界框）信息，这些边界框定义了图像中目标对象的位置。
#         # self.get_input_head_maps(input["bboxes"])：这个函数的作用是根据边界框生成头部掩码。
#         # 生成头部掩码调：用 self.get_input_head_maps 方法，根据边界框信息生成头部掩码（head maps）。这些掩码用于标记图像中头部区域的位置。
#         # 每个边界框对应一个掩码，标记出物体的头部位置。该函数的输出是每张图像中每个人物的掩码。
#         # torch.cat(..., dim=0)：将所有图像中的头部掩码拼接成一个大的tensor，
#         # 拼接掩码：使用 torch.cat 将生成的头部掩码在第 0 维（通常是批量维度）进行拼接，形成一个完整的张量。
#         # 维度是 [total_heads, height, width]，其中 total_heads 是所有图像和所有人物的总数。
#         # head_maps.to(x.device)：将生成的头部掩码移动到和特征图相同的设备（例如GPU）上，确保它们在同一计算资源上进行操作
#         head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)  # 生成并拼接头部掩码
        
#         # head_maps.unsqueeze(dim=1)：增加一个维度，使得掩码的形状从 [total_heads, height, width] 
#         # 变为 [total_heads, 1, height, width]，这使得它可以与特征图在维度上对齐。
#         # self.head_token.weight.unsqueeze(-1).unsqueeze(-1)：
#         # self.head_token.weight 是一个可训练的参数，表示每个人物头部的特征信息。通过unsqueeze方法，
#         # 将它扩展为一个与掩码大小匹配的形状（即 [num_heads, channels, height, width]），使得它可以与头部掩码相乘
#         # head_map_embeddings = head_maps * self.head_token.weight：这是将头部掩码信息和头部token特征进行加权融合。
#         head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)  # 将头部掩码编码到特征中
#         x = x + head_map_embeddings  # 特征图中融合头部位置信息
#         # flatten(start_dim=2)：将特征图展平成序列，从第2维开始展平（即将height和width展平成一维）。
#         # 展平后，x 的形状变为 [batch_size, height * width, channels]。
#         # permute(0, 2, 1)：调整维度顺序，将 [batch_size, height * width, channels] 转换为 [batch_size, channels, height * width]，这是Transformer模块所需的输入格式。
#         x = x.flatten(start_dim=2).permute(0, 2, 1)  # 将空间维度展平成序列：b (h*w) c -> b c h w

#         if self.inout:  # 如果需要in/out预测
#         # torch.cat(..., dim=1)：将in/out token添加到序列的开头。这个token用于让模型学习物体是否进入或退出场景。
#             x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)  # 添加in/out专用token
        
#         x = self.transformer(x)  # 将序列送入Transformer模块

#         if self.inout:  # 如果需要in/out预测
#             inout_tokens = x[:, 0, :]  # 取出第一个token作为in/out特征
#             inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)  # 进行in/out预测
#             inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)  # 将预测结果按照每张图分开
#             x = x[:, 1:, :]  # 去掉in/out token，只保留场景tokens
#         x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)  # 将序列恢复成特征图 b (h*w) c -> b c h w
#         x = self.heatmap_head(x).squeeze(dim=1)  # 生成单通道热图
#         x = torchvision.transforms.functional.resize(x, self.out_size)  # 将热图调整到目标尺寸
        
#         # utils.split_tensors(x, num_ppl_per_img)：将生成的热图分开，按每张图像中的人数分割，使得每个人物都有独立的热图。

#         heatmap_preds = utils.split_tensors(x, num_ppl_per_img)  # 将热图分开成按图像组织的list

#         return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}  # 返回热图和in/out预测结果

#     def get_input_head_maps(self, bboxes):  # 生成头部掩码图
#         head_maps = []  # 初始化头部掩码列表
#         for bbox_list in bboxes:  # 遍历每张图的头部框
#             img_head_maps = []  # 当前图片的头部掩码列表
#             for bbox in bbox_list:  # 遍历每个人的头部框
#                 if bbox is None:  # 如果没有头部框
#      # 遍历图像中的每个边界框（bbox），如果 bbox 为 None，则说明该图像没有有效的头部框，生成一个全零的掩码。
#      # torch.zeros(self.featmap_h, self.featmap_w) 创建一个全零的张量，大小为 featmap_h 和 featmap_w，对应特征图的高度和宽度。
#                     img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))  # 填充全零图
#                 else:
#     # 如果 bbox 存在，xmin, ymin, xmax, ymax 分别是头部框的左上角和右下角坐标（通常是归一化的坐标，范围在 [0, 1]）。
#     # 将这些归一化坐标转换为实际的像素坐标，通过将坐标乘以特征图的宽度和高度，然后四舍五入为整数
#                     xmin, ymin, xmax, ymax = bbox  # 提取框的坐标
#                     width, height = self.featmap_w, self.featmap_h  # 特征图尺寸
#                     xmin = round(xmin * width)  # 归一化到特征图尺寸
#                     ymin = round(ymin * height)
#                     xmax = round(xmax * width)
#                     ymax = round(ymax * height)
#     # 创建一个全零的掩码 head_map，大小为特征图的高度和宽度。
#     # 然后，将头部框的区域（通过 [ymin:ymax, xmin:xmax] 选择区域）置为 1，表示该区域是头部所在的区域.
#                     head_map = torch.zeros((height, width))  # 初始化全零掩码
#                     head_map[ymin:ymax, xmin:xmax] = 1  # 将bbox区域置1
#                     img_head_maps.append(head_map)  # 保存头部掩码
#     # 使用 torch.stack(img_head_maps) 将当前图像的多个头部掩码堆叠成一个张量，保存在 head_maps 列表中。
#     # head_maps 列表中会包含每张图像的头部掩码。
#             head_maps.append(torch.stack(img_head_maps))  # 合并所有人头部掩码
#         return head_maps  # 返回最终头部掩码列表

#     def get_gazelle_state_dict(self, include_backbone=False):  # 导出模型权重
#         if include_backbone:  # 是否包含骨干网络
#             return self.state_dict()
#         else:
#             return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone")}  # 排除掉骨干网络的参数

#     def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):  # 加载已有的权重
#         current_state_dict = self.state_dict()  # 当前模型参数
#         keys1 = current_state_dict.keys()
#         keys2 = ckpt_state_dict.keys()

#         if not include_backbone:  # 如果不包含backbone
#             keys1 = set([k for k in keys1 if not k.startswith("backbone")])
#             keys2 = set([k for k in keys2 if not k.startswith("backbone")])
#         else:
#             keys1 = set(keys1)
#             keys2 = set(keys2)

#         if len(keys2 - keys1) > 0:  # 检查多余参数
#             print("WARNING unused keys in provided state dict: ", keys2 - keys1)
#         if len(keys1 - keys2) > 0:  # 检查缺失参数
#             print("WARNING provided state dict does not have values for keys: ", keys1 - keys2)

#         for k in list(keys1 & keys2):  # 加载公共部分
#             current_state_dict[k] = ckpt_state_dict[k]
        
#         self.load_state_dict(current_state_dict, strict=False)  # 非严格模式加载参数


# # 定义2D位置编码函数
# def positionalencoding2d(d_model, height, width):
#     if d_model % 4 != 0:  # 检查d_model是否是4的倍数
#         raise ValueError("位置编码要求d_model为4的倍数")
#     pe = torch.zeros(d_model, height, width)  # 初始化位置编码张量
#     d_model = int(d_model / 2)  # 每一方向占用d_model的一半
#     div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))  # 计算缩放因子
#     pos_w = torch.arange(0., width).unsqueeze(1)  # 列方向位置
#     pos_h = torch.arange(0., height).unsqueeze(1)  # 行方向位置
#     pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)  # 宽度方向sin编码
#     pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)  # 宽度方向cos编码
#     pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)  # 高度方向sin编码
#     pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)  # 高度方向cos编码
#     return pe  # 返回位置编码


# # 定义GazeLLE模型工厂函数
# def get_gazelle_model(model_name):
#     factory = {  # 定义模型名字到具体实例函数的映射
#         "gazelle_dinov2_vits14": gazelle_dinov2_vits14,
#         "gazelle_dinov2_vitb14": gazelle_dinov2_vitb14,
#         "gazelle_dinov2_vitl14": gazelle_dinov2_vitl14,
#         "gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout,
#         "gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout,
#     }
#     assert model_name in factory.keys(), "invalid model name"  # 检查输入模型名是否有效
#     return factory[model_name]()  # 返回实例化后的模型和transform

# # 定义具体的模型构建函数
# def gazelle_dinov2_vits14():
#     backbone = DinoV2Backbone('dinov2_vits14')  # 使用ViT-S14骨干
#     transform = backbone.get_transform((448, 448))  # 获取输入预处理
#     model = GazeLLE(backbone)  # 创建模型
#     return model, transform

# def gazelle_dinov2_vitb14():
#     backbone = DinoV2Backbone('dinov2_vitb14')  # 使用ViT-B14骨干
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone)
#     return model, transform

# def gazelle_dinov2_vitl14():
#     backbone = DinoV2Backbone('dinov2_vitl14')  # 使用ViT-L14骨干
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone)
#     return model, transform

# def gazelle_dinov2_vitb14_inout():
#     backbone = DinoV2Backbone('dinov2_vitb14')  # 使用ViT-B14骨干，带in/out输出
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone, inout=True)
#     return model, transform

# def gazelle_dinov2_vitl14_inout():
#     backbone = DinoV2Backbone('dinov2_vitl14')  # 使用ViT-L14骨干，带in/out输出
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone, inout=True)
#     return model, transform



# 改变22
# self.weight_image = nn.Parameter(torch.tensor(0.5))  # 图像特征的权重
 # self.weight_depth = nn.Parameter(torch.tensor(0.5))  # 深度特征的权重
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.vision_transformer import Block
# import math
# import gazelle.utils as utils
# from gazelle.backbone import DinoV2Backbone
# import torchvision


# # 定义特征加权融合模块
# class FeatureWeightedFusion(nn.Module):
#     """
#     特征加权融合模块，用于对图像特征和深度特征进行加权融合。
#     """
#     def __init__(self, dim):
#         super(FeatureWeightedFusion, self).__init__()
#         # 初始化图像和深度特征的权重
#         self.weight_image = nn.Parameter(torch.tensor(0.5))  # 图像特征的权重
#         self.weight_depth = nn.Parameter(torch.tensor(0.5))  # 深度特征的权重

#     def forward(self, image_features, depth_features):
#         # 对图像和深度特征进行加权
#         weighted_image = image_features * self.weight_image
#         weighted_depth = depth_features * self.weight_depth
#         # 返回加权后的融合特征
#         return weighted_image + weighted_depth


# # 定义特征校准模块
# class FeatureCalibration(nn.Module):
#     """
#     特征校准模块，用于对特征进行通道级别的校准。
#     """
#     def __init__(self, dim):
#         super(FeatureCalibration, self).__init__()
#         # 1x1卷积用于特征校准
#         self.conv = nn.Conv2d(dim, dim, kernel_size=1)
#         # Sigmoid激活函数
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, features):
#         # 生成校准权重并校准特征
#         calibrated = self.sigmoid(self.conv(features))
#         return features * calibrated


# # 定义特征增强模块
# class FeatureEnhancement(nn.Module):
#     """
#     特征增强模块，通过卷积操作增强特征的表达能力。
#     """
#     def __init__(self, dim):
#         super(FeatureEnhancement, self).__init__()
#         # 两个3x3卷积用于特征增强
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         # ReLU激活函数
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, features):
#         enhanced = self.relu(self.conv1(features))
#         enhanced = self.conv2(enhanced)
#         # 返回增强后的特征与原始特征的和（残差连接）
#         return features + enhanced


# # 定义注意力引导的特征融合模块
# class AttentionFusion(nn.Module):
#     """
#     注意力引导的特征融合模块，利用注意力机制对图像特征和深度特征进行融合。
#     """
#     def __init__(self, dim):
#         super(AttentionFusion, self).__init__()
#         # 注意力机制
#         self.attention = nn.Sequential(
#             nn.Conv2d(dim, 1, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, image_features, depth_features):
#         # 计算注意力权重
#         attention = self.attention(image_features)
#         # 根据注意力权重融合图像特征和深度特征
#         fused_features = image_features * attention + depth_features * (1 - attention)
#         return fused_features


# # 定义多模态特征交互模块
# class ModalityInteraction(nn.Module):
#     """
#     多模态特征交互模块，用于图像特征和深度特征的交互。
#     """
#     def __init__(self, dim):
#         super(ModalityInteraction, self).__init__()
#         # 卷积层用于特征交互
#         self.image_to_depth = nn.Conv2d(dim, dim, kernel_size=1)
#         self.depth_to_image = nn.Conv2d(dim, dim, kernel_size=1)
#         # Softmax用于生成注意力权重
#         self.softmax = nn.Softmax(dim=1)
#         # 多头注意力机制
#         self.image_attn = nn.MultiheadAttention(dim, num_heads=8)
#         self.depth_attn = nn.MultiheadAttention(dim, num_heads=8)

#     def forward(self, image_features, depth_features):
#         # 特征交互操作
#         depth_attn = self.softmax(self.image_to_depth(image_features))
#         image_attn = self.softmax(self.depth_to_image(depth_features))
#         image_features = image_features + depth_features * image_attn
#         depth_features = depth_features + image_features * depth_attn

#         # 注意力机制操作
#         batch_size, embedding_dim, height, width = image_features.shape
#         image_features = image_features.reshape(batch_size, embedding_dim, -1).permute(0, 2, 1)
#         depth_features = depth_features.reshape(batch_size, embedding_dim, -1).permute(0, 2, 1)

#         image_features = self.image_attn(image_features, image_features, image_features)[0]
#         depth_features = self.depth_attn(depth_features, depth_features, depth_features)[0]

#         image_features = image_features.permute(0, 2, 1).reshape(batch_size, embedding_dim, height, width)
#         depth_features = depth_features.permute(0, 2, 1).reshape(batch_size, embedding_dim, height, width)

#         return image_features, depth_features


# class GazeLLE(nn.Module):
#     """
#     主模型类，整合了多模态特征融合、特征校准、特征增强、注意力引导融合和Transformer编码等功能。
#     """
#     def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
#         super().__init__()
#         self.backbone = backbone
#         self.dim = dim
#         self.num_layers = num_layers
#         self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
#         self.in_size = in_size
#         self.out_size = out_size
#         self.inout = inout
        
#         # 将主干网络输出特征转换为目标维度
#         self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
#         # 头部特征嵌入
#         self.head_token = nn.Embedding(1, self.dim)
#         # ReLU激活函数
#         self.relu = nn.ReLU(inplace=True)
#         # 位置编码
#         self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w).squeeze())
        
#         # 如果进行进出预测，初始化相应的嵌入层和头
#         if self.inout:
#             self.inout_token = nn.Embedding(1, self.dim)
        
#         # 使用Dropout防止过拟合
#         self.dropout = nn.Dropout(0.1)
        
#         # Transformer模块
#         self.transformer = nn.Sequential(*[
#             Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
#             for i in range(num_layers)
#         ])

#         # 热图预测头
#         self.heatmap_head = nn.Sequential(
#             nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
#             nn.Conv2d(dim, 1, kernel_size=1, bias=False),
#             nn.Sigmoid()
#         )

#         # 如果进行进出预测，初始化相应的头
#         if self.inout:
#             self.inout_head = nn.Sequential(
#                 nn.Linear(self.dim, 128),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(128, 1),
#                 nn.Sigmoid()
#             )

#         # 初始化多模态特征交互模块
#         self.interaction = ModalityInteraction(self.dim)
#         # 初始化深度特征卷积层
#         self.depth_conv = nn.Conv2d(1, self.dim, kernel_size=1)
#         # 初始化特征加权融合模块
#         self.fusion = FeatureWeightedFusion(self.dim)
#         # 初始化特征校准模块
#         self.calibration = FeatureCalibration(self.dim)
#         # 初始化特征增强模块
#         self.enhancement = FeatureEnhancement(self.dim)
#         # 初始化注意力引导的特征融合模块
#         self.attention_fusion = AttentionFusion(self.dim)

#     def forward(self, input):
#         # 获取每张图像中的人数
#         num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        
#         # 图像特征提取
#         x = self.backbone.forward(input["images"])
#         x = self.linear(x)
#         x = self.dropout(x)

#         # 深度特征提取
#         depth_map = input["depth_map"]
#         if depth_map.shape[1] != 1:
#             raise ValueError("Depth map must have 1 channel")
#         depth_features = self.depth_conv(depth_map)
#         depth_features = self.dropout(depth_features)

#         # 特征对齐
#         if depth_features.shape[2:] != x.shape[2:]:
#             depth_features = F.interpolate(depth_features, size=x.shape[2:], mode='bilinear', align_corners=False)
        
#         # 特征交互
#         x, depth_features = self.interaction(x, depth_features)

#         # 特征融合
#         x = self.fusion(x, depth_features)
#         x = self.calibration(x)
#         x = self.enhancement(x)
#         x = self.attention_fusion(x, depth_features)

#         # 位置编码
#         x = x + self.pos_embed
        
#         # 重复张量
#         x = utils.repeat_tensors(x, num_ppl_per_img)
        
#         # 头部特征图
#         head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
#         head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
#         x = x + head_map_embeddings
#         x = x.flatten(start_dim=2).permute(0, 2, 1)

#         # 处理进出预测
#         if self.inout:
#             x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)
        
#         # Transformer编码
#         x = self.transformer(x)

#         # 生成预测结果
#         if self.inout:
#             inout_tokens = x[:, 0, :]
#             inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
#             inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
#             x = x[:, 1:, :]

#         # 热图预测
#         x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
#         x = self.heatmap_head(x).squeeze(dim=1)
#         x = torchvision.transforms.functional.resize(x, self.out_size)
        
#         heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

#         return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

#     def get_input_head_maps(self, bboxes):
#         # 根据边界框生成头部特征图
#         head_maps = []
#         for bbox_list in bboxes:
#             img_head_maps = []
#             for bbox in bbox_list:
#                 if bbox is None:
#                     img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))
#                 else:
#                     xmin, ymin, xmax, ymax = bbox
#                     width, height = self.featmap_w, self.featmap_h
#                     xmin = round(xmin * width)
#                     ymin = round(ymin * height)
#                     xmax = round(xmax * width)
#                     ymax = round(ymax * height)
#                     head_map = torch.zeros((height, width))
#                     head_map[ymin:ymax, xmin:xmax] = 1
#                     img_head_maps.append(head_map)
#             head_maps.append(torch.stack(img_head_maps))
#         return head_maps

#     def get_gazelle_state_dict(self, include_backbone=False):
#         # 获取模型状态字典
#         if include_backbone:
#             return self.state_dict()
#         else:
#             return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone")}

#     def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
#         # 加载模型状态字典
#         current_state_dict = self.state_dict()
#         keys1 = current_state_dict.keys()
#         keys2 = ckpt_state_dict.keys()

#         if not include_backbone:
#             keys1 = set([k for k in keys1 if not k.startswith("backbone")])
#             keys2 = set([k for k in keys2 if not k.startswith("backbone")])
#         else:
#             keys1 = set(keys1)
#             keys2 = set(keys2)

#         if len(keys2 - keys1) > 0:
#             print("WARNING unused keys in provided state dict: ", keys2 - keys1)
#         if len(keys1 - keys2) > 0:
#             print("WARNING provided state dict does not have values for keys: ", keys1 - keys2)

#         for k in list(keys1 & keys2):
#             current_state_dict[k] = ckpt_state_dict[k]
        
#         self.load_state_dict(current_state_dict, strict=False)


# def positionalencoding2d(d_model, height, width):
#     """
#     生成2D位置编码。
#     """
#     if d_model % 4 != 0:
#         raise ValueError("位置编码要求d_model为4的倍数")
#     pe = torch.zeros(d_model, height, width)
#     d_model = int(d_model / 2)
#     div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
#     pos_w = torch.arange(0., width).unsqueeze(1)
#     pos_h = torch.arange(0., height).unsqueeze(1)
#     pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     return pe


# def get_gazelle_model(model_name):
#     """
#     获取GazeLLE模型。
#     """
#     factory = {
#         "gazelle_dinov2_vits14": gazelle_dinov2_vits14,
#         "gazelle_dinov2_vitb14": gazelle_dinov2_vitb14,
#         "gazelle_dinov2_vitl14": gazelle_dinov2_vitl14,
#         "gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout,
#         "gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout,
#     }
#     assert model_name in factory.keys(), "invalid model name"
#     return factory[model_name]()


# def gazelle_dinov2_vits14():
#     """
#     创建gazelle_dinov2_vits14模型。
#     """
#     backbone = DinoV2Backbone('dinov2_vits14')
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone)
#     return model, transform

# def gazelle_dinov2_vitb14():
#     """
#     创建gazelle_dinov2_vitb14模型。
#     """
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone)
#     return model, transform

# def gazelle_dinov2_vitl14():
#     """
#     创建gazelle_dinov2_vitl14模型。
#     """
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone)
#     return model, transform

# def gazelle_dinov2_vitb14_inout():
#     """
#     创建gazelle_dinov2_vitb14_inout模型。
#     """
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone, inout=True)
#     return model, transform

# def gazelle_dinov2_vitl14_inout():
#     """
#     创建gazelle_dinov2_vitl14_inout模型。
#     """
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone, inout=True)
#     return model, transform

# 随机权重
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.vision_transformer import Block
# import math
# import gazelle.utils as utils
# from gazelle.backbone import DinoV2Backbone
# import torchvision


# # 定义特征加权融合模块
# class FeatureWeightedFusion(nn.Module):
#     """
#     特征加权融合模块，用于对图像特征和深度特征进行加权融合。
#     """
#     def __init__(self, dim):
#         super(FeatureWeightedFusion, self).__init__()
#         # 初始化图像和深度特征的权重，使用均匀分布初始化
#         self.weight_image = nn.Parameter(torch.rand(1))
#         self.weight_depth = nn.Parameter(torch.rand(1))

#     def forward(self, image_features, depth_features):
#         # 对图像和深度特征进行加权
#         weighted_image = image_features * self.weight_image
#         weighted_depth = depth_features * self.weight_depth
#         # 返回加权后的融合特征
#         return weighted_image + weighted_depth


# # 定义特征校准模块
# class FeatureCalibration(nn.Module):
#     """
#     特征校准模块，用于对特征进行通道级别的校准。
#     """
#     def __init__(self, dim):
#         super(FeatureCalibration, self).__init__()
#         # 1x1卷积用于特征校准
#         self.conv = nn.Conv2d(dim, dim, kernel_size=1)
#         # Sigmoid激活函数
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, features):
#         # 生成校准权重并校准特征
#         calibrated = self.sigmoid(self.conv(features))
#         return features * calibrated


# # 定义特征增强模块
# class FeatureEnhancement(nn.Module):
#     """
#     特征增强模块，通过卷积操作增强特征的表达能力。
#     """
#     def __init__(self, dim):
#         super(FeatureEnhancement, self).__init__()
#         # 两个3x3卷积用于特征增强，增加卷积层的数量
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)  # 添加更多卷积层
#         # ReLU激活函数
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, features):
#         enhanced = self.relu(self.conv1(features))
#         enhanced = self.relu(self.conv2(enhanced))
#         enhanced = self.conv3(enhanced)  # 使用更多卷积层
#         # 返回增强后的特征与原始特征的和（残差连接）
#         return features + enhanced


# # 定义注意力引导的特征融合模块
# class AttentionFusion(nn.Module):
#     """
#     注意力引导的特征融合模块，利用注意力机制对图像特征和深度特征进行融合。
#     """
#     def __init__(self, dim):
#         super(AttentionFusion, self).__init__()
#         # 注意力机制，增加注意力通道数
#         self.attention = nn.Sequential(
#             nn.Conv2d(dim, dim // 2, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(dim // 2, 1, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, image_features, depth_features):
#         # 计算注意力权重
#         attention = self.attention(image_features)
#         # 根据注意力权重融合图像特征和深度特征
#         fused_features = image_features * attention + depth_features * (1 - attention)
#         return fused_features


# # 定义多模态特征交互模块
# class ModalityInteraction(nn.Module):
#     """
#     多模态特征交互模块，用于图像特征和深度特征的交互。
#     """
#     def __init__(self, dim):
#         super(ModalityInteraction, self).__init__()
#         # 卷积层用于特征交互
#         self.image_to_depth = nn.Conv2d(dim, dim, kernel_size=1)
#         self.depth_to_image = nn.Conv2d(dim, dim, kernel_size=1)
#         # Softmax用于生成注意力权重
#         self.softmax = nn.Softmax(dim=1)
#         # 多头注意力机制，增加注意力头数
#         self.image_attn = nn.MultiheadAttention(dim, num_heads=16)
#         self.depth_attn = nn.MultiheadAttention(dim, num_heads=16)

#     def forward(self, image_features, depth_features):
#         # 特征交互操作
#         depth_attn = self.softmax(self.image_to_depth(image_features))
#         image_attn = self.softmax(self.depth_to_image(depth_features))
#         image_features = image_features + depth_features * image_attn
#         depth_features = depth_features + image_features * depth_attn

#         # 注意力机制操作
#         batch_size, embedding_dim, height, width = image_features.shape
#         image_features = image_features.reshape(batch_size, embedding_dim, -1).permute(0, 2, 1)
#         depth_features = depth_features.reshape(batch_size, embedding_dim, -1).permute(0, 2, 1)

#         image_features = self.image_attn(image_features, image_features, image_features)[0]
#         depth_features = self.depth_attn(depth_features, depth_features, depth_features)[0]

#         image_features = image_features.permute(0, 2, 1).reshape(batch_size, embedding_dim, height, width)
#         depth_features = depth_features.permute(0, 2, 1).reshape(batch_size, embedding_dim, height, width)

#         return image_features, depth_features


# class GazeLLE(nn.Module):
#     """
#     主模型类，整合了多模态特征融合、特征校准、特征增强、注意力引导融合和Transformer编码等功能。
#     """
#     def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
#         super().__init__()
#         self.backbone = backbone
#         self.dim = dim
#         self.num_layers = num_layers
#         self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
#         self.in_size = in_size
#         self.out_size = out_size
#         self.inout = inout
        
#         # 将主干网络输出特征转换为目标维度
#         self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
#         # 头部特征嵌入
#         self.head_token = nn.Embedding(1, self.dim)
#         # ReLU激活函数
#         self.relu = nn.ReLU(inplace=True)
#         # 位置编码
#         self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w).squeeze())
        
#         # 如果进行进出预测，初始化相应的嵌入层和头
#         if self.inout:
#             self.inout_token = nn.Embedding(1, self.dim)
        
#         # 使用Dropout防止过拟合
#         self.dropout = nn.Dropout(0.1)
        
#         # Transformer模块，增加模块数量
#         self.transformer = nn.Sequential(*[
#             Block(dim=self.dim, num_heads=16, mlp_ratio=4, drop_path=0.1)
#             for i in range(num_layers)
#         ])

#         # 热图预测头
#         self.heatmap_head = nn.Sequential(
#             nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
#             nn.Conv2d(dim, 1, kernel_size=1, bias=False),
#             nn.Sigmoid()
#         )

#         # 如果进行进出预测，初始化相应的头
#         if self.inout:
#             self.inout_head = nn.Sequential(
#                 nn.Linear(self.dim, 128),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(128, 1),
#                 nn.Sigmoid()
#             )

#         # 初始化多模态特征交互模块
#         self.interaction = ModalityInteraction(self.dim)
#         # 初始化深度特征卷积层
#         self.depth_conv = nn.Conv2d(1, self.dim, kernel_size=1)
#         # 初始化特征加权融合模块
#         self.fusion = FeatureWeightedFusion(self.dim)
#         # 初始化特征校准模块
#         self.calibration = FeatureCalibration(self.dim)
#         # 初始化特征增强模块
#         self.enhancement = FeatureEnhancement(self.dim)
#         # 初始化注意力引导的特征融合模块
#         self.attention_fusion = AttentionFusion(self.dim)

#     def forward(self, input):
#         # 获取每张图像中的人数
#         num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        
#         # 图像特征提取
#         x = self.backbone.forward(input["images"])
#         x = self.linear(x)
#         x = self.dropout(x)

#         # 深度特征提取
#         depth_map = input["depth_map"]
#         if depth_map.shape[1] != 1:
#             raise ValueError("Depth map must have 1 channel")
#         depth_features = self.depth_conv(depth_map)
#         depth_features = self.dropout(depth_features)

#         # 特征对齐
#         if depth_features.shape[2:] != x.shape[2:]:
#             depth_features = F.interpolate(depth_features, size=x.shape[2:], mode='bilinear', align_corners=False)
        
#         # 特征交互
#         x, depth_features = self.interaction(x, depth_features)

#         # 特征融合
#         x = self.fusion(x, depth_features)
#         x = self.calibration(x)
#         x = self.enhancement(x)
#         x = self.attention_fusion(x, depth_features)

#         # 位置编码
#         x = x + self.pos_embed
        
#         # 重复张量
#         x = utils.repeat_tensors(x, num_ppl_per_img)
        
#         # 头部特征图
#         head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
#         head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
#         x = x + head_map_embeddings
#         x = x.flatten(start_dim=2).permute(0, 2, 1)

#         # 处理进出预测
#         if self.inout:
#             x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)
        
#         # Transformer编码
#         x = self.transformer(x)

#         # 生成预测结果
#         if self.inout:
#             inout_tokens = x[:, 0, :]
#             inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
#             inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
#             x = x[:, 1:, :]

#         # 热图预测
#         x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
#         x = self.heatmap_head(x).squeeze(dim=1)
#         x = torchvision.transforms.functional.resize(x, self.out_size)
        
#         heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

#         return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

#     def get_input_head_maps(self, bboxes):
#         # 根据边界框生成头部特征图
#         head_maps = []
#         for bbox_list in bboxes:
#             img_head_maps = []
#             for bbox in bbox_list:
#                 if bbox is None:
#                     img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))
#                 else:
#                     xmin, ymin, xmax, ymax = bbox
#                     width, height = self.featmap_w, self.featmap_h
#                     xmin = round(xmin * width)
#                     ymin = round(ymin * height)
#                     xmax = round(xmax * width)
#                     ymax = round(ymax * height)
#                     head_map = torch.zeros((height, width))
#                     head_map[ymin:ymax, xmin:xmax] = 1
#                     img_head_maps.append(head_map)
#             head_maps.append(torch.stack(img_head_maps))
#         return head_maps

#     def get_gazelle_state_dict(self, include_backbone=False):
#         # 获取模型状态字典
#         if include_backbone:
#             return self.state_dict()
#         else:
#             return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone")}

#     def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
#         # 加载模型状态字典
#         current_state_dict = self.state_dict()
#         keys1 = current_state_dict.keys()
#         keys2 = ckpt_state_dict.keys()

#         if not include_backbone:
#             keys1 = set([k for k in keys1 if not k.startswith("backbone")])
#             keys2 = set([k for k in keys2 if not k.startswith("backbone")])
#         else:
#             keys1 = set(keys1)
#             keys2 = set(keys2)

#         if len(keys2 - keys1) > 0:
#             print("WARNING unused keys in provided state dict: ", keys2 - keys1)
#         if len(keys1 - keys2) > 0:
#             print("WARNING provided state dict does not have values for keys: ", keys1 - keys2)

#         for k in list(keys1 & keys2):
#             current_state_dict[k] = ckpt_state_dict[k]
        
#         self.load_state_dict(current_state_dict, strict=False)


# def positionalencoding2d(d_model, height, width):
#     """
#     生成2D位置编码。
#     """
#     if d_model % 4 != 0:
#         raise ValueError("位置编码要求d_model为4的倍数")
#     pe = torch.zeros(d_model, height, width)
#     d_model = int(d_model / 2)
#     div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
#     pos_w = torch.arange(0., width).unsqueeze(1)
#     pos_h = torch.arange(0., height).unsqueeze(1)
#     pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     return pe


# def get_gazelle_model(model_name):
#     """
#     获取GazeLLE模型。
#     """
#     factory = {
#         "gazelle_dinov2_vits14": gazelle_dinov2_vits14,
#         "gazelle_dinov2_vitb14": gazelle_dinov2_vitb14,
#         "gazelle_dinov2_vitl14": gazelle_dinov2_vitl14,
#         "gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout,
#         "gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout,
#     }
#     assert model_name in factory.keys(), "invalid model name"
#     return factory[model_name]()


# def gazelle_dinov2_vits14():
#     """
#     创建gazelle_dinov2_vits14模型。
#     """
#     backbone = DinoV2Backbone('dinov2_vits14')
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone)
#     return model, transform

# def gazelle_dinov2_vitb14():
#     """
#     创建gazelle_dinov2_vitb14模型。
#     """
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone)
#     return model, transform

# def gazelle_dinov2_vitl14():
#     """
#     创建gazelle_dinov2_vitl14模型。
#     """
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone)
#     return model, transform

# def gazelle_dinov2_vitb14_inout():
#     """
#     创建gazelle_dinov2_vitb14_inout模型。
#     """
#     backbone = DinoV2Backbone('dinov2_vitb14')
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone, inout=True)
#     return model, transform

# def gazelle_dinov2_vitl14_inout():
#     """
#     创建gazelle_dinov2_vitl14_inout模型。
#     """
#     backbone = DinoV2Backbone('dinov2_vitl14')
#     transform = backbone.get_transform((448, 448))
#     model = GazeLLE(backbone, inout=True)
#     return model, transform