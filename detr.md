# DETR

![Screenshot from 2025-05-14 09-01-57](https://github.com/user-attachments/assets/9a2ae0ee-873d-4450-a805-9bc98c82da8a)

DETR（DEtection TRansformer）是 Facebook 提出的一种目标检测方法，它将目标检测任务转化为一个集合预测（set prediction）问题，用一种端到端的方式来解决，不依赖于传统的检测器结构（比如 anchor、NMS 等）。

---

## 主要创新点
**端到端的目标检测：**

DETR 抛弃了传统检测器中常见的手工设计组件（例如锚框生成、候选框提议、非极大值抑制等），而是通过一个统一的神经网络结构，直接输出所有物体的边界框和类别。

**使用 Transformer 架构：**

DETR 在 CNN 特征提取器（如 ResNet）后接入一个标准的 Transformer 编码器-解码器架构，利用其强大的全局建模能力来理解图像中的物体及其关系。

**集合匹配损失函数：**

采用了一个基于匈牙利算法的集合匹配损失函数（Hungarian Loss），对预测的边界框与真实框进行一一匹配，避免了重复预测。



---

## 工作原理概述

输入图像通过 CNN 提取特征；

Transformer 编码器建模图像中所有区域的全局关系；

解码器使用一组可学习的“object queries”来并行地预测物体（每个 query 对应一个检测结果）；

最终输出一组固定数量的预测（其中部分为“no object”类别），直接生成所有检测结果。

---

## 优点

简洁统一：DETR 不需要额外设计的候选区域、anchor、NMS 后处理，整体结构非常简洁。

全局建模能力强：Transformer 的自注意力机制能捕捉图像中各区域间的复杂关系，尤其对大型物体检测表现更好。

可扩展性强：仅需加一个 mask 分支，DETR 就可以轻松扩展到全景分割任务。

---

## 缺点和挑战

训练收敛慢：相比传统方法(resnet)需要更长时间的训练(10x epoch)才能达到较好性能（收敛）；

对小物体检测效果较差：由于其特征提取粒度较粗，Transformer 在建模小目标时不够精准；

固定数量的预测槽：其输出是固定数量的 object queries，不适合检测数量极其多的场景。

---

## 🔹 输入

- 尺寸为 `3 × H₀ × W₀` 的彩色图像。
- 模型固定生成 `N` 个预测结果（例如 `N = 100`），其中一部分会对应真实目标，其余被标记为“无目标”。

---

## 🔹 模型结构

#### 1. CNN Backbone（特征提取）

- 通常使用 **ResNet-50** 或 **ResNet-101**。
- 输出为一个低分辨率的二维特征图 `f ∈ ℝ^{C × H × W}`， `C = 2048`（ResNet 最后一层的输出通道数），`H` 和 `W` 是原图的 `1/32`。
- 使用一个 `1 × 1` 卷积将通道数降为 `d`（例如 `d = 256`），得到 `z₀ ∈ ℝ^{d × H × W}`。

#### 2. Transformer Encoder

- 输入是将 `z₀` 展平后的序列（维度为 `d × HW`），加上**固定的空间位置编码**。
- 编码器为标准 Transformer 结构（包含多头自注意力、前馈网络、残差连接和 LayerNorm）。
- 输出是编码后的**全图上下文特征序列**。

#### 3. Transformer Decoder

- 输入是 `N` 个**可学习的 object queries（目标查询向量）**，每个向量长度为 `d`。（维度为 `N × d`）
- Decoder 每一层执行：
  - **自注意力**：学习查询之间的相互关系。
  - **交叉注意力**：查询与 Encoder 输出进行交互，获取图像上下文。
- 输出是 `N × d` 的嵌入向量，每个表示一个目标候选。

#### 4. 预测头（FFN）

- 每个 Decoder 输出向量送入两个小的共享的 3 层前馈神经网络（FFN），输出两个内容：
  - 一个类别（包括“无目标”类别 ∅）
  - 一个边界框（`cx, cy, w, h`，归一化坐标）

---

### 🔹 输出

- `N` 个类别概率（使用 Softmax 分类）
- `N` 个归一化边界框
  
- 后续使用 **匈牙利算法**将预测与 Ground Truth 进行一一匹配，计算总损失（分类损失 + 框损失）。


---

### 整体流程总结

```
images (NestedTensor)
   ↓
CNN Backbone（如 ResNet） ➝ features[-1], pos[-1]
   ↓
1x1 conv input_proj + flatten
   ↓
Transformer Encoder-Decoder（含 learnable object queries）
   ↓
Decoder outputs hs ➝ class_embed + bbox_embed
   ↓
分类结果 pred_logits + 边框 pred_boxes（归一化）
```
---

## 匈牙利匹配

### 🧩 场景设定

我们设定如下：

- **模型预测的目标数（queries） = 3**
- **图像中真实目标数（GT） = 2**

---

### 模型预测输出如下：

| 预测索引 | 类别概率（softmax后）       | 边界框 `(cx, cy, w, h)`      |
|----------|------------------------------|-------------------------------|
| P0       | `[0.1, 0.8, 0.1]`            | `(0.5, 0.5, 0.4, 0.3)`        |
| P1       | `[0.7, 0.2, 0.1]`            | `(0.1, 0.1, 0.2, 0.2)`        |
| P2       | `[0.3, 0.3, 0.4]`            | `(0.9, 0.9, 0.1, 0.1)`        |

> 假设类别顺序为 `[A, B, C]`，共 3 类

---

### 🏷️ Ground Truth（GT）标注如下：

| GT索引 | 类别 | 边界框 `(cx, cy, w, h)`       |
|--------|------|-------------------------------|
| G0     | B    | `(0.52, 0.52, 0.38, 0.28)`     |
| G1     | A    | `(0.15, 0.15, 0.25, 0.25)`     |

---

#### ✅ Step 1: 找出所有GT的类别，计算 分类代价 cost_class  `cost_class = 1 - p(GT类)`

| 预测\GT | G0 (B类)       | G1 (A类)       |
|---------|----------------|----------------|
| P0      | `1 - 0.8 = 0.2`| `1 - 0.1 = 0.9`|
| P1      | `1 - 0.2 = 0.8`| `1 - 0.7 = 0.3`|
| P2      | `1 - 0.3 = 0.7`| `1 - 0.3 = 0.7`|

```
cost_class = -out_prob[:, tgt_ids]
```

---

#### ✅ Step 2: 边界框 L1 距离代价cost_box（粗略估算）不管class，就找最近的bbox

| 预测\GT | G0         | G1         |
|---------|------------|------------|
| P0      | ≈ `0.06`   | ≈ `1.1`    |
| P1      | ≈ `0.9`    | ≈ `0.1`    |
| P2      | ≈ `1.4`    | ≈ `1.5`    |

```
out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size × num_queries, 4]
tgt_bbox = torch.cat([v["boxes"] for v in targets])  # 所有 GT 框
cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
```

---

#### ✅ Step 3: GIoU 代价（负 GIoU）

| 预测\GT | G0       | G1       |
|---------|----------|----------|
| P0      | `-0.8`   | `-0.1`   |
| P1      | `-0.1`   | `-0.9`   |
| P2      | `-0.2`   | `-0.2`   |


```
cost_giou = -generalized_box_iou(
    box_cxcywh_to_xyxy(out_bbox), 
    box_cxcywh_to_xyxy(tgt_bbox)
)
```

---

#### ✅ Step 4: 总匹配代价（加权求和，权重全为 1）   

$$
C(i,j)=λcls​分类代价(i,j)+λL1​L1代价(i,j)+λIoU​GIoU代价(i,j)
$$

| 预测\GT | G0                    | G1                    |
|---------|-----------------------|-----------------------|
| P0      | `0.2 + 0.06 + 0.8 = 1.06` | `0.9 + 1.1 + 0.1 = 2.1`  |
| P1      | `0.8 + 0.9 + 0.1 = 1.8`   | `0.3 + 0.1 + 0.9 = 1.3`  |
| P2      | `0.7 + 1.4 + 0.2 = 2.3`   | `0.7 + 1.5 + 0.2 = 2.4`  |

```
C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
```
---

#### ✅ Step 5: 匈牙利匹配结果（最小代价分配）
在一个 二维代价矩阵C 中，选出 一一对应的匹配（每行最多选一个列，每列最多选一个行），使得 总代价之和最小。【query个数 * class数】

例如下表中，格子中为分类误差 + L1 + GIoU

|         | gt[0]=person1 | gt[1]=person2 | gt[2]=dog |
| ------- | ------------- | ------------- | --------- |
| pred[0] | **0.1**       | 1.5           | 2.0       |
| pred[1] | 2.0           | 1.7           | **0.2**   |
| pred[2] | 1.4           | **0.3**       | 1.8       |
| pred[3] | 1.2           | 1.1           | 2.5       |
| pred[4] | 2.0           | 2.2           | 2.1       |




### 输出
序固定就意味着模型假设“GT1 永远是某个位置/类别的物体”，这对真实数据不成立（目标的顺序会变、目标数会变、类别会变）。

如果强行排序，反而可能导致模型过拟合到排序规则，而不是学会泛化地检测目标。row_ind = [0, 1, 2]
col_ind = [1, 0, 2]

## Step 1：Flatten BEV 特征图 + 自注意力建模
```
import torch
import torch.nn as nn

# 假设输入的 BEV 特征图
C, H, W = 256, 200, 200
bev_feature = torch.randn(C, H, W)  # [C, H, W]

# 转为 [H*W, C]，供 transformer encoder 使用
bev_feature_flat = bev_feature.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]

# 加上 batch dim: [1, H*W, C]
bev_feature_seq = bev_feature_flat.unsqueeze(0)

# 定义 Transformer Encoder
transformer_encoder = nn.TransformerEncoder(
    encoder_layer=nn.TransformerEncoderLayer(
        d_model=C,
        nhead=8,
        dim_feedforward=512,
        dropout=0.1,
        batch_first=True
    ),
    num_layers=6
)

# 输出 Encoder 后的全局建模特征：[1, H*W, C]
encoder_output = transformer_encoder(bev_feature_seq)
```

## ✅ Step 2：用 DETR 的 Object Queries 做 Cross Attention

```
# Learnable object queries：[N_query, C]
N_query = 100
object_queries = nn.Parameter(torch.randn(N_query, C))  # 需要注册到模型中才能训练

# 加上 batch dim: [1, N_query, C]
object_queries = object_queries.unsqueeze(0)

# 定义 Transformer Decoder
transformer_decoder = nn.TransformerDecoder(
    decoder_layer=nn.TransformerDecoderLayer(
        d_model=C,
        nhead=8,
        dim_feedforward=512,
        dropout=0.1,
        batch_first=True
    ),
    num_layers=6
)

# Decoder：Cross Attention(Q=query, K=V=encoder_output)
decoder_output = transformer_decoder(object_queries, encoder_output)
# shape: [1, N_query, C]
‵‵‵
## ✅ 输出分类和边界框回归（DETR-style）

‵‵‵
# 分类头
class_head = nn.Linear(C, 10)  # 10类目标

# 边界框回归头（这里假设输出 [cx, cy, w, h]）
bbox_head = nn.Linear(C, 4)

# 输出
class_logits = class_head(decoder_output)  # [1, 100, 10]
bboxes = bbox_head(decoder_output).sigmoid()  # [1, 100, 4] (归一化坐标)
```
