# Fast_bev
<img width="1385" height="685" alt="Screenshot from 2025-09-15 10-05-24" src="https://github.com/user-attachments/assets/e162e677-fbd6-45a4-842d-24a1b6de6fa0" />


# Muti-scale image encoder

<img width="541" height="696" alt="Screenshot from 2025-09-15 10-01-10" src="https://github.com/user-attachments/assets/e030beca-eed8-4f8c-8af1-d0d4be65a246" />


# 代码流程

backbone → neck(FPN) → backproject → neck_3d

设：
```
batch_size = 12（每个 mini-batch 有 12 组样本）

seq_len = 120（比如时序帧数量）

nv = 6（每个时刻有 6 个相机）

所以原始输入图像组织是：
[12, 120\*6, 3, H, W]

在 extract_feat 里 reshape 成：
[12\*120\*6, 3, H, W]（一次性喂 backbone）。
```

### backbone (resnet18)
```
x = self.backbone(
            img
        ) 
```
**输出：**

P2: [12\*120\*6, **C2, H2, W2**]

P3: [12\*120\*6, **C3, H3, W3**]

P4: [12\*120\*6, **C4, H4, W4**]

P5: [12\*120\*6, **C5, H5, W5**]

## 多尺度特征融合
### neck 多尺度维度统一 (FPN：1x1conv + 逐级 top-down 融合：上采样+相加)
```
def _inner_forward(x):
            out = self.neck(x)
            return out  # P2: [6, 64, 232, 400]

                        # P3: [6, 64, 116, 200]

                        # P4: [6, 64, 58, 100]

                        # P5: [6, 64, 29, 50]
```
**输出：**

P2: [12\*120\*6, **64**, H2, W2]

P3: [12\*120\*6, **64**, H3, W3]

P4: [12\*120\*6, **64**, H4, W4]

P5: [12\*120\*6, **64**, H5, W5]

### neck_fuse 尺寸统一
根据 multi_scale_id，选择某一层（比如 P2 或 P5），把其他层都 resize 到同样大小，然后 concat 在通道维度，最后卷积压缩：
```
for msid in self.multi_scale_id:# multi_scale_id=[0] 为例（表示要在 P2 上做融合）
                # fpn output fusion
                if getattr(self, f'neck_fuse_{msid}', None) is not None:
                    fuse_feats = [mlvl_feats[msid]]
                    for i in range(msid + 1, len(mlvl_feats)):
                        resized_feat = resize(
                            mlvl_feats[i], 
                            size=mlvl_feats[msid].size()[2:], 
                            mode="bilinear", 
                            align_corners=False)
                        fuse_feats.append(resized_feat)
                
                    if len(fuse_feats) > 1:
                        fuse_feats = torch.cat(fuse_feats, dim=1) # 把所有同尺度的 feature 在 通道维度上拼接。
                    else:
                        fuse_feats = fuse_feats[0]
                    fuse_feats = getattr(self, f'neck_fuse_{msid}') # 用一个卷积（neck_fuse_{msid}）做降维或融合。
                    mlvl_feats_.append(fuse_feats)
                else:
                    mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_
```
resize 实际就是 插值函数（调用 torch.nn.functional.interpolate.

在 multi_scale_id=[0] 时：融合 P2 + P3 + P4 + P5

在 multi_scale_id=[1] 时：融合 P3 + P4 + P5

在 multi_scale_id=[2] 时：融合 P4 + P5

在 multi_scale_id=[3] 时：只用 P5（没有更深层可上采样）

**输出：**

假设 multi_scale_id=[0]：输出 [12\*120\*6, 64, H2, W2]

### 举例

**情况一：multi_scale_id=[0]**

👉 表示选择 P2 作为融合基准：P2 [6, 64, 232, 400]，那就把 P3、P4、P5 都 resize 到 (232, 400)

P3 [6, 64, 116, 200] → [6, 64, 232, 400]

P4 [6, 64, 58, 100] → [6, 64, 232, 400]

P5 [6, 64, 29, 50] → [6, 64, 232, 400]

Concat：

[6, 64*4, 232, 400] = [6, 256, 232, 400]

再经过一个 3×3 Conv（neck_fuse_0）降到指定维度，比如 64：

输出： [6, 64, 232, 400]


# projection
```
projection = self._compute_projection(# 得到 projection 矩阵
```
**(a) inplace 模式 (backproject_inplace)**

直接把同一个 voxel 对应的多个相机采样结果 累加 / 覆盖到 voxel 里。

通常是加法（sum）或平均（mean）。

**(b) vanilla 模式 (backproject_vanilla)**

会额外计算一个 valid 掩码：统计每个 voxel 被多少个相机看见。

把所有相机采样的特征先 相加，再除以 valid，做成平均：

输出：
[12, 120*64, 200, 200, 4]


# neck3d
```
def _inner_forward(x):
            # v1/v2: [bs, lvl*seq*c, vx, vy, vz] -> [bs, c', vx, vy] (c' = 192)
            # v3/v4: [bs, z1*c1+z2*c2+..., vx, vy, 1] -> [bs, c', vx, vy]
            # 3D 卷积 或 (Conv3D + BN + ReLU)
            out = self.neck_3d(x)
            return out
```
**输出：**
[12, C’, 200, 200]

# BBox Head
三个头分别预测

cls_scores: [B, 200\*200\*8, num_classes]

bbox_preds: [B, 200\*200\*8, 9] (Δx, Δy, Δz, Δw, Δl, Δh, Δyaw, Δvx, Δvy)

dir_cls_preds: [B, 200\*200\*8, 2] 方向


# bbox loss
bbox_preds: [B, 200\*200\*8, 9] 与 gt 做 **SmoothL1Loss**

```
pred = (x=1.2, y=0.5, z=0.0, w=2.0, l=4.0, h=1.5, θ=0.1, vx=0.0, vy=0.0)
gt =   (x=1.0, y=0.6, z=0.0, w=2.2, l=3.8, h=1.6, θ=0.2, vx=0.0, vy=0.0)
Δ = pred - gt = (0.2, -0.1, 0.0, -0.2, 0.2, -0.1, -0.1, 0.0, 0.0)

L_smooth = Σ_i SmoothL1(Δ_i)
≈ |0.2| + |−0.1| + |0| + |−0.2| + |0.2| + |−0.1| + |−0.1| + 0 + 0
≈ 0.9
```

做完转概率形式：
```
matched_box_prob = torch.exp(-loss_bbox)
```
输出 

```
exp(-0.9) ≈ 0.41
```
**3. 与分类概率融合**
```
matched_cls_prob = 0.8   # anchor 被预测成“汽车”的概率
```

融合：

```
matched_prob = matched_cls_prob * matched_box_prob
             = 0.8 * 0.41
             ≈ 0.33
```
**4. positive_bag_prob**
<img width="315" height="98" alt="Screenshot from 2025-09-19 16-02-49" src="https://github.com/user-attachments/assets/6aa55e96-7249-4360-8ce2-635da99f601a" />


```
# matched_cls_prob: top-k anchors 的分类概率
# matched_box_prob: top-k anchors 的定位概率
matched_prob = matched_cls_prob * matched_box_prob

# 权重归一化
weight = 1 / torch.clamp(1 - matched_prob, 1e-12, None)
weight /= weight.sum(dim=1).unsqueeze(dim=-1)

# bag_prob = top-k anchor 概率的加权和
bag_prob = (weight * matched_prob).sum(dim=1)
bag_prob = bag_prob.clamp(0, 1)

# BCE(bag_prob, 1)
return self.alpha * F.binary_cross_entropy(
    bag_prob, torch.ones_like(bag_prob), reduction='none')
```


**5. negative_bag_prob**

假设：

分类概率 cls_prob = [0.9, 0.6, 0.2, 0.1]   # 每个anchor属于“车”的概率

匹配概率 box_prob = [0.95, 0.2, 0.0, 0.0]

```
prob = cls_prob * (1 - box_prob)
```
prob = [0.9*(1-0.95), 0.6*(1-0.2), 0.2*(1-0.0), 0.1*(1-0.0)]
     = [0.045,         0.48,        0.2,         0.1]

如果 anchor 和 GT 高度重合（box_prob≈1），则 (1-box_prob)≈0，prob≈0 → 减弱这个 anchor 的负样本权重。

(2) BCE 损失：- [y*log(p) + (1-y)*log(1-p)]

这里负样本的标签全是 0，所以公式退化为：

$$
BCE(𝑝,0)=−log(1−𝑝)
$$
```
BCE = [-log(1-0.045), -log(1-0.48), -log(1-0.2), -log(1-0.1)]
    ≈ [0.046,         0.653,        0.223,       0.105]
```
(3) 加 focal loss 权重：

$$
FL(𝑝)=(𝑝𝑟𝑜𝑏^𝛾)∗BCE
$$
```
FL = [0.045^2*0.046, 0.48^2*0.653, 0.2^2*0.223, 0.1^2*0.105]
   ≈ [0.00009,       0.151,        0.009,       0.001]
```

最后乘上 (1-alpha) = 0.75

loss = [0.00007, 0.113, 0.007, 0.001]
