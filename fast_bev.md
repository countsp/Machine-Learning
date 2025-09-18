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
