# LSS

## 一、Lift（提升图像特征到3D）

### 输入：

- 多个相机图像：{Xk ∈ ℝ³×H×W}
- 每个图像的外参矩阵 Ek 和内参矩阵 Ik

### 步骤：

1. **为每个图像独立处理**：
   - 使用 CNN（如 EfficientNet）提取每个像素的上下文特征向量 c ∈ ℝᶜ；
   - 同时预测该像素的深度概率分布 α ∈ ℝᴰ（D 为离散深度层数）。
2. **构建“体素特征体”**：
   - 每个像素被“提升”为一条射线，对应 D 个深度层；
   - 每一层上的特征向量为 `c_d = α_d * c` (深度概率分布 * 特征向量)；
   - 最终形成一个**frustum-shaped point cloud**（棱台形点云），表示该像素在不同深度上的潜在表示。

------

## 二、Splat（投影到鸟瞰图）

### 步骤：

1. **从图像空间投影到 BEV 坐标系**：
   - 利用相机的外参和内参将棱台形点云坐标映射到 BEV 平面上。
2. **将点云投影为 BEV 特征图**：
   - 使用 Pillar Pooling（柱状池化）将点云特征进行聚合，形成固定尺寸的 BEV 表示张量 `y ∈ ℝᶜ×X×Y`；
   - 每个 BEV cell 聚合落入其中的点云特征之和。
3. **使用 BEV CNN 进一步处理**：
   - 使用 ResNet 结构处理 BEV 特征图，输出语义分割图（如车道、可通行区域、车辆等）。

------

## 三、Shoot（运动规划）

### 目标：

基于 BEV 语义图，预测最优的行驶轨迹，执行端到端运动规划。

### 步骤：

1. **准备轨迹模板**：

   - 通过 K-Means 对大量专家驾驶轨迹聚类，得出 K 个模板轨迹 `{τ₁, τ₂, ..., τ_K}`。

2. **评分机制**：

   - 计算每个轨迹 τᵢ 在 BEV 成本图中的代价：

     $' p(τi∣o)=exp⁡(−∑(x,y)∈τico(x,y))∑τ∈Texp⁡(−∑(x,y)∈τco(x,y))p(τ_i | o) = \frac{\exp(-\sum_{(x,y) ∈ τ_i} c_o(x,y))}{\sum_{\tau ∈ T} \exp(-\sum_{(x,y) ∈ τ} c_o(x,y))}p(τi∣o)=∑τ∈Texp(−∑(x,y)∈τco(x,y))exp(−∑(x,y)∈τico(x,y)) '$

   - 将规划任务转化为一个 K 类分类问题（选择最佳轨迹模板）。

3. **训练方式**：

   - 使用交叉熵损失，最小化专家轨迹与模板轨迹之间的距离；
   - 可实现端到端训练，使感知模块与规划模块联合优化。

------

## 四、模型特性与优势

- **端到端可微**：从图像到 BEV 表示、再到轨迹预测均可反向传播；
- **无需激光雷达**：完全基于 RGB 图像学习语义和几何；
- **鲁棒性强**：支持摄像头缺失和外参扰动的鲁棒训练；
- **可零样本泛化到新摄像头布局**（zero-shot camera rig transfer）。

# ParkingE2E 对应代码

**1.BEV 参数计算**

**功能：**

根据 cfg.bev_x_bound / bev_y_bound / bev_z_bound 生成 BEV 网格划分参数。

**代码：**
```
bev_res, bev_start_pos, bev_dim = calculate_birds_eye_view_parameters(...)
self.bev_res = nn.Parameter(bev_res, requires_grad=False)    # 网格分辨率 [0.1, 0.1, 20.0]
self.bev_start_pos = nn.Parameter(bev_start_pos, requires_grad=False) # 第一个格子中心点 [-9.95, -9.95, 0]
self.bev_dim = nn.Parameter(bev_dim, requires_grad=False)    # BEV 格子数量 [200, 200, 1]
```
**2.Frustum 视锥体构建**

功能：构建像素坐标 (u, v) 与深度 d 组合的三维网格，表示像素反投影空间。告诉你每个 (d,h,w) 对应相机坐标的哪一个点(u,v,d)。代码中是 48 个 bin，每个 bin 间隔：0.25 m，总深度覆盖：0.5 m ~ 12.25 m

代码：
```
self.frustum = self.create_frustum()
```
输入图像大小 256×256，下采样 8 → 32×32，深度切片 48，所以：

frustum.shape = (48, 32, 32, 3)储存 32*32个像素 + 48个深度 bin 对应的三维点坐标（u,v,depth）


**3.相机特征编码 (CamEncoder)**

功能：用 EfficientNet 提取特征 + 可选深度分布预测。

代码：
```
self.cam_encoder = CamEncoder(self.cfg, self.depth_channel)
```

**4.几何投影到世界坐标**

功能：将 frustum 投影到 3D 世界坐标，结合相机内参、外参。

代码：
```
geom = self.get_geometry(intrinsics, extrinsics)
```

**5.BEV 特征聚合 (Lift-Splat)**

功能：把多视角图像特征根据几何位置投影到 BEV 平面，并对落入同一格子的特征做求和/聚合。

代码：
```
bev_feature = self.proj_bev_feature(geom, x)
```

**6.输出**

功能：输出 BEV 特征图 & 深度预测。

```
return bev_feature.squeeze(1), pred_depth
```

# camera_encoder

CamEncoder 用的是 stem + 前 4~5 个 stage 的 blocks，去掉 head。

```
CamEncoder(
  (backbone): nn.Sequential(               # 裁剪过的 EfficientNet
      (stem): nn.Sequential(
          (0): Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
          (1): BatchNorm2d(32)
          (2): Swish()
      )
      (blocks): nn.Sequential(
          MBConvBlock(...),
          MBConvBlock(...),
          ...
          MBConvBlock(...)   # 保留到 stage5
      )
  )
  
  (feature_branch): nn.Sequential(
      (0): DeepLabHead(...)
      (1): UpsamplingConcat(...)
  )
  
  (depth_branch) [if enabled]: nn.Sequential(
      (0): DeepLabHead(...)
      (1): UpsamplingConcat(...)
  )
)
```



**主干网络 (Backbone)**

使用 EfficientNet（B0 或 B4）。

通过 delete_unused_layers 裁剪掉无用的高层 block 和分类 head，只保留前 4~5 个 stage，用作特征提取。

对比efficientnet的原始结构，CamEncoder用了Stage1-5

```
Stem: Conv3x3 (stride=2) → (H/2, W/2)

Stage1:  MBConv1,  repeat=1, stride=1   → (H/2, W/2)
Stage2:  MBConv6,  repeat=2, stride=2   → (H/4, W/4)
Stage3:  MBConv6,  repeat=2, stride=2   → (H/8, W/8)
Stage4:  MBConv6,  repeat=3, stride=2   → (H/16, W/16)
Stage5:  MBConv6,  repeat=3, stride=1   → (H/16, W/16) 

Stage6:  MBConv6,  repeat=4, stride=2   → (H/32, W/32)
Stage7:  MBConv6,  repeat=1, stride=1   → (H/32, W/32)

Head: Conv1x1 + BN + Swish + Pool + FC

```


输出特征分辨率大约是输入的 1/8（因为 bev_down_sample=8）。

**Feature 分支**

输入：EfficientNet 提取的高层特征（input_1）和较浅层特征（input_2）。

DeepLabHead（空洞卷积） → 增大感受野。

UpsamplingConcat（上采样 + 拼接） → 融合深浅层特征。

最终输出通道数 = cfg.bev_encoder_in_channel = C（比如 64）。

功能：得到用于 BEV 投影的图像语义特征。

**Depth 分支（可选）**

只有当 use_depth_distribution=True 才启用。

代码中用DeepLabHead，在提取出特征后直接接入。

```
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, hidden_channel=256):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36], hidden_channel),   # 多尺度空洞卷积
            nn.Conv2d(hidden_channel, hidden_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(),
            nn.Conv2d(hidden_channel, num_classes, 1)         # 1x1 conv 映射到 num_classes
        )

```

# VoxelSumming
假设 BEV 网格大小 = [200,200]，每个 voxel 存一个 C=64 维的特征向量。

点A 特征：[1,2,3,...,64] 落到格子 (100,120)

点B 特征：[0.5,1.0,1.5,...,32] 也落到格子 (100,120)

那么 voxel(100,120) 的特征就是直接求和：

```
[1+0.5, 2+1.0, 3+1.5, ..., 64+32]
```
