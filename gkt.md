# GKT

![Screenshot from 2025-05-20 14-42-30](/home/office2004/Pictures/Screenshot from 2025-05-20 14-42-30.png)

**该研究聚焦于自动驾驶中的一个关键任务：如何从多视角的二维摄像头图像中高效且鲁棒地学习鸟瞰图（BEV，Bird's Eye View）表示。为此，作者提出了一种全新的2D到BEV表示学习机制——几何引导的核变换器（Geometry-guided Kernel Transformer, GKT）。**



## 主要创新点：

1. **几何引导机制**：GKT利用粗略的几何先验信息，引导Transformer关注图像中更具判别性的区域，从而提升BEV表示的准确性。
2. **核展开策略**：在预估的二维位置周围展开特征区域，并通过交互生成BEV表示。
3. **高效的查表索引（LUT）方法**：为了提高推理速度，GKT在推理时不再依赖摄像头的标定参数，而是通过预先构建的查找表快速获取图像索引。
4. **强鲁棒性**：即使摄像头在实际运行中发生偏移，GKT仍能保持对目标区域的关注，不受BEV高度预设值的影响。
5. **高效率与性能**：在3090 GPU上达到了72.3 FPS（2080Ti上为45.6 FPS），在nuScenes数据集（100m×100m，0.5m分辨率）上实现了38.0 mIoU的实时分割精度，达到目前同类方法中的最高水平。



---

## 工作原理概述

### **🧱 输入：多视角图像**

输入是多个摄像头的环视图像

### 提取多尺度图像特征

将所有视角图像输入到共享 CNN backbone（如 ResNet）+ FPN 提取金字塔特征：

$image_features∈[nlevel,nview,C,H,W]$

### 构建 BEV 网格 learnable query[w,h,C]

BEV 空间被均匀划分为一个网格（如 200×200），每个格子：

- 代表一个地面 patch（如 0.5m×0.5m）
- 对应一个 3D 坐标 Pi=(xi,yi,z)P_i = (x_i, y_i, z)Pi=(xi,yi,z)
- 搭配一个 learnable query 向量

### 几何引导 BEV 点 → 投影到图像坐标(3d->2d)
使用相机内外参，将每个 BEV 点投影到多个视角的图像上

这个位置用于从图像特征中提取一个 **局部 patch**



**BEV Query中一个对应的视角数不同，怎么办？**

GKT 中的做法是：所有 BEV 点统一展开相同数量的 patch，padding 不可见视角，让 Key/Value 的长度固定，这样可以和 BEV query 做标准 attention。



### 图像特征展开 Patch

* 在每个点的周围展开一个固定大小的窗口（如 5×5 patch）

* 多个视角、多尺度下的 patch 被收集到一起

* 然后 flatten 

### **BEV Query 与 Patch 通过 Transformer 交互**

使用标准 Multi-Head Attention：

- Query：BEV query 
- Key / Value：展开后的 patch 特征

执行 attention 计算，更新 BEV feature：

### 将所有更新后的 reshape 成 BEV 网格结构

[ BEV 表征 ] → 用于检测 / 分割 / 预测等任务





### LUT

GKT 中的 LUT（Lookup Table）是在推理阶段用于**快速查找每个 BEV 网格点在各个摄像头图像中的投影位置**，从而用于展开 patch 区域的采样坐标。是工程性的创新点而不是算法向。



传统方法如 Lift-Splat、BEVDet 每次都用相机内参 + 外参计算：

$ Q=K⋅(R⋅P+t)Q = K \cdot (R \cdot P + t)Q=K⋅(R⋅P+t)$

对 BEV 网格中每一个点、每个相机视角、每一帧都重新计算太慢！

