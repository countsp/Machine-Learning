# 深度可分离卷积

 深度可分离卷积将常规卷积操作分解为两个阶段：

1. **深度卷积（Depthwise Convolution）**：对每个输入通道独立卷积，只负责空间特征提取；对输入的每个通道独立应用一个 Dk×Dk的空间卷积核，完成空间特征提取；

   ![depth](https://i-blog.csdnimg.cn/blog_migrate/30c175b0bef7a15e776e1870a0f32ccd.png)

2. **逐点卷积（Pointwise Convolution）**：负责通道间的线性组合与融合；对深度卷积输出的所有通道，使用1×1的卷积核进行线性组合，完成通道间特征融合。

   ![pointwise](https://i-blog.csdnimg.cn/blog_migrate/e9d03c5bc05043ea86c4142076e50f0e.png)

### Primary Goal（主要目标）：

### 降低模型的参数和计算复杂度，保留和近似常规卷积的表达能力；

对比普通conv

![regular conv](https://i-blog.csdnimg.cn/blog_migrate/b418b74103a4aa4e63b741c4b9f016b3.png)



![Screenshot from 2025-06-20 15-44-21](/home/office2004/Pictures/Screenshot from 2025-06-20 15-44-21.png)

**Entry Flow**

- 以一个常规的 3×3 卷积开始，用于对输入图像（通常 299×299×3)做初步特征提取；
- 接着是一系列带有残差连接的深度可分离卷积 (Depthwise Separable Convolution) 模块，每个模块内部先做深度卷积再做逐点卷积，并通过 stride=2 的方式逐步降低空间分辨率；
- 入口部分的设计目标是快速减小特征图的空间大小，同时逐步增加通道数，为后续更深的模块做好准备。

**Middle Flow**

- 包含 **8 次完全相同** 的深度可分离卷积模块堆叠（均带有线性残差分支），每个模块内部不改变空间分辨率（stride=1）；
- 这一部分负责在保持特征图大小的同时，深度挖掘更丰富的空间和通道特征。

**Exit Flow**

- 先是一两个深度可分离卷积模块，用于进一步整合和升维特征；
- 随后接全局平均池化将空间信息汇聚成通道向量；
- 最后接一个全连接/Logistic 回归层（在 ImageNet 上通常是 1000-way 分类），完成最终的类别预测。
