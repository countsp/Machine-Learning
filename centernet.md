CenterNet（Objects as Points, 2019）把 **目标检测** 转化为 **关键点检测**，核心思想是：

- **检测目标中心点（center point）**
- **回归尺寸（w,h）+ 偏移量（offset）**
- 最终得到目标边界框。

结构上分为 3 个部分：

1. **Backbone（特征提取网络）**
   - 常用 ResNet/DLA-34/Hourglass 等作为骨干网络。
   - 输入图像 → 提取多尺度特征。
2. **Center Detection Head（中心点检测头）**
   - 输出 **heatmap**：每个类别一个通道，表示目标中心点的概率分布。
   - 监督方式：在真实中心点位置处放置高斯核（Gaussian kernel）。
3. **Regression Heads（回归头）**
   - **Size 分支**：预测目标的宽度和高度 (w, h)。
   - **Offset 分支**：预测在下采样特征图上的偏移，修正量化误差。
