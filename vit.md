# VIT

![Screenshot from 2025-05-20 15-29-59](https://github.com/user-attachments/assets/c1cb2896-509a-46c5-bdfe-2f62427d7ac9)

在视觉任务中，注意力机制通常是与卷积网络结合使用，或者只替代卷积网络的某些组件，同时保留整体结构。我们证明了这种对 CNN 的依赖并非必要，**纯粹的 Transformer 直接应用于图像 patch 序列，同样可以在图像分类任务中取得出色表现**。



### 模型结构

1. **图像分割成 Patch**：

   * 输入图像被划分为固定大小（如 16x16 像素）的 patch。

   * 每个 patch 被展平并线性映射为向量 $x_p$ ，形成 patch embedding。

2. **类比 NLP 的词向量处理**：

   * $x_p$加上一个 learnable 的 [class] token，用于图像分类。
  
   * 乘以可学习的权重W

   * 加入位置编码（positional embedding）以保持空间信息。

     $Image Feature = x_p · W  +  E_{pos}$

3. **Transformer 编码器**：

   * 输入 patch 序列进入标准的 Transformer 编码器（多头自注意力 + MLP 块，带有残差连接和 LayerNorm）。
   * 输出中 [class] token 的表示被用于最终分类。
