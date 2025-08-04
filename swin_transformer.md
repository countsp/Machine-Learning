# Swin_transformer

**本文提出了一种采用层次化结构的Transformer，并使用移动窗口（Shifted Windows）来计算特征表示。移动窗口机制仅在不重叠的局部窗口内进行自注意力计算，提升了计算效率，同时又保留了跨窗口的信息交互能力。该架构不仅能够灵活建模不同尺度的信息，而且其计算复杂度相对于图像大小是线性的。**

![Screenshot from 2025-07-17 11-02-05](/home/office2004/Pictures/Screenshot from 2025-07-17 11-02-05.png)

### Patch Partition 

- 将输入图像切成固定大小 patch（4×4）
- 每个 patch 展平后输入线性层（维度 3->48）
- 得到 `[H/4, W/4, C]` 的特征图，相当于初步下采样



### Linear Embedding

nn.Linear(48, C)

### Patch Merging

类似 CNN 的 pooling：

- 每 2×2 相邻 patch 合并
- 通道数 ×2，空间尺寸 /2
- 实现多尺度建模能力

```
x = x.view(B, H, W, C)

x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

x = self.norm(x)  # 对每个 4C 维 token 做 LayerNorm
x = self.reduction(x) # Linear: 4C → 2C
```

![patch merging](https://i-blog.csdnimg.cn/direct/58279113326c4ce1a1c7e0d2383bc0d6.png#pic_center)

---

### W-MSA

**W-MSA（Window-based Multi-head Self-Attention）** 是 Swin Transformer 中的注意力机制，它在**固定大小的局部窗口内**计算 self-attention，不是像 ViT 那样在整张图上全局计算。



在整张图上计算 self-attention，如左下图，计算量是：

O(N2)(N 是 token 数)\mathcal{O}(N^2) \quad \text{(N 是 token 数)}O(N2)(N 是 token 数)

对高分辨率图像（比如 224×224）来说，token 太多，代价极大！



把图像划分为多个**固定大小的小窗口**（例如 7×7），每个窗口内单独做 self-attention，**不会跨窗口通信**，如右下图

- 计算量变成：

  O(M⋅W2)(M 是窗口数，W 是窗口内 token 数) 图中 token=4，M=4

- 成本从平方下降到线性！

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/545285fb54da4dc6b7fc776edc5a1aed.png#pic_center)

---

### SW-MSA

首先将第一行移动至最后一行，第一列移动到最后一列。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8ceaa87900644daa9802f041236dd780.png#pic_center)
