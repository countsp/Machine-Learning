# VIT

![Screenshot from 2025-05-20 15-29-59](https://github.com/user-attachments/assets/c1cb2896-509a-46c5-bdfe-2f62427d7ac9)

在视觉任务中，注意力机制通常是与卷积网络结合使用，或者只替代卷积网络的某些组件，同时保留整体结构。我们证明了这种对 CNN 的依赖并非必要，**纯粹的 Transformer 直接应用于图像 patch 序列，同样可以在图像分类任务中取得出色表现**。



### 模型结构

## 1. 输入图像切分成 Patch

- 输入图像大小：`H × W × C`（例如 `224 × 224 × 3`）
- Patch 大小：`P × P`（例如 `16 × 16`）

因此，总共会得到：

$$
\frac{H}{P} \times \frac{W}{P} = \frac{224}{16} \times \frac{224}{16} = 14 \times 14 = 196 \text{ 个 patch}
$$

## 2. Patch 展平 + 线性投影（Patch Embedding）

每个 patch 是 `16 × 16 × 3 = 768` 维的向量，将其通过一个线性层投影到 D 维空间：

$$
P^2 \cdot C = 16^2 \times 3 = 768
$$

假设 D = 768，那么每个 patch 变为一个 `768` 维向量。

最终得到：
$$
X = [x_1, x_2, \dots, x_{196}] \in \mathbb{R}^{196 \times D}
$$

---

## 3. 加入 `[CLS]` Token

在序列前添加一个可训练的分类 token：

$$
x_{\text{cls}} \in \mathbb{R}^D
$$

得到新的序列：

$$
Z_0 = [x_{\text{cls}}, x_1, x_2, \dots, x_{196}] \in \mathbb{R}^{197 \times D}
$$

---

## 4. 加入位置编码（Position Embedding）

Transformer 无顺序感知，需要加上可学习的位置编码，和Z0同维度：

$$
E_{\text{pos}} \in \mathbb{R}^{197 \times D}
$$

于是输入为：

$$
Z_0 = Z_0 + E_{\text{pos}}
$$

---

## 5. 输入 Transformer Encoder

Transformer 编码器包含 \(L\) 层，每层执行以下操作：

1. 多头自注意力（Multi-Head Self-Attention, MSA）

2. 残差连接 + LayerNorm（PreNorm）

3. 前馈神经网络（MLP，含两层全连接 + GELU）

4. 再次残差连接 + LayerNorm

   

- 每层包含: 多头注意力 + 残差 + LayerNorm + MLP + 残差 + LayerNorm
- 总共迭代 \(L\) 次（如 12、24、32）

---

## 6. 提取最终 `[CLS]` 表示

经过编码器后输出：

$$
Z_L \in \mathbb{R}^{197 \times D}
$$

取第一个 token 的输出作为图像整体表示：

$$
y = z_{\text{cls}}^{(L)} \in \mathbb{R}^D
$$

---

## 7. 分类 Head（MLP）

将 \(y\) 输入分类头，输出 logits 维度为类别数 \(K\)：

$$
\hat{y} = \mathrm{MLP}(y) \in \mathbb{R}^K
$$

最终通过 softmax 得到分类概率：

$$
\text{prob} = \mathrm{softmax}(\hat{y})
$$

# 代码

```
import torch
import torch.nn as nn

class ViTEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, N, D], e.g., [batch_size, num_patches+1, embedding_dim]
        
        # Multi-Head Self-Attention block
        x_res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)  # MHSA
        x = x + x_res  # residual connection

        # MLP block
        x_res2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + x_res2  # residual connection

        return x

```

### 使用了PRE-LN

### ✅ 1. 更稳定的训练

- Pre-LN 会让残差连接处的数值分布更一致
- 避免深层模型梯度爆炸或消失，尤其是 ViT 通常有 12 层、24 层甚至更深

### ✅ 2. 更适合小 batch 训练

- Post-LN 训练时梯度更新更不稳定，尤其在小 batch 下
- ViT 初期受限于显存，batch size 比 NLP 小，Pre-LN 更鲁棒

### ✅ 3. 被后续研究验证效果更优

- 包括 GPT-2、BERT Pre-LN、ViT、DeiT、Swin Transformer 等，都采用 Pre-LN
- 实验结果显示更易收敛，性能更稳定
