# 逻辑回归（Logistic Regression）

是一种广泛用于二分类问题的统计方法。

与线性回归不同，逻辑回归的输出是一个概率值，表示样本属于某个类别的概率。

其核心思想是通过一个 Sigmoid 函数 将线性回归的结果映射到 [0, 1] 之间，以此实现分类任务。

**线性部分**：
  
$z = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n = \mathbf{w}^T \mathbf{x}$

其中x1,x2等为原始数据经过预处理后的特征值。

**Sigmoid 映射：**

$\sigma(z) = \frac{1}{1 + e^{-z}}$

**分类决策：**

若 $\sigma(z) > 0.5$，预测为 正类 (1)

若 $\sigma(z) \leq 0.5$，预测为 负类 (0)
# 梯度下降

梯度下降法（Gradient Descent）是一种常用于优化机器学习模型参数的算法，主要用于最小化损失函数。其核心思想是通过迭代更新参数，使损失函数逐渐减小，直到达到局部最小值（对于凸函数则是全局最小值）。
原理。

![Screenshot from 2024-09-23 09-27-42](https://github.com/user-attachments/assets/5c1695c2-69e0-47e1-86c7-c749c27fb5b4)

梯度下降的基本更新公式为：

$' θ=θ−α⋅∇J(θ) '$

其中：

θ 表示参数向量（例如权重）。
α 表示学习率，控制每次迭代参数更新的步长。
∇J(θ) 表示损失函数 J(θ) 对参数 θ 的梯度


## 训练效果差

**Overfitting**

**1.增加训练集**

**2.data augmentation**

**3.（根据先验）限制训练模型，原模型可能太复杂了**

![Screenshot from 2025-02-21 10-35-38](https://github.com/user-attachments/assets/c436978c-4335-443f-b96e-7355f9a7ad0e)

![Screenshot from 2025-02-21 10-38-03](https://github.com/user-attachments/assets/45e3c7fd-d1d0-44a8-a67e-d7db320154bb)

![Screenshot from 2025-02-21 10-56-25](https://github.com/user-attachments/assets/5c808f7e-7344-4766-bbe9-12041616cdcb)

**4.mismatch**

训练集和验证集分布不统一：训练用去年数据、推理用当下数据

**5.调参**

✅ 一、正则化方法（控制模型复杂度） 

L2 正则化（权重衰减）	正则化的目标是在学习表达能力和控制复杂度之间找到一个平衡点，惩罚模型参数过大，鼓励权重靠近 0，使模型更简单	--weight_decay=1e-4 是常用值，适度地惩罚太大的权重，而不是直接把它们归零。

Dropout	随机屏蔽部分神经元，防止网络依赖某些特征	常用值 0.1 ~ 0.5，在 Transformer 中一般设置较小

Early stopping	验证集精度不提升就提前停止训练，防止过拟合训练数据	手动或用回调实现

✅ 二、调整模型与训练策略


减小模型容量	减少模型参数数量，避免“记住”训练数据	更浅的 ResNet、更小的 Transformer

学习率衰减（lr schedule）	后期降低学习率，微调模型以更好泛化	--lr_drop 是典型策略

梯度裁剪（clip_max_norm）	控制参数变化范围，避免模型震荡	特别适合 Transformer

✅ 三、数据增强

图像增强	生成更多“变化版本”的图像，增加模型鲁棒性	翻转、裁剪、旋转、颜色扰动等

Mixup / CutMix	两张图混合训练，打乱标签与像素对应关系	增强模型泛化能力

随机遮挡（random erasing）	模拟 occlusion，让模型学习更强特征	特别适合目标检测任务

✅ 四、合理使用验证集与训练监控

设置验证集	用于判断模型是否真正泛化，不要用训练集做评估

画 loss 曲线（train vs val）	如果 train loss 降低但 val loss 上升，说明过拟合了

每轮评估，保存 best val acc 模型	避免保存过拟合的参数

✅ 五、使用更多数据

数据越多 → 越难过拟合 → 模型训练更稳。

使用数据增强；

合成数据；

迁移学习（从大数据集预训练再 fine-tune）；

使用伪标签（semi-supervised learning）。



# 避免critical point (局部最优解、鞍点)

判断是local minimal 或者 saddle point : 计算heissian 矩阵

![Screenshot from 2025-02-21 11-10-24](https://github.com/user-attachments/assets/fdacdf1f-232d-4e1d-80da-7fdcfcbe8889)

输出 矩阵后 计算eigen value ，取出eigen Value 的 一个eigenvector

![Screenshot from 2025-02-21 11-13-48](https://github.com/user-attachments/assets/60a47787-4559-42de-a12c-20e981bba60e)

# 使用momentum 避免进入critical point (类似势能)

![Screenshot from 2025-02-21 17-22-24](https://github.com/user-attachments/assets/e3f2b37c-c45f-4c1f-8223-de64817aca5c)


![Screenshot from 2025-02-21 17-25-42](https://github.com/user-attachments/assets/c6f5108a-ef00-424d-b7f4-fa06109b86ef)

# 大的batch和小的batch对比

![Screenshot from 2025-02-21 17-15-36](https://github.com/user-attachments/assets/380fd754-9b87-4075-bb98-7284db3f7ae3)

# 梯度下降
梯度下降的每个参数学习率应该不同，否则会有这种情况：

![Screenshot from 2025-02-24 08-50-28](https://github.com/user-attachments/assets/dcd0c52f-272f-4862-a882-5aa09aa8c3db)

针对每个参数定制学习率

![Screenshot from 2025-02-24 08-50-43](https://github.com/user-attachments/assets/80d87407-40f5-477b-9f55-37f5597d3e8b)

可以用RMS修改学习率

![Screenshot from 2025-02-24 08-52-18](https://github.com/user-attachments/assets/d7c48f78-047d-4f44-b527-e02df6252208)

RMSProp方法

![Screenshot from 2025-02-24 08-57-32](https://github.com/user-attachments/assets/630cdad3-2a55-4735-b4e5-af8a435734e2)

Adam就是用了RMSProp + Momentum
![Screenshot from 2025-02-24 08-57-03](https://github.com/user-attachments/assets/a5bcc695-44fc-48cd-876d-708a0d215a29)

# 学习率

学习率的改变影响训练效果

![Screenshot from 2025-02-24 09-22-21](https://github.com/user-attachments/assets/98bccde7-1225-4f5d-b5f2-6926ae78a45c)

Resnet和Transformer都用了warm up 学习率策略

# 优化总结

![Screenshot from 2025-02-24 09-26-27](https://github.com/user-attachments/assets/5c0bf677-526e-4205-8f61-2ab88719707a)

# softmax

1.归一化

2.区分度提高（大的更大，小的更小）

3.和sigmoid是一样的

# loss

![Screenshot from 2025-02-24 09-53-56](https://github.com/user-attachments/assets/3c993bcf-32cc-4e5a-ac70-253beb87a24a)

# overfitting
待选择模型太多了，参数量太大了

# why deeper not fatter

![Screenshot from 2025-02-28 08-54-01](https://github.com/user-attachments/assets/bd6df288-bd22-4212-a408-4c8666fbabb7)

层数多，效率高
![Screenshot from 2025-02-28 08-56-07](https://github.com/user-attachments/assets/938f958f-3c66-4b14-b732-1d22bec1798d)


# GAN
同一个输入，输出不同，需要用到GAN 网络。

![Screenshot from 2025-03-03 08-51-19](https://github.com/user-attachments/assets/431ec87c-41c8-45ce-bcd0-ef16b2bbd8c3)

需要训练一个discriminator, 对抗generator。

![Screenshot from 2025-03-03 08-53-10](https://github.com/user-attachments/assets/7f87da9b-7281-4bd5-90f5-f402121c2937)

![Screenshot from 2025-03-03 09-05-36](https://github.com/user-attachments/assets/231ea0ff-1194-429a-bc5e-ba3a24b97fa0)


# Self-Attention

![Screenshot from 2025-03-03 11-00-54](https://github.com/user-attachments/assets/7255b85e-cde4-4f2a-820a-ff8a886d820a)

![Screenshot from 2025-03-03 11-03-25](https://github.com/user-attachments/assets/3171fb98-e43c-4d06-9bc4-e2a51e1c0e5c)


![Screenshot from 2025-03-07 13-26-20](https://github.com/user-attachments/assets/6915c94c-9c1b-40bf-b73b-bb271d784917)

![Screenshot from 2025-03-07 13-28-05](https://github.com/user-attachments/assets/2810bdbf-9b1d-4985-b9e8-b0d188d29d39)

# Multihead-attention

![Screenshot from 2025-03-07 13-31-01](https://github.com/user-attachments/assets/6740b5c2-f7be-481e-b7d4-e34088e7e037)

# Self-Attention vs CNN

![Screenshot from 2025-03-07 14-02-46](https://github.com/user-attachments/assets/0969299b-d75d-4540-936e-aa1495316a31)

cnn是简化的self-attention，它只关注周边 Pixel 作为 receptive field；

Self-Attention 相当于 cnn 进阶，它的receptive field 不知周边 pixel，而是整张图。

# Self-Attention vs RNN

RNN 考虑不到 后续input vector的内容。使用双向RNN可以，但是最左侧输出要考虑最右侧（最远端）内容，需要一路承载信息。

![Screenshot from 2025-03-07 14-09-58](https://github.com/user-attachments/assets/0f1b96f8-fbb8-4271-82a1-6955e2e490c8)

而且RNN不是并行的，需要一路推理。

# Batch Normalization

愚公移山式优化误差平面

![Screenshot from 2025-03-07 14-26-51](https://github.com/user-attachments/assets/51950534-47fb-4bda-99ca-254bbace7236)

对整个input的向量做normalization，参数变化会影响 u , sigma，不能将所有数据放入GPU中训练，所以对batch做normalization

![Screenshot from 2025-03-07 14-38-26](https://github.com/user-attachments/assets/a97b308a-2317-47f7-9944-afd9deb1f60f)

推理时，要实时对每一个input而不是batch做推理，这个u ,sigma从哪里来？ 在train时候 ， 估算了 u 和 sigma 的平均。

![Screenshot from 2025-03-07 14-48-46](https://github.com/user-attachments/assets/57d31841-ee88-4e98-9568-f825dc78968e)

$$
running_mean_t​=(1−m)⋅running_mean_(t−1​)+m⋅μbatch​
$$

# Transformer

![Screenshot from 2025-03-10 10-27-57](https://github.com/user-attachments/assets/ce0cc1b2-ced3-4eb5-bacd-6d5749bad9c4)

一个翻译器的训练阶段：数据集x为“machine learning”,y为”机器学习“。

encoder通过self-attention和ffd直接输出一个vector seq为m1,然后m1和‘/begin’进入decoder'，推测第一个token。

然后第一个token与“机”作交叉熵验证然后优化decoder

然后m1和第一个token推测第二个token

第二个token与“器”作交叉熵验证然后优化decode直到推到"/end"。
