# 逻辑回归（Logistic Regression）

是一种广泛用于二分类问题的统计方法。

与线性回归不同，逻辑回归的输出是一个概率值，表示样本属于某个类别的概率。

其核心思想是通过一个 Sigmoid 函数 将线性回归的结果映射到 [0, 1] 之间，以此实现分类任务。

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


训练效果差

Oerfitting
1.增加训练集

2.data augmentation

3.（根据先验）限制训练模型，原模型可能太复杂了

![Screenshot from 2025-02-21 10-35-38](https://github.com/user-attachments/assets/c436978c-4335-443f-b96e-7355f9a7ad0e)

![Screenshot from 2025-02-21 10-38-03](https://github.com/user-attachments/assets/45e3c7fd-d1d0-44a8-a67e-d7db320154bb)

![Screenshot from 2025-02-21 10-56-25](https://github.com/user-attachments/assets/5c808f7e-7344-4766-bbe9-12041616cdcb)

4.mismatch

训练集和验证集分布不统一：训练用去年数据、推理用当下数据

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
