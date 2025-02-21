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

计算heissian 矩阵

![Screenshot from 2025-02-21 11-10-24](https://github.com/user-attachments/assets/fdacdf1f-232d-4e1d-80da-7fdcfcbe8889)

