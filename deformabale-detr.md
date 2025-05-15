# Deformable DETR


![deformable-detr](https://github.com/fundamentalvision/Deformable-DETR/raw/main/figs/illustration.png)

SenseTime与中科大、香港中文大学的研究人员合作发表在ICLR 2021上的论文，提出了一种改进版的DETR——Deformable DETR。该方法保留了DETR“端到端无需手工设计”的优势，同时解决了其训练收敛慢和小目标检测性能差的问题。

![Screenshot from 2025-05-15 19-06-43](https://github.com/user-attachments/assets/e37860d6-1978-4083-9b55-6b386f8dc47c)

论文核心是引入了Deformable Attention（可变形注意力模块）：

* 该模块只关注少量的采样点（如每个查询4个点），而不是整个特征图；

* 采样点的偏移（offsets）和注意力权重（attention weights）都由查询特征生成；

* 支持多尺度特征输入，替代传统的FPN结构实现尺度融合；

* 注意力计算复杂度由 $O(H^2W^2)$ 降为线性 $O(HW)$。
