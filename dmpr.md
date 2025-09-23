# DMPR

1.输入是 环视鸟瞰图 (surround-view / BEV 图像)，大小统一调整为 512×512。

2.整张 BEV 图像划分为 16×16=256 个小块 (cell)。每个 cell 只负责预测落在该区域的一个标记点（L 或 T 角点）。

3.回归预测 (Directional Marking-Point Regression)

接一个 1×1 卷积，把通道数从 1024 压缩到 6，输出 tensor 尺寸：[batch, 6, 16, 16]，对应 6 维向量：(cx,cy,s,cosθ,sinθ,C)

- cx, cy：角点在 cell 内的相对位置（偏移量）。

- s：角点形状，二分类（T 型或 L 型）。

- cosθ, sinθ：角点的方向角，不直接回归 θ，而是回归正余弦值，避免角度不连续的问题。

- C：confidence，预测该 cell 是否有角点。

- 
