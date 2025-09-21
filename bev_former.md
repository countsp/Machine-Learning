# BEVFormer

<img width="2410" height="942" alt="Screenshot from 2025-09-21 19-01-24" src="https://github.com/user-attachments/assets/a222d9de-c507-4261-9551-9be71f861c34" />


**以往的相机感知方法（如FCOS3D、DETR3D等）主要依赖2D图像特征，难以有效整合多视角信息，尤其在长距离、遮挡和运动状态估计方面表现不足。相比之下，BEV表示具有视角统一、结构清晰的优势，便于下游任务处理。然而，从2D图像生成高质量BEV特征是一个病态问题，传统方法常依赖深度估计，易受误差累积影响。**

**BEVFormer 的目标是避免对深度依赖，直接利用时空注意力机制从图像中学习 BEV 表征。**



**BEV Queries（BEV查询）**：

- 将BEV空间离散成固定网格，每个格子用一个**可学习查询向量**表示，形成一组BEV Queries。
- 每个查询通过注意力机制从图像和历史BEV中提取对应信息。

**Spatial Cross-Attention（空间交叉注意力）**：

- 每个BEV查询通过**Deformable Attention**机制，仅从图像中**视野范围内的兴趣区域**提取特征，避免了全局注意力的高计算开销。
- 利用相机内外参将BEV坐标映射到各个视角图像中。

**Temporal Self-Attention（时间自注意力）**：

- 模拟RNN中的隐藏状态传递，利用**历史BEV特征**与当前BEV Queries交互，捕捉目标运动信息，提升速度估计与遮挡物检测能力。
- 计算效率高，仅使用上一个时间帧的BEV，无需堆叠多个时间帧。

**任务头部设计**：

- 采用Deformable DETR结构进行3D目标检测，直接预测目标的三维位置、尺寸、朝向和速度。
- 分割任务使用Panoptic SegFormer结构，支持语义级别的地图分割，如道路、车道、车辆等。


---

### ✅ 输入：

- **多摄像头图像**（6个视角，每帧）：`Iₜ = {Iₜ¹, Iₜ², ..., Iₜ⁶}`

- **前一时刻的 BEV 特征**：`Bₜ₋₁`（t−1 时刻）

- **相机参数**：每个摄像头的内参和外参

  

### 🧱 Step 1：图像编码（Backbone 提取特征）

使用共享主干网络（如 ResNet101 + FPN）对每个摄像头图像提取多尺度特征：

```
Fₜ = {Fₜ¹, Fₜ², ..., Fₜ⁶}
```



### 🧩 Step 2：初始化 BEV Queries

- 创建一个形状为 `H×W` 的 **BEV Query 网格**（如 200×200），每个位置一个 Learnable Embedding，代表 BEV 空间中的一个网格单元。

- 添加位置编码（Position Embedding）

  ###  🔁 Step 3：BEVFormer 编码器（共 6 层，每层执行以下操作）

  #### a. **Temporal Self-Attention（时间自注意力）**

  - 目标：融合 `Bₜ₋₁` 的时间信息
  - 将 `Bₜ₋₁` 与当前 BEV Queries `Q` 对齐（使用 ego-motion），作为注意力键值对
  - 每个 BEV Query 从 `Bₜ₋₁` 中提取运动与历史线索

  #### b. **Spatial Cross-Attention（空间交叉注意力）**

  - 每个 BEV Query：
    - 在 BEV 平面(可以映射到xy)上升成一个柱体（多个高度点）
    - 利用相机投影矩阵将柱体投影到图像平面，得到多个 2D 参考点（reference points）
    - 使用 **Deformable Attention** 从命中的图像特征区域中采样
    - 汇总多个摄像头的信息

  #### c. **Feed Forward + LayerNorm**

  - 对每个 BEV Query 特征进行前馈网络和规范化处理
  - 输出新的 BEV 特征 `Bₜ`，作为下一层输入或最终结果
