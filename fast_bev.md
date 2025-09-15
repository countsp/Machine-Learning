# Encoder

**通道对齐 (1×1 Conv)**

- 每个尺度的特征先通过一个 **1×1 卷积**，把通道数变成相同的 CCC。
- 例如：
  - F1′=Conv1x1(F1)∈512×512×CF1' = Conv1x1(F1) \in 512 \times 512 \times CF1′=Conv1x1(F1)∈512×512×C
  - F2′=Conv1x1(F2)∈256×256×CF2' = Conv1x1(F2) \in 256 \times 256 \times CF2′=Conv1x1(F2)∈256×256×C
  - …

**自顶向下上采样 + 融合**

- 把小分辨率的特征上采样到大分辨率，再和同尺度的特征相加/拼接。
- 例如：
  - F2′′=F2′+UpSample(F3′)F2'' = F2' + UpSample(F3')F2′′=F2′+UpSample(F3′)，维度都是 256×256×C256 \times 256 \times C256×256×C
  - F1′′=F1′+UpSample(F2′′)F1'' = F1' + UpSample(F2'')F1′′=F1′+UpSample(F2′′)，维度都是 512×512×C512 \times 512 \times C512×512×C

**最终输出多尺度特征**

- 经过进一步卷积处理，得到统一后的 3 层：
  - F1/4F_{1/4}F1/4 = 512×512×C512 \times 512 \times C512×512×C
  - F1/8F_{1/8}F1/8 = 256×256×C256 \times 256 \times C256×256×C
  - F1/16F_{1/16}F1/16 = 128×128×C128 \times 128 \times C128×128×C
