# RT-detr

<img width="1828" height="445" alt="Screenshot from 2025-11-18 14-59-46" src="https://github.com/user-attachments/assets/dd28d1e2-f54d-4912-9730-f4b33047f367" />

A： 直接concat

B： S3 S4 S5 分别做self-attention后concat

C： S3 S4 S5 先concat 融合后做self-attention

D： S3 S4 S5 分别做self-attention 后 PANet-style特征金字塔融合 

E: S3 S4 S5 分别做一层Layer的Self-attention，然后做


### PAN融合
```
def Variant_D_CSF(S3_enhanced, S4_enhanced, S5_enhanced):
    """
    输入：经过SSE尺度内交互增强后的特征{S3, S4, S5}
    CSF采用类似PANet的自上而下+自下而上融合路径
    """
    
    # 1. 自上而下路径（Top-down）
    P5 = Conv1x1(S5_enhanced)  # 高层特征调整
    P4 = Upsample(P5) + Conv1x1(S4_enhanced)  # 上采样融合
    P3 = Upsample(P4) + Conv1x1(S3_enhanced)  # 上采样融合
    
    # 2. 自下而上路径（Bottom-up）  
    N3 = Conv3x3(P3)
    N4 = Conv3x3(Downsample(N3) + P4)  # 下采样融合
    N5 = Conv3x3(Downsample(N4) + P5)  # 下采样融合
    
    return [N3, N4, N5]  # 融合后的多尺度特征
```

### AIFI: 单层Self-Attention
```
S5_flattened = Flatten(S5)  # 序列化高层特征
    S5_attended = SingleLayerSelfAttention(S5_flattened)  # 单层Transformer
    F5 = Reshape(S5_attended)  # 恢复特征图形状
```

### CCFF:

提出观点：

**直接拼接后做交互是冗余的**

直接拼接多尺度特征

```
P3 = backbone_stage3()  # [B, C, H3, W3] - 低级特征
P4 = backbone_stage4()  # [B, C, H4, W4] - 中高级特征  
P5 = backbone_stage5()  # [B, C, H5, W5] - 高级特征（已包含P3、P4的信息）
```
拼接


```
concat_features = concat([P3, P4, P5])  # [B, C, H3*W3 + H4*W4 + H5*W5]
```


在拼接特征上做自注意力
```
interacted = self_attention(concat_features)# 问题：P5已经包含了P3、P4的语义信息，再次让它们交互是冗余的！
```

问题分析：


P5 已从 P3、P4 中提取了语义信息。拼接后做自注意力，会让 P5 与 P3、P4 再次交互，这相当于对已处理的信息重复处理，造成冗余。



## 正确方式（RT-DETR 的做法）：

先尺度内交互，再跨尺度融合
```
P3 = backbone_stage3()  # 低级特征
P4 = backbone_stage4()  # 中高级特征
P5 = backbone_stage5()  # 高级特征
```

尺度内交互（AIFI）先增强自己的特征表示
```
P3_enhanced = self_attention(P3)  # 只在P3内部交互
P4_enhanced = self_attention(P4)  # 只在P4内部交互
P5_enhanced = self_attention(P5)  # 只在P5内部交互
```

跨尺度融合（CCFM - FPN/PAN结构）再融合，因为每个尺度已经充分增强过了
```
fused = FPN_fusion([P3_enhanced, P4_enhanced, P5_enhanced])
```
