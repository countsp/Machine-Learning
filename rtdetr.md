# RT-detr

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
