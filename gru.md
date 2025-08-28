# GRU

## GRU 的核心结构

![../_images/gru-3.svg](https://zh.d2l.ai/_images/gru-3.svg)

GRU 主要包含两个“门”：

- **更新门（Update Gate, z_t）**：决定当前时刻隐藏状态有多少来自过去的记忆，有多少来自新的输入。
- **重置门（Reset Gate, r_t）**：决定在生成候选状态时，要“忘记”多少过去的记忆。

计算流程：

1. 输入当前时刻特征 xt，上一个隐藏状态 ht−1。

2. 计算更新门 zt和重置门 rt：
   $$
   z_t = \sigma(W_z x_t + U_z h_{t-1})
   $$

   $$
   r_t = \sigma(W_r x_t + U_r h_{t-1})
   $$

   

3. 得到候选隐藏状态  h~t：
   $$
   \tilde{h}_t = \tanh(W_h x_t + U_h(r_t \odot h_{t-1}))
   $$
   

4. 最终隐藏状态更新：
   $$
   h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
   $$
   

这里的 σ 是 Sigmoid 函数，⊙ 是逐元素相乘。

```
import torch, torch.nn as nn

gru = nn.GRU(input_size=4, hidden_size=16, num_layers=1, batch_first=True)

x = torch.randn(32, 10, 4)  # (batch=32, seq=10, feature=4)

# case1: 不传 h0，默认全零
out1, h1 = gru(x)

# case2: 传入自定义 h0
h0 = torch.randn(1, 32, 16)  # (num_layers=1, batch=32, hidden=16)
out2, h2 = gru(x, h0)
```

**PyTorch 的 GRU 会在内部自动保存和更新 ht−1，用户不需要手动维护。**

```
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```



