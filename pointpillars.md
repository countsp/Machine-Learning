### 点云输入：
```
batched_pts = [
    torch.tensor([[1.2, -0.3, 0.8, 0.9],
                  [3.5,  1.7, -0.2, 0.5],
                  ... ]),   # 第1帧点云
    torch.tensor([[0.2,  2.1, 0.0, 1.0],
                  [5.4, -1.2, 0.3, 0.7],
                  ... ])    # 第2帧点云
]
```
Voxelization 就是核心，把点云划分成体素（pillar）

1.划分网格 (voxel_size, point_cloud_range), 代码里 voxel_size=[0.16, 0.16, 4]

2.每个点 (x,y,z) 通过公式
```
xi = int((x - x_min) / voxel_size_x)
yi = int((y - y_min) / voxel_size_y)
zi = int((z - z_min) / voxel_size_z)
```

就能知道它属于哪个 pillar

3.每个 pillar 最多保存 max_num_points=32 个点，多的丢弃，不足的补零。

### 最后 PillarLayer 输出： 

**pillars:** (Batch * num_total_pillars（pillar数量）, max_num_points(每个 pillar 最多保存点数) , 4（[𝑥,𝑦,𝑧,𝑟]）

**coors_batch:** (num_total_pillars, 1+3): pillar 的位置 [batch_id, xi, yi, zi]。

```
[0, 0, 0, 0]   → 第0帧，第0个pillar在网格 (xi=0, yi=0, zi=0)
[0, 0, 1, 0]   → 第0帧，第1个pillar在网格 (xi=0, yi=1, zi=0)
...
[0, 9, 9, 0]   → 第0帧，第100个pillar在 (xi=9, yi=9)
[1, 0, 0, 0]   → 第1帧，第0个pillar
...
[19, 9, 9, 0]  → 第19帧，第100个pillar
```

**npoints_per_pillar:** (num_total_pillars,):每个 pillar 内实际点数。

---

# PillarEncoder

**1. 计算相对特征**

相对 pillar 内点云重心
```
 offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None] # (p1 + p2 + ... + pb, num_points, 3)
```

相对 pillar 网格中心
```
x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (p1 + p2 + ... + pb, num_points, 1)
y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (p1 + p2 + ... + pb, num_points, 1)
```

**2. 拼接成 9 维特征:原始 4 维 (x,y,z,r) + 重心偏移 3 维 + pillar 中心偏移 2 维 = 9 维。**
```
features = torch.cat(
    [pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1
)
```
**3.1×1 卷积（MLP提取点特征）**
```
features = F.relu(self.bn(self.conv(features)))
```

**4.max pooling 得到 pillar 特征向量**
每个 pillar 内点的特征取最大值,每个通道上所有点的特征值取最大值:类似 N个点-> 一个点
```
pooling_features = torch.max(features, dim=-1)[0]
```

**5. scatter 回 BEV 网格**
```
batched_canvas = []
for i in range(bs):  # 遍历 batch
    cur_coors_idx = coors_batch[:, 0] == i
    cur_coors = coors_batch[cur_coors_idx, :]
    cur_features = pooling_features[cur_coors_idx]

    canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), 
                         dtype=torch.float32, device=device)
    canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
    canvas = canvas.permute(2, 1, 0).contiguous()   # (C, Y, X)
    batched_canvas.append(canvas)

batched_canvas = torch.stack(batched_canvas, dim=0)  # (bs, C, Y, X)
```
