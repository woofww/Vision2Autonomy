# 第 04 章：从角点到匹配与 RANSAC

[English](README.md) | [简体中文](README.zh-CN.md)

> 本章回答一个关键问题：两张图里的两个角点，如何判断它们来自现实世界中的同一个位置？

## 学习目标

完成本章后，你应该能够：

- 区分检测器、描述子、匹配器和几何验证器；
- 手工推导零均值、单位范数的灰度补丁描述子；
- 解释最近邻、Lowe 比率检验和双向一致性各自删除什么错误；
- 用归一化 DLT 从至少四对点估计 Homography；
- 解释 RANSAC 为什么能在错误匹配中恢复正确模型；
- 说明特征匹配如何服务于视觉里程计、定位、拼接与三维重建。

## 前置知识

- 第 01 章的像素、卷积与 NumPy 数组；
- 第 03 章的 Harris 角点与 `(row, column)` 坐标；
- 向量范数、矩阵乘法、齐次坐标和 SVD 的基本概念。

## 1. 完整流水线

```text
图像 A/B → Harris 关键点 → 局部补丁描述 → 最近邻候选
        → 比率检验/双向检查 → RANSAC → Homography + 内点
```

检测器只回答“哪里值得看”。描述子把关键点周围的外观变成向量；匹配器比较向量；RANSAC 再检查这些对应关系是否服从同一个几何运动。外观相似不等于几何正确，这个分层非常重要。

## 2. 从零构造补丁描述子

以关键点为中心截取 `p×p` 灰度窗口，拉平成向量 `x`：

```math
d = \frac{x-\bar{x}}{\max(\lVert x-\bar{x}\rVert_2, \epsilon)}
```

减去均值使描述子对整体亮度偏移更稳定；除以 L2 范数降低全局对比度变化的影响。越过图像边界的关键点会被删除，因为补零会人为改变描述子。

这是教学基线，不具备旋转和尺度不变性。下一次升级到方向对齐、SIFT/ORB 时，你会明确知道新增结构解决了什么问题。

## 3. 最近邻、比率检验与双向一致性

对描述子 `dᵢ`，使用欧氏距离寻找图像 B 中最近和次近的候选：

```math
r = \frac{\lVert d_i-d_{1}\rVert_2}{\lVert d_i-d_{2}\rVert_2}
```

当 `r` 很小时，第一名明显优于第二名；当 `r≈1` 时，局部纹理存在歧义。代码默认 `ratio_threshold=0.8`。双向一致性还要求 `Aᵢ` 的最佳候选是 `Bⱼ`，且 `Bⱼ` 的最佳候选也必须是 `Aᵢ`。

![特征匹配候选](../../docs/assets/chapter04_feature_matches.png)

参考图由同一场景平移 `(row=12, column=18)` 生成。为了让几何剔除可观察，生成器确定性加入两个歧义候选；它们不是人工修图，重新执行脚本会得到相同结果。

## 4. Homography 与归一化 DLT

平面场景或纯旋转相机中的对应点满足：

```math
s\begin{bmatrix}u\\v\\1\end{bmatrix}
=H\begin{bmatrix}x\\y\\1\end{bmatrix},\qquad H\in\mathbb{R}^{3\times3}
```

`H` 只有八个独立自由度，因此至少需要四对非退化对应点。实现先把两组点平移到质心，再缩放到平均距离 `√2`，然后构造线性方程 `Ah=0`，取 SVD 最小奇异值对应的向量并反归一化。这一步能显著改善大坐标下的数值稳定性。

注意：特征模块的关键点使用 `(row, column)`；进入几何计算前必须转换为 `(x, y)=(column, row)`。

## 5. RANSAC：让多数一致性战胜错误匹配

每轮 RANSAC：

1. 随机抽取四对匹配；
2. 用 DLT 拟合候选 `H`；
3. 投影所有源点并计算重投影误差；
4. 把误差不超过阈值的对应标为内点；
5. 保留内点最多、平均误差更小的模型；
6. 用全部内点重新拟合最终 `H`。

![RANSAC 内点与离群点](../../docs/assets/chapter04_ransac_inliers.png)

绿色是服从统一平移几何的内点，红色是被 RANSAC 拒绝的错误候选。这里固定 `seed=11`，保证结果可复现。

## 6. 运行代码

从仓库根目录执行：

```bash
python -m pip install -e ".[dev]"
python -m vision2autonomy.examples.reference_images
pytest tests/test_matching.py -q
```

最小调用示例：

```python
from vision2autonomy.features.matching import (
    describe_patches, match_descriptors, ransac_homography
)

da = describe_patches(image_a, corners_a, patch_size=11)
db = describe_patches(image_b, corners_b, patch_size=11)
matches = match_descriptors(da.descriptors, db.descriptors, ratio_threshold=0.8)

src_xy = da.keypoints[matches.pairs[:, 0]][:, ::-1]
dst_xy = db.keypoints[matches.pairs[:, 1]][:, ::-1]
model = ransac_homography(src_xy, dst_xy, threshold=2.0, seed=11)
```

## 7. 参数与调试

| 参数 | 默认值 | 作用 | 调大后的影响 |
|---|---:|---|---|
| `patch_size` | 9 | 描述区域边长 | 信息更多，但更怕旋转、形变和遮挡 |
| `ratio_threshold` | 0.8 | 最近邻歧义门槛 | 匹配更多，同时误匹配风险增加 |
| `mutual` | `True` | 双向一致性 | 关闭后召回率更高、可靠性更低 |
| `threshold` | 2.0 px | RANSAC 内点重投影阈值 | 容忍噪声，也可能接纳错误点 |
| `max_iterations` | 1000 | RANSAC 抽样轮数 | 成功概率提高，耗时增加 |
| `seed` | 0 | 随机数种子 | 改变抽样序列；固定后可复现 |

遇到匹配太少时，依次检查：角点是否落在边界、补丁是否因旋转发生变化、比率阈值是否过严。遇到模型错误时，先查看红绿连线，而不是盲目增加 RANSAC 轮数。

## 8. 与自动驾驶的联系

- **视觉里程计**：跨帧匹配静态场景点，估计相机运动；
- **定位与建图**：把当前观测关联到地图地标；
- **多相机标定**：寻找相机间的共同观测；
- **三维重建**：匹配是三角化深度之前的数据关联步骤；
- **失效检测**：低纹理、重复车道线、动态车辆都会让匹配退化。

Homography 只适合平面或近似纯旋转情形。真实道路具有深度变化，下一阶段会进入相机模型、对极几何、基础矩阵和三角化。

## 9. 自测问题

1. 为什么 Harris 响应强不代表匹配一定正确？
2. 为什么重复窗格或车道虚线会让比率接近 1？
3. 四对点中只要有一对错误，直接 DLT 会怎样？
4. 为什么 RANSAC 的阈值单位是像素？
5. Homography 为什么不能完整表达具有明显视差的三维道路？

