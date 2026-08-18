# Vision2Autonomy

[English](README.md) | [简体中文](README.zh-CN.md)

从第一性原理学习计算机视觉，并逐步构建自动驾驶视觉感知系统。

Vision2Autonomy 是一个循序渐进、测试驱动的计算机视觉学习库。项目从图像处理的数值基础开始，逐步覆盖传统视觉、几何视觉和深度学习，最终形成一个基于摄像头的自动驾驶感知系统。

## 为什么创建这个仓库

每个主题都从三个层次展开：

1. **理解原理**：用清晰的文档解释数学原理、算法假设和设计选择。
2. **从零实现**：先使用 NumPy 实现核心算法，再引入 OpenCV、PyTorch 等高级工具。
3. **验证结果**：通过单元测试、参考实现对照和可复现实验验证正确性，而不只是展示一张看起来不错的结果图。

这里的“从零开始”并不意味着拒绝成熟框架，而是要求在使用框架前理解关键步骤，并能够解释算法为何有效、何时失效。

## 当前里程碑：从零实现 Canny

第一阶段已经实现：

- 支持明确边界填充方式的向量化二维卷积；
- Gaussian 核生成与平滑；
- Sobel 水平、垂直梯度、幅值和方向；
- 非极大值抑制；
- 双阈值分类与八邻域滞后连接；
- 可查看每个中间阶段的结果对象；
- 图片命令行处理工具；
- 可重复生成的案例图片与自动化测试。

核心实现没有调用 `cv2.Canny`。

## 安装与测试

建议使用 Python 3.10 或更高版本：

```bash
python -m pip install -e ".[dev]"
pytest
```

对单张图片执行边缘检测：

```bash
v2a-canny input.jpg edges.png --low 40 --high 100
```

也可以使用 Python API：

```python
from PIL import Image
import numpy as np

from vision2autonomy import canny

image = np.asarray(Image.open("input.jpg").convert("L"))
edges = canny(image, low_threshold=40, high_threshold=100)
Image.fromarray(edges).save("edges.png")
```

## 项目结构

```text
src/vision2autonomy/    可复用的正式库代码
chapters/               按学习顺序组织的章节文档
tests/                  单元测试与集成测试
examples/               可直接执行的兼容示例入口
docs/assets/            由代码生成并提交的参考图片
docs/roadmap.zh-CN.md   中文学习路线
assignment/             保留的 2016 年原始课程作业
```

完整发展计划见[中文路线图](docs/roadmap.zh-CN.md)。

## 案例文档约定

每个案例必须提供：

- 明确的学习目标和前置知识；
- 算法原理、公式、假设和实现步骤；
- 可以直接执行的复现命令；
- 完全一致的输入图片；
- 有意义的关键中间结果；
- 最终输出参考图；
- 参数说明与调整建议；
- 结果分析与已知局限；
- 覆盖示例路径的自动化测试。

生成的参考图片保存在 `docs/assets/`，并且必须能够由仓库中的代码重新生成。完整要求见[中文案例文档规范](docs/CASE_STANDARD.zh-CN.md)。

重新生成当前全部参考图：

```bash
python -m vision2autonomy.examples.reference_images
```

## 项目历史

这个仓库最初是一份 2016 年的计算机视觉课程作业。原始代码和提交历史被保留下来，新的代码库则按照现代工程规范重新建设。这样既能看到项目当前的实现，也能看到它从课程作业逐步发展为完整学习体系的过程。

