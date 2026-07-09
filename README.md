# makemore 学习项目 · 从 MLP 到 WaveNet

> 跟随 Andrej Karpathy「Neural Networks: Zero to Hero」系列，从零手写一个字符级语言模型：
> 输入 `names.txt`（32K 个英文名字），学会生成"像名字但不存在"的新名字。
> 全程手推反向传播、亲手实现 BatchNorm，并配了 8 篇成体系的中文笔记。

从 **bigram → MLP → BatchNorm → 模块化 → WaveNet**，一步步把一个能跑的语言模型拆到零件级理解。

---

## 📚 知识点全景

这个项目覆盖的核心知识点（每一条都有对应笔记 + 可跑的 notebook）：

**模型与数据**
- 字符级语言模型：滑动窗口构建样本、`.` 兼任起止符、train/dev/test 三段划分
- Embedding 分布式表示：查表 `C[X]`、`reshape` 展平拼接、PCA/SVD 可视化字符向量

**前向 & 反向传播**
- 手写反向传播（backprop ninja）：`dlogits = probs - onehot`、`tanh'=1-a²`、逐层链式法则
- Embedding 反向的特殊性：lookup 的梯度要按索引**累加**（`index_add_`）
- 用 `torch.autograd` 对拍验证手写梯度（误差 ~1e-9）

**初始化 & 数据流**
- 方差传播公式 `std(out)=√fan_in·std(W)·std(in)`，理解每层 std 怎么变
- LeCun / Xavier / Kaiming 三种初始化的推导与区别（前向 vs 前向+反向）
- 输出层压小权重让初始 loss≈`-ln(1/27)`，避免"曲棍球柄"

**激活函数**
- Sigmoid / Tanh / ReLU / Leaky·PReLU·ELU / GELU / Swish 的图像、导数、优缺点、选型
- tanh 饱和 → 反向梯度死亡，与"前向饱和=反向梯度消失"的一体两面

**归一化**
- BatchNorm：前向、running 统计量、backward 手推（逐节点 + 封闭式）、利弊
- LayerNorm：与 BN 只差"归一化轴"，为什么 Transformer 用它、RMSNorm 简介

**梯度问题**
- 梯度消失 / 爆炸的连乘本质，六大解药（初始化、归一化、不饱和激活、残差、梯度裁剪、合适 lr）

**训练与调优**
- 学习率：LR range test 找最优、阶梯衰减 / 余弦退火 / warmup
- Batch size：噪声 vs 速度权衡（噪声 ~1/√bs）
- 网格搜索 vs 随机搜索、正则化（dropout / weight decay）
- 评估指标大全：loss 曲线、激活直方图、update:data ratio、梯度健康、定性采样

**架构进阶**
- 模块化封装（`Linear / BatchNorm1d / Tanh / Embedding / Sequential` 类）
- **WaveNet**：用 `FlattenConsecutive` 做层级（树状）融合，让上下文逐层翻倍，替代一次性拍扁

---

## 🗂️ 文档导航

### 笔记（`00`–`08`，数字即阅读顺序）

| 文件 | 作用 |
|---|---|
| [00-笔记索引.md](00-笔记索引.md) | **总入口**：阅读路线图 + 七篇速览 + 一条主线串联 |
| [01-MLP训练完整笔记.md](01-MLP训练完整笔记.md) | 全流程：数据→模型→初始化→训练→诊断→调优→采样 |
| [02-数据流与梯度流笔记.md](02-数据流与梯度流笔记.md) | 数值直觉：一份数据穿过每层时 mean/std 怎么变（全实测数字） |
| [03-激活函数笔记.md](03-激活函数笔记.md) | 激活函数横向对比 + 梯度消失/爆炸 |
| [04-超参数与评估指标笔记.md](04-超参数与评估指标笔记.md) | 调参（lr/batch/搜索）+ 训练时采集什么指标去分析 |
| [05-BatchNorm笔记.md](05-BatchNorm笔记.md) | BN 深挖：动机/前向/backward 推导/利弊 |
| [06-embedding反向传播笔记.md](06-embedding反向传播笔记.md) | embedding 梯度回传为什么要"累加" |
| [07-LayerNorm笔记.md](07-LayerNorm笔记.md) | LN 深挖：前向反向代码+数学、与 BN 异同、Transformer 为何用它 |
| [08-底层论文清单.md](08-底层论文清单.md) | 每个知识点对应的原始论文（带链接）+ 推荐阅读顺序 |

### Notebook（对应 Karpathy 第 1–5 讲的进阶路径）

| 文件 | 作用 |
|---|---|
| `makemore-01.ipynb` | **第 1 讲 bigram**：计数法 + 单层神经网络起步 |
| `makemore-02-0.ipynb` | **MLP 基础**：embedding→tanh→softmax、训练循环、采样、PCA 可视化 |
| `makemore-02-1.ipynb` | **找学习率**：LR range test（lr 指数扫描）+ MLP 训练 |
| `makemore-02-2.ipynb` | **手写反向传播 baseline**：逐层手推 + autograd 对拍验证 |
| `makemore-02-3.ipynb` | **加 BatchNorm + 学习率衰减**：tanh 饱和 23%→6%，loss 双降 |
| `makemore-02-4-monitor-gridsearch.ipynb` | **完整监控 + 网格搜索**：训练指标看板 + 超参搜索 |
| `makemore-02-5-byclass.ipynb` | **模块化重构**：用 `classlist.py` 的层类搭建，向 PyTorch 风格靠拢 |
| `makemore-02-6.ipynb` | **WaveNet**：`FlattenConsecutive` 层级树状结构（n_embd=24, n_hidden=128） |

### 代码 & 数据

| 文件 | 作用 |
|---|---|
| `classlist.py` | 自己封装的层：`Linear / BatchNorm1d / Tanh / FlattenConsecutive / Embedding / Sequential` + `calculate_gain`，供 02-5 / 02-6 导入 |
| `names.txt` | 数据集：32K 个英文名字（来自 ssa.gov 2018） |
| `bn_*.png` | 02-3/02-4 训练时导出的监控图（loss 曲线、per-batch、监控面板） |

---

## 🧭 建议学习路线

```
notebook 主线（动手）           笔记（理解）
────────────────────────────────────────────────
01           bigram         →   —
02-0 / 02-1  MLP + 找 lr     →   01, 02, 04
02-2         手写反传        →   06(embedding 那步)
02-3         BatchNorm       →   03(激活), 05(BN)
02-4         监控 + 网格搜索  →   04(评估指标)
02-5         模块化          →   —
02-6         WaveNet         →   07(LayerNorm 作为对比)
                              ↑
                        全程可查 08(论文清单)
```

**下一步**：读 08 清单里的《Attention Is All You Need》——你会发现 embedding、LayerNorm、残差、GELU、Adam 这些零件都已经认识了，剩下的新东西主要就是 self-attention。makemore 到这里就接上 GPT 了。

---

## ⚙️ 环境

- Python 3.11 + PyTorch（本项目用 conda 环境 `dl`）
- 依赖：`torch`、`numpy`、`matplotlib`
- notebook 里的相对路径 `open("names.txt")` 要求在项目根目录启动，别把 notebook 移进子文件夹

---

## 🙏 来源与致谢

本仓库是学习 [Andrej Karpathy · makemore](https://github.com/karpathy/makemore) 与其
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 系列课程的产物，
在原仓库基础上补充了大量中文笔记、手写反向传播实现与实验。原始 `makemore.py` 及数据集版权归原作者，遵循 MIT 许可（见 `LICENSE`）。

