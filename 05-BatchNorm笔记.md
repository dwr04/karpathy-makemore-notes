# Batch Normalization 笔记

> 场景：`Linear -> BatchNorm -> Tanh` 这种结构里，为什么需要 BN、它怎么工作、backward 怎么推、以及利弊。

---

## 1. 动机

深层网络里，每一层的 pre-activation `h = Wx + b` 的分布，会随着训练中 `W, b` 的更新而漂移（哪怕初始化用了 Kaiming init，那也只保证 `t=0` 时刻分布合理）。分布一旦跑偏：

- 对 `tanh`：方差过大 → 大量神经元饱和（`|h|` 很大）→ 局部梯度 `1 - tanh(h)^2 ≈ 0` → 梯度消失。
- 层数越深，这种偏移是**复合**的，前面层的漂移会传导并放大到后面层。

BN 的思路：与其祈祷分布保持良好，不如在网络内部**显式、主动地**把每一层的 pre-activation 重新标准化。

---

## 2. 前向公式

对一个 batch（大小 `m`），某个神经元维度上的 pre-activation 值 $h_1, \dots, h_m$：

**Step 1 — batch 均值：**
$$
\mu_B = \frac{1}{m}\sum_{i=1}^m h_i
$$

**Step 2 — batch 方差：**
$$
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (h_i - \mu_B)^2
$$

**Step 3 — 标准化：**
$$
\hat{h}_i = \frac{h_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

**Step 4 — 缩放平移（可学习参数 $\gamma, \beta$）：**
$$
y_i = \gamma \hat{h}_i + \beta
$$

### 要点

- $\gamma, \beta$ 是可学习参数，不是写死的。如果没有它们，相当于强制每个神经元输出标准正态分布，剥夺了网络的表达能力。极端情况下网络可以学出 $\gamma = \sigma_B,\ \beta=\mu_B$，把标准化"撤销"。
- 紧跟在 BN 前面的 `Linear` 层，其 bias `b` 是多余的——因为减去 $\mu_B$ 时 `b` 会被直接抵消，等价于被 $\beta$ 取代。所以实践中常写 `Linear(..., bias=False)`。
- **训练 vs 推理**：训练时用当前 batch 的 $\mu_B,\sigma_B^2$；推理时改用训练过程中滑动平均得到的 running mean/var（不参与梯度计算），保证单样本预测时输出确定、不依赖 batch 组成：
$$
\mu_{\text{running}} \leftarrow (1-\alpha)\,\mu_{\text{running}} + \alpha\,\mu_B
$$
（$\sigma^2_{\text{running}}$ 同理）

---

## 3. 反向传播推导

### 3.1 为什么不好推

普通逐元素操作（比如 `tanh`）求导时，$y_i$ 只依赖 $h_i$ 自己。但 BN 里，$\mu_B$ 和 $\sigma_B^2$ 是**整个 batch** 的函数，所以 $\hat h_i$ 对某个 $h_j$ 求导时，不能只看 $i=j$ 这一项——batch 里每个样本都通过 $\mu_B, \sigma_B^2$ 这两条路径互相耦合。也就是说，$h_j$ 的一次改变，会通过均值/方差影响到**所有** $\hat h_i$，进而影响所有 $y_i$。

### 3.2 计算图拆解

把前向拆成基本节点，方便一步步反传：

$$
\mu_B = \frac{1}{m}\sum_i h_i
\quad\to\quad
d_i = h_i - \mu_B
\quad\to\quad
\sigma_B^2 = \frac{1}{m}\sum_i d_i^2
$$
$$
\quad\to\quad
s = (\sigma_B^2+\epsilon)^{-1/2}
\quad\to\quad
\hat h_i = d_i \cdot s
\quad\to\quad
y_i = \gamma \hat h_i + \beta
$$

已知上游梯度 $\dfrac{\partial L}{\partial y_i}$（记作 $dy_i$），逐节点反传：

**① 对 $\gamma, \beta$：**
$$
\frac{\partial L}{\partial \gamma} = \sum_{i=1}^m dy_i \cdot \hat h_i,
\qquad
\frac{\partial L}{\partial \beta} = \sum_{i=1}^m dy_i
$$

**② 对 $\hat h_i$：**
$$
d\hat h_i = dy_i \cdot \gamma
$$

**③ 对 $s$（$\hat h_i = d_i \cdot s$，$s$ 被所有 $i$ 共用）：**
$$
\frac{\partial L}{\partial s} = \sum_{i=1}^m d\hat h_i \cdot d_i
$$

**④ 对 $\sigma_B^2$（经过 $s=(\sigma_B^2+\epsilon)^{-1/2}$）：**
$$
\frac{\partial s}{\partial \sigma_B^2} = -\frac{1}{2}(\sigma_B^2+\epsilon)^{-3/2}
\quad\Rightarrow\quad
\frac{\partial L}{\partial \sigma_B^2} = \frac{\partial L}{\partial s}\cdot\left(-\frac12(\sigma_B^2+\epsilon)^{-3/2}\right)
$$

**⑤ 对 $d_i$：** 这里 $d_i$ 有两条路径——一条直接经过 $\hat h_i = d_i s$，一条经过 $\sigma_B^2 = \frac1m\sum d_i^2$：
$$
\frac{\partial L}{\partial d_i} = d\hat h_i \cdot s \;+\; \frac{\partial L}{\partial \sigma_B^2}\cdot \frac{2d_i}{m}
$$

**⑥ 对 $\mu_B$（经过 $d_i = h_i - \mu_B$，同样被所有 $i$ 共用）：**
$$
\frac{\partial L}{\partial \mu_B} = -\sum_{i=1}^m \frac{\partial L}{\partial d_i}
$$

**⑦ 对 $h_i$（同样两条路径：直接经过 $d_i$，以及经过 $\mu_B=\frac1m\sum h_i$）：**
$$
\frac{\partial L}{\partial h_i} = \frac{\partial L}{\partial d_i} + \frac{1}{m}\cdot\frac{\partial L}{\partial \mu_B}
$$

把 ①→⑦ 串起来就是完整链式法则，可以看到 $dh_i$ 确实依赖 batch 里所有其他样本的 $dy_j$，这正是"耦合"的来源。

### 3.3 化简到封闭形式

上面拆分法对应到代码里可以逐步实现，但如果代入化简，利用两个恒等式：

$$
\sum_{i=1}^m \hat h_i = 0, \qquad \sum_{i=1}^m \hat h_i^2 = m
$$

（这两个恒等式源于 $\hat h$ 本身就是标准化过的量，均值为 0、方差为 1）

可以把 ①~⑦ 合并成一个非常紧凑的封闭表达式：

$$
\frac{\partial L}{\partial h_i} = \frac{\gamma}{m\sqrt{\sigma_B^2+\epsilon}}\left[
m\, dy_i \;-\; \sum_{j=1}^m dy_j \;-\; \hat h_i \sum_{j=1}^m dy_j \hat h_j
\right]
$$

这个形式在实现里只需要两次 `sum`（对 $dy$ 求和、对 $dy \cdot \hat h$ 求和），比逐节点反传省掉大量中间变量，是各框架底层 BN kernel 的标准写法。

> 推导练习建议：自己从 ①~⑦ 出发，把 $d_i, s, \sigma_B^2$ 都用 $\hat h_i$ 和 $\sigma_B$ 代换回去，再用上面两个恒等式消项，能推出这个封闭式——这是很好的"backprop ninja"式练习。

---

## 4. 优缺点

### 优点

- **缓解梯度消失/爆炸**：主动把 pre-activation 拉回到合理范围（均值0方差1附近），配合 $\gamma,\beta$ 保留网络表达力。
- **降低对初始化的敏感度**：即使初始化不够精细，BN 也能在训练中把分布"拉回来"，让训练对学习率、初始化的选择更宽容。
- **允许更大的学习率**：分布被稳定控制后，网络对较大学习率的容忍度提高，收敛更快。
- **一定的正则化效果**：训练时用的是当前 batch 的统计量，天然带有噪声（同一样本在不同 batch 里被不同地标准化），有类似 dropout 的正则化副作用。

### 缺点

- **强依赖 batch size**：batch 太小时 $\mu_B, \sigma_B^2$ 估计噪声很大，训练不稳定；batch=1 时方差为 0，直接失效。
- **训练/推理行为不一致**：训练用 batch 统计量，推理用 running 统计量，这个切换（`model.train()` / `model.eval()`）是很多隐蔽 bug 的来源——忘记切换会导致推理结果异常。
- **样本间产生耦合**：同一样本的输出在数学上依赖于同 batch 里的其他样本，这在语义上比较别扭，在某些场景（比如在线学习、序列长度变化的场景）不适用。
- **对序列模型不友好**：在 RNN/Transformer 里，每个时间步的统计量不稳定，且不同 batch 内序列长度可能不同，BN 用起来很别扭。

> 这也是为什么后来的架构（尤其 Transformer）几乎都换成了 **LayerNorm**——对单个样本内部的特征做标准化，不依赖 batch，没有上面这些耦合问题。

---

## 5. Kaiming Init vs BatchNorm：分工

| | Kaiming Init | BatchNorm |
|---|---|---|
| 解决的问题 | 起点在哪（$t=0$ 时刻分布） | 起点之后怎么走（训练过程中主动纠偏） |
| 性质 | 一次性、静态 | 每次前向都生效、动态 |
| 关系 | 二者互补，不是替代 |

好的初始化让训练"起跑姿势正确"，BN 让每一步训练都能自我纠偏，二者结合效果最好；实践中会发现用了 BN 之后，网络对初始化的敏感度明显下降。




完整求导
好，我们用你刚才在纠结的那套变量名（`hprebn`, `bnmeani`, `bnvar`, `bnraw`, `bngain`, `bnbias`）重新走一遍，这样能跟你实际的代码对上。

## 0. 前向回顾（明确每一步的形状）

```python
bnmeani = hprebn.mean(0, keepdim=True)          # (1, n)
bndiff  = hprebn - bnmeani                       # (m, n)
bndiff2 = bndiff ** 2                             # (m, n)
bnvar   = bndiff2.sum(0, keepdim=True) / (m-1)   # (1, n)  -- 注意这里是 m-1（贝塞尔校正）
bnvar_inv = (bnvar + 1e-5) ** -0.5                # (1, n)
bnraw   = bndiff * bnvar_inv                      # (m, n)
z1      = bngain * bnraw + bnbias                 # (m, n)
```

固定某一列 $j$（某个神经元），把 `m` 个样本的下标记为 $i=1,\dots,m$，对应到笔记的记号：

$$
h_i \to \text{hprebn}_{i,j},\quad \mu_B \to \text{bnmeani}_j,\quad d_i \to \text{bndiff}_{i,j}
$$
$$
\sigma_B^2 \to \text{bnvar}_j,\quad s\to \text{bnvar\_inv}_j,\quad \hat h_i \to \text{bnraw}_{i,j},\quad y_i \to z1_{i,j}
$$

## 1. 从最后往前，一个节点一个节点拆

已知上游传来的 $dz1 = \partial L/\partial z1$，形状 `(m, n)`。

### ① `z1 = bngain * bnraw + bnbias`

`bngain`, `bnbias` 是 `(1,n)`，前向被广播复制了 `m` 份 → **反向要把这 `m` 份梯度加回来**（这是上次讲广播时说的"前向广播，反向求和"）：

```python
dbngain = (dz1 * bnraw).sum(0, keepdim=True)   # (1, n)
dbnbias = dz1.sum(0, keepdim=True)             # (1, n)
dbnraw  = dz1 * bngain                          # (m, n)  -- 这一步不用求和，逐元素乘，形状不变
```

对应数学式（固定神经元 $j$，对 $i$ 求和）：
$$
\frac{\partial L}{\partial \gamma_j} = \sum_{i=1}^m dz1_{i,j}\cdot \text{bnraw}_{i,j}, \qquad
\frac{\partial L}{\partial \beta_j} = \sum_{i=1}^m dz1_{i,j}
$$

### ② `bnraw = bndiff * bnvar_inv`

这里 `bndiff` 是 `(m,n)`，`bnvar_inv` 是 `(1,n)` —— 又是一次广播，同样的道理，`bnvar_inv` 反向要对 `i` 求和：

```python
dbndiff_1  = dbnraw * bnvar_inv                        # (m, n) 这条路径先算一部分
dbnvar_inv = (dbnraw * bndiff).sum(0, keepdim=True)    # (1, n)
```

注意我这里叫 `dbndiff_1`，因为 `bndiff` 待会儿还有第二条路径要汇入（见 ④），这里先只算第一条。

### ③ `bnvar_inv = (bnvar + 1e-5) ** -0.5`

纯逐元素操作，链式法则直接求导：
$$
\frac{\partial \text{bnvar\_inv}}{\partial \text{bnvar}} = -\frac12(\text{bnvar}+\epsilon)^{-3/2}
$$
```python
dbnvar = dbnvar_inv * (-0.5 * (bnvar + 1e-5) ** -1.5)   # (1, n)
```

### ④ `bnvar = bndiff2.sum(0) / (m-1)`

`sum(0)` 反向就是把梯度**广播回去**（跟 sum 相反：前向求和，反向复制）：
```python
dbndiff2 = dbnvar * (1.0 / (m-1)) * torch.ones_like(bndiff2)   # (m, n)
```

### ⑤ `bndiff2 = bndiff ** 2`

```python
dbndiff_2 = dbndiff2 * 2 * bndiff    # (m, n)  -- 这是 bndiff 的第二条路径
```

**到这里，`bndiff` 两条路径的梯度要相加：**
```python
dbndiff = dbndiff_1 + dbndiff_2      # (m, n)
```

这正是笔记里说的"$d_i$ 有两条路径"——一条经过 $\hat h_i = d_i \cdot s$（对应这里的①②），一条经过 $\sigma_B^2 = \frac1m\sum d_i^2$（对应④⑤）。

### ⑥ `bndiff = hprebn - bnmeani`

`hprebn` 是 `(m,n)`，`bnmeani` 是 `(1,n)`，减法也是广播：
```python
dhprebn_1 = dbndiff.clone()                          # 直接路径，(m, n)
dbnmeani  = (-dbndiff).sum(0, keepdim=True)           # (1, n)  -- 广播的反向是求和
```

### ⑦ `bnmeani = hprebn.mean(0, keepdim=True)`

`mean` 反向是"复制并均分"：
```python
dhprebn_2 = dbnmeani * (1.0/m) * torch.ones_like(hprebn)   # (m, n)
```

**最终 `hprebn` 两条路径相加：**
```python
dhprebn = dhprebn_1 + dhprebn_2
```

## 2. 逐节点版本 vs 封闭形式

上面这一大串就是笔记里公式①~⑦逐节点展开后、按你实际变量名重写的版本——每一步都对应得上。Karpathy 在课上会让你先写这个"慢速正确版"去对 `torch.autograd` 验证数值是否一致，验证过了之后，再推导笔记里那个化简后的封闭形式：

$$
\frac{\partial L}{\partial \text{hprebn}_{i,j}} = \frac{\gamma_j}{m\sqrt{\sigma_{B,j}^2+\epsilon}}\left[
m\, dz1_{i,j} - \sum_{k=1}^m dz1_{k,j} - \hat h_{i,j}\sum_{k=1}^m dz1_{k,j}\hat h_{k,j}
\right]
$$

对应代码里差不多是这样一行（`bnvar_inv` 就是 $\frac{1}{\sqrt{\sigma_B^2+\epsilon}}$）：

```python
dhprebn = bngain * bnvar_inv / m * (
    m * dz1 
    - dz1.sum(0, keepdim=True) 
    - bnraw * (dz1 * bnraw).sum(0, keepdim=True)
)
```

## 3. 建议的验证方式

如果你现在在写这部分练习，最保险的做法是：

1. 先写上面①~⑦"慢速版"，每一步都是单独的中间变量。
2. 用 `torch.autograd.grad` 或者 `.backward()` 拿到 PyTorch 自动求导算出来的 `hprebn.grad`。
3. 用 `torch.allclose` 或者手算最大误差（`(dhprebn - hprebn.grad).abs().max()`）对比，误差应该在 `1e-9` 量级。
4. 再换成上面那个封闭形式的单行代码，同样验证一遍，确认两个版本梯度完全一致。

这一套"手推 → 慢速代码验证 → 化简 → 再验证"的流程，正是 Karpathy "become a backprop ninja" 那节课要练的东西。你要是想，我可以把这几步写成一段完整可跑的 PyTorch 代码，让你直接在 Mac 上跑一遍数值验证。