# Layer Normalization 笔记

> 姊妹篇 `05-BatchNorm笔记.md`。LN 和 BN 的数学几乎一模一样,**唯一的区别是"沿哪个轴归一化"**:
> BN 沿 batch 轴(跨样本),LN 沿 feature 轴(单样本内部)。本篇的 backward 完全模仿 BN 那篇的写法。
> 本篇所有代码都用 `nn.LayerNorm` 和 `torch.autograd` **数值验证过**(误差 ~1e-6)。

---

## 0. 一张图看懂 BN vs LN 的差别

设输入是 `(m, n)`:`m` 个样本,每个样本 `n` 维特征。

```
        feature →  (n 维)
       ┌───────────────────────┐
样本↓  │ x11 x12 x13 ... x1n    │  ← LN: 对这一行(一个样本)内部求 μ,σ
 (m个) │ x21 x22 x23 ... x2n    │  ← LN: 对这一行
       │ ...                    │
       │ xm1 xm2 xm3 ... xmn    │
       └───────────────────────┘
          ↑
          BN: 对这一列(一个特征,跨所有样本)求 μ,σ
```

- **BN**:每个**特征**独立,统计量沿 batch 轴(dim 0)算 → μ,σ 形状 `(1, n)`。**依赖 batch**。
- **LN**:每个**样本**独立,统计量沿 feature 轴(dim 1)算 → μ,σ 形状 `(m, 1)`。**不依赖 batch**。

> 一句话:**BN 归一化"竖着切",LN 归一化"横着切"。** 剩下的公式、推导、代码结构全都一样,只是求和的轴换了。

---

## 1. 动机:为什么要有 LN

BN 的两个致命短板(见 BN 笔记第 4 节):
1. **强依赖 batch size**:batch 小时统计量噪声大;batch=1 时方差为 0 直接失效。
2. **训练/推理不一致**:要维护 running mean/var,`train()`/`eval()` 切换是 bug 温床。
3. **对序列模型不友好**:RNN/Transformer 里每个时间步统计量不稳,序列长度还可能变。

LN 的思路:**既然跨样本统计有这么多麻烦,那就别跨样本——在单个样本内部、对它自己的 n 个特征做标准化。**
这样:
- 不依赖 batch(batch=1 也能用,每个样本自成一体)。
- 训练和推理**完全一致**(不需要 running 统计量,不需要区分 train/eval)。
- 天然适配变长序列(每个 token 各归一化各的)。

---

## 2. 前向:代码 + 数学

对第 `i` 个样本(第 `i` 行),在它自己的 `n` 个特征 $x_{i,1},\dots,x_{i,n}$ 上:

**Step 1 — 样本内均值(沿特征轴):**
$$
\mu_i = \frac{1}{n}\sum_{k=1}^n x_{i,k}
$$

**Step 2 — 样本内方差(有偏,除以 n):**
$$
\sigma_i^2 = \frac{1}{n}\sum_{k=1}^n (x_{i,k}-\mu_i)^2
$$

**Step 3 — 标准化:**
$$
\hat x_{i,k} = \frac{x_{i,k}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}
$$

**Step 4 — 缩放平移(可学习 $\gamma,\beta$,形状 `(n,)`,跨样本共享):**
$$
y_{i,k} = \gamma_k\,\hat x_{i,k} + \beta_k
$$

```python
lnmean    = x.mean(1, keepdim=True)                  # (m, 1)  ← 沿 dim 1(特征)!
lndiff    = x - lnmean                                # (m, n)
lnvar     = (lndiff**2).mean(1, keepdim=True)         # (m, 1)  有偏,除以 n
lnvar_inv = (lnvar + 1e-5) ** -0.5                    # (m, 1)
lnraw     = lndiff * lnvar_inv                         # (m, n)
y         = lngain * lnraw + lnbias                    # (m, n)  lngain/lnbias 是 (n,)
```

> **和 BN 代码对比**:把 BN 里所有的 `.mean(0)` / `.sum(0)` 换成 `.mean(1)` / `.sum(1)`,统计量形状从 `(1,n)` 变成 `(m,1)`,就是 LN。**其余一模一样。**

### 两个关键的"轴不对称"

这是 LN 最容易搞混、也是和 BN 最本质的区别:

| | 归一化统计量(μ,σ) | 可学习参数 γ,β |
|---|---|---|
| 沿哪个轴 | **feature 轴 dim 1** | 每个 feature 一个,沿 **batch 轴 dim 0** 广播 |
| 形状 | `(m, 1)` | `(n,)` |

也就是说:**归一化沿特征轴,但 γ,β 沿样本轴广播(每个特征一份,所有样本共用)。** 这个"错位"直接决定了 backward 里哪些求和沿 dim 0、哪些沿 dim 1(见下)。

---

## 3. 反向:逐节点推导(模仿 BN)

### 3.1 耦合发生在哪

和 BN 一样,$\mu_i,\sigma_i^2$ 是一整行的函数,所以同一个样本内 `n` 个特征通过 μ,σ 互相耦合——改动 $x_{i,p}$ 会影响该行**所有** $\hat x_{i,k}$。
**区别**:BN 的耦合发生在**同一列的 m 个样本之间**;LN 的耦合发生在**同一行的 n 个特征之间**。所以下面所有"沿归一化轴的求和"都是 `sum(1)` 而不是 BN 的 `sum(0)`。

### 3.2 逐节点反传(和 BN 笔记 ①~⑦ 一一对应)

已知上游 `dy = ∂L/∂y`,形状 `(m, n)`。

**① `y = lngain * lnraw + lnbias`**
`lngain/lnbias` 是 `(n,)`,前向沿 batch 轴(dim 0)广播了 `m` 份 → **反向沿 dim 0 求和**:

```python
dlngain = (dy * lnraw).sum(0)      # (n,)   ← 沿样本轴求和(和BN的sum(0)一样!)
dlnbias = dy.sum(0)                # (n,)
dlnraw  = dy * lngain              # (m, n) 逐元素,形状不变
```

$$
\frac{\partial L}{\partial\gamma_k}=\sum_{i=1}^m dy_{i,k}\,\hat x_{i,k},\qquad
\frac{\partial L}{\partial\beta_k}=\sum_{i=1}^m dy_{i,k}
$$

> 注意:γ,β 的梯度这里是 `sum(0)`——因为 γ,β 是沿样本轴共享的。这一步的轴和 BN **相同**,别被"LN 沿 dim 1"带偏。

**② `lnraw = lndiff * lnvar_inv`**(`lnvar_inv` 是 `(m,1)`,沿 dim 1 广播 → 反向沿 dim 1 求和)

```python
dlndiff_1  = dlnraw * lnvar_inv                     # (m, n) 路径1
dlnvar_inv = (dlnraw * lndiff).sum(1, keepdim=True) # (m, 1) ← 沿特征轴求和
```

**③ `lnvar_inv = (lnvar + eps) ** -0.5`**(逐元素)

```python
dlnvar = dlnvar_inv * (-0.5 * (lnvar + 1e-5) ** -1.5)   # (m, 1)
```
$$
\frac{\partial\,\text{lnvar\_inv}}{\partial\,\text{lnvar}} = -\tfrac12(\sigma_i^2+\epsilon)^{-3/2}
$$

**④ `lnvar = (lndiff**2).mean(1)`**(mean 反向 = 除以 n 再沿 dim 1 广播回去)

```python
dlndiff2 = dlnvar * (1.0/n) * torch.ones_like(lndiff)   # (m, n)
```

**⑤ `lndiff2 = lndiff ** 2`**(lndiff 的第二条路径)

```python
dlndiff_2 = dlndiff2 * 2 * lndiff                       # (m, n)
dlndiff   = dlndiff_1 + dlndiff_2                       # 两条路径汇合
```
> 和 BN 一样,$d_i$(这里是 lndiff)有两条路径:一条经 $\hat x = d\cdot s$(①②),一条经 $\sigma^2=\frac1n\sum d^2$(④⑤)。

**⑥ `lndiff = x - lnmean`**(减法广播,`lnmean` 沿 dim 1)

```python
dx_1     = dlndiff.clone()                             # 直接路径 (m, n)
dlnmean  = (-dlndiff).sum(1, keepdim=True)             # (m, 1) ← 沿特征轴求和
```

**⑦ `lnmean = x.mean(1)`**(mean 反向 = 除以 n 再沿 dim 1 广播)

```python
dx_2 = dlnmean * (1.0/n) * torch.ones_like(x)          # (m, n)
dx   = dx_1 + dx_2                                      # x 两条路径汇合
```

**验证结果**:上面逐节点版对 `torch.autograd` 的 `dx` 最大误差 **4.8e-7**,`dgain` 1.9e-6,`dbias` 0 —— 全部一致 ✅。

---

## 4. 反向:化简成封闭形式

和 BN 完全同构,把 ①~⑦ 代入化简(用 $\sum_k \hat x_{i,k}=0,\ \sum_k\hat x_{i,k}^2=n$),得:

$$
\frac{\partial L}{\partial x_{i,k}} = \frac{1}{n\sqrt{\sigma_i^2+\epsilon}}\left[
n\,d\hat x_{i,k} \;-\; \sum_{p=1}^n d\hat x_{i,p} \;-\; \hat x_{i,k}\sum_{p=1}^n d\hat x_{i,p}\hat x_{i,p}
\right]
$$

其中 $d\hat x_{i,k}=dy_{i,k}\cdot\gamma_k$,所有 $\sum_p$ **沿特征轴(同一行的 n 个特征)**。

```python
dlnraw = dy * lngain                                   # (m, n)  = dxhat
dx = lnvar_inv / n * (
    n * dlnraw
    - dlnraw.sum(1, keepdim=True)                      # ← 沿 dim 1
    - lnraw * (dlnraw * lnraw).sum(1, keepdim=True)    # ← 沿 dim 1
)
```

**验证结果**:封闭式对 autograd 的 `dx` 最大误差 **9.5e-7** ✅。

### ⚠️ 和 BN 封闭式的一个隐蔽区别

BN 封闭式里 `bngain` 能**提到括号外面**当公因子:
```python
dhprebn = bngain * bnvar_inv / m * ( m*dz1 - dz1.sum(0) - bnraw*(dz1*bnraw).sum(0) )
#         ^^^^^^ 提到外面
```
因为 BN 里 γ 沿 batch 轴是常数(对求和的那个轴而言不变),可以因式分解出来。

但 **LN 不行**:LN 沿特征轴求和,而 γ_k 恰恰是**逐特征变化**的,它在求和号里面变来变去,**提不出来**。所以 LN 必须先算 `dlnraw = dy * lngain`,把 γ **留在括号内**:
```python
dx = lnvar_inv / n * ( n*dlnraw - dlnraw.sum(1) - lnraw*(dlnraw*lnraw).sum(1) )
#    没有 lngain 在最外面, 它已经融进 dlnraw 了
```
> 这是把 BN 代码改成 LN 时**最容易写错**的一处:不能简单把 `sum(0)` 改成 `sum(1)`、再把 gain 留在外面——gain 必须先乘进 dxhat。

---

## 5. LN vs BN 异同总表

| 维度 | BatchNorm | LayerNorm |
|---|---|---|
| 归一化轴 | batch 轴(跨样本,dim 0) | feature 轴(单样本内,dim 1) |
| 统计量 μ,σ 形状 | `(1, n)` | `(m, 1)` |
| 是否依赖 batch size | **是**(小 batch 失效) | **否**(batch=1 也行) |
| 训练/推理是否一致 | 否,需 running 统计量 + train/eval 切换 | **是**,完全一致,无需 running |
| 样本间是否耦合 | 是(同列样本互相影响) | **否**(每个样本独立) |
| γ,β 形状 | `(1, n)` | `(n,)` |
| γ,β 梯度求和轴 | dim 0 | dim 0(**相同!**) |
| 归一化反向求和轴 | dim 0 | dim 1 |
| 封闭式里 gain | 可提到外面 | 必须乘进 dxhat |
| 正则化副作用 | 有(batch 噪声) | 基本没有 |
| 典型场景 | CNN、图像 | Transformer、RNN、NLP |

**相同点**:数学形式完全同构(标准化 + 仿射),backward 都是"两条路径 + 两次求和"的封闭式,都用 γ,β 保住表达力。
**不同点**:一句话——**归一化的轴不同**,由此派生出上表所有差异。

---

## 6. 为什么 Transformer 用 LN 而不是 BN

Transformer 的输入是 `(batch, seq_len, d_model)`——一批句子,每句若干 token,每个 token 是 `d_model` 维向量。用 BN 会同时踩中它所有的雷:

1. **序列变长,batch 统计量不稳**
   不同 batch 里句子长短不一、padding 多少不一,"同一位置跨样本"的统计量噪声极大,甚至无意义。LN 对**每个 token 自己的 d_model 维**归一化,和序列长度、batch 大小完全解耦。

2. **自回归推理时 batch 常常=1**
   生成任务一个一个 token 往外蹦,推理时 batch=1、序列逐步增长。BN 在 batch=1 时方差为 0 直接崩;LN 完全不受影响(每个 token 自成一体)。

3. **训练/推理必须一致**
   BN 要维护 running 统计量并切换 train/eval,在逐 token 生成、KV cache 这类场景里极易出错。LN 训练推理同一套公式,天然安全。

4. **样本独立性符合语义**
   一个句子的表示不应该依赖"同 batch 里碰巧还有哪些别的句子"。BN 引入的这种跨样本耦合在 NLP 里是有害的;LN 每个 token 独立,干净。

5. **残差结构的搭配**
   Transformer 是 `x + Sublayer(x)` 的残差堆叠,LN 放在残差路径上(Pre-LN: `x + Sublayer(LN(x))`)能稳定每一层的输入尺度,是深层 Transformer 能训起来的关键之一。

> 一句话:**Transformer 面对的是"变长、batch 常为 1、要逐 token 生成、样本必须独立"的场景,而这些恰好全是 BN 的死穴、LN 的主场。**

补充:后来还衍生出 **RMSNorm**(LLaMA 等在用)——LN 的简化版,只除以均方根、不减均值、去掉 β,更省算力,效果相当。

---

## 7. 一句话总结

- **LN = 把 BN 的归一化轴从"跨样本"换成"单样本内跨特征"**,数学同构,代码就是 `.mean(0)`→`.mean(1)`。
- **前向**:对每行的 n 个特征求 μ,σ 标准化,再逐特征 γ,β 仿射。
- **反向**:逐节点版和 BN 一一对应(耦合从"同列样本"变成"同行特征");封闭式同构,但 **gain 必须乘进 dxhat、不能提到外面**(因为它沿求和轴变化)。
- **优于 BN 之处**:不依赖 batch、训练推理一致、样本独立、适配变长序列。
- **Transformer 选 LN**:变长 + batch=1 生成 + 样本独立 + 残差稳定,全是 BN 的雷、LN 的主场。
