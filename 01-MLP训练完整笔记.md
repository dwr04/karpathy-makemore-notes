# MLP 字符级语言模型 · 训练全流程笔记

> 基于 `makemore-02-2`(手写反向传播 baseline)和 `makemore-02-3`(加 BatchNorm + 学习率衰减)
> 覆盖:数据→模型→初始化→训练→诊断→调优→采样 的完整链路
> 配套:`06-embedding反向传播笔记.md`(单独讲 demb→dC 那一步)

---

## 0. 整体架构一览

```
字符索引 Xb (m, block_size)
   │  C[Xb]  查表 embedding
   ▼
emb (m, block_size, embedding_dim)
   │  reshape 展平拼接
   ▼
emb_flat (m, block_size*embedding_dim)
   │  @ W1          ← 隐藏层线性变换 (02-3 去掉了 b1)
   ▼
hprebn (m, hidden_size)
   │  BatchNorm     ← 02-3 新增
   ▼
z1 (m, hidden_size)
   │  tanh          ← 非线性
   ▼
a1 (m, hidden_size)
   │  @ W2 + b2     ← 输出层
   ▼
logits (m, vocab_size)
   │  softmax → cross_entropy
   ▼
loss (标量)
```

**参数清单**(02-3):`C, W1, bngain, bnbias, W2, b2` —— 注意没有 `b1`(被 BatchNorm 的减均值抵消)。

---

## 1. 数据准备

### 1.1 字符 ↔ 索引映射

```python
chars = sorted(list(set(''.join(words))))
stoi = {'.': 0}                                  # '.' 固定占 0,作为 起始/结束 符
stoi.update({s: i+1 for i, s in enumerate(chars)})
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)                            # 27 (26字母 + '.')
```

**要点**:`.` 一符两用,既是序列开头的 padding,也是序列结束标记。模型学会"输出 `.`"就等于学会"名字到这里该结束了"。

### 1.2 滑动窗口构建样本 (block_size=3)

```python
for w in words:
    context = [0] * block_size            # 开头用 '...' 填充
    for ch in w + '.':                    # 结尾补 '.' 让模型学结束
        ix = stoi[ch]
        X.append(context)                 # 输入:前3个字符
        Y.append(ix)                      # 标签:第4个字符
        context = context[1:] + [ix]      # 窗口右移一格
```

`emma` → `... → e`,`..e → m`,`.em → m`,`emm → a`,`mma → .`(结束)。

**block_size 的权衡**:越大看的上下文越长、模型越强,但输入维度 `block_size*embedding_dim` 线性增长,参数和计算量都上升。这是超参数,值得调。

### 1.3 三段划分 train/dev/test = 80/10/10

```python
indices = torch.randperm(N)               # 打乱,避免按名字顺序切分产生偏差
n1 = int(N * 0.8); n2 = int(N * 0.9)
X_train = X[indices[:n1]]     # 训练:更新参数
X_dev   = X[indices[n1:n2]]   # 验证:调超参数、判断过拟合
X_test  = X[indices[n2:]]     # 测试:最终报告,全程只碰一次
```

**铁律**:test 集在整个开发过程中一次都不能看,否则你会不自觉地把超参数往 test 上过拟合,报出来的分数是假的。调参只用 dev。

---

## 2. 参数初始化(最容易被忽视、但影响巨大)

### 2.1 输出层:压小,让初始 loss 接近理论值

```python
W2 = torch.randn((hidden_size, vocab_size)) * 0.01   # 乘 0.01 压小
b2 = torch.zeros((vocab_size,))
```

**为什么**:初始时希望模型对 27 个字符"一视同仁"(均匀分布),此时
`loss ≈ -ln(1/27) ≈ 3.30`。如果 W2 不压小,初始 logits 会乱七八糟地放大,
softmax 输出某些字符概率畸高 → 初始 loss 可能是几十,前几百步全浪费在
"把这个错误的自信掰回来"上(loss 曲线开头那个陡降的"曲棍球柄")。

> 02-3 实测初始 loss = 3.295,和 3.30 几乎吻合 ✅

### 2.2 隐藏层:Kaiming 初始化,控制激活值方差

```python
W1 = torch.randn((embedding_dim*block_size, hidden_size)) * (5/3)/(fan_in**0.5)
#                                                             ↑gain  ↑扇入归一化
```

- `fan_in = embedding_dim*block_size`(输入维度),除以它的开方 → 让 `hprebn` 的方差保持在 ~1,不随层宽变化而爆炸或消失。
- `5/3` 是 **tanh 的增益**:tanh 会压缩数据(斜率<1),乘 5/3 补偿这个收缩,让激活分布不至于挤成一团。

### 2.3 一句话:初始化的两个目标

1. **输出层**要"谦虚"(logits 接近 0)→ 初始 loss 正常。
2. **隐藏层**要让激活值方差 ≈ 1,不饱和也不塌缩 → 梯度能顺畅流动。

有了 BatchNorm 之后,第 2 点的压力大大减轻(见下节),但理解原理仍然重要。

---

## 3. 前向传播

```python
emb      = C[Xb]                          # (m, block, embd) 查表
emb_flat = emb.reshape(emb.shape[0], -1)  # (m, block*embd) 展平
hprebn   = emb_flat @ W1                   # (m, hidden) 线性,无 b1
# --- BatchNorm(见第 6 节) ---
z1       = bngain * bnraw + bnbias
a1       = torch.tanh(z1)                  # 非线性
logits   = a1 @ W2 + b2                    # (m, vocab)
```

### softmax 的数值稳定技巧

```python
logits_max = logits.max(1, keepdim=True).values
logits = logits - logits_max              # 每行减最大值,不改变 softmax 结果
counts = logits.exp()                     # 现在最大是 exp(0)=1,不会 overflow
probs  = counts / counts.sum(1, keepdim=True)
```

**为什么减最大值**:`exp(大数)` 会溢出成 inf。softmax 对整体平移不变
(`softmax(x) = softmax(x-c)`),所以减掉每行最大值,数学结果不变但数值安全。

---

## 4. 损失函数:交叉熵

```python
loss = -probs[range(m), Yb].log().mean()
# 等价于 F.cross_entropy(logits, Yb)
```

只取"正确答案那个字符"的预测概率,取 log 取负再平均。概率越接近 1,loss 越接近 0。

- **perplexity = exp(loss)**,更直观:`exp(2.21) ≈ 9.1` 意思是"平均每个位置模型在约 9 个候选间纠结"。可以拿来和别人横向比。

---

## 5. 反向传播(手写,逐层)

链式法则,从 loss 一路回传。核心记住:**任何变量 X 的梯度 dX 形状永远和 X 相同**。

```python
# softmax + cross_entropy 合并求导,结果极简洁
dlogits = probs.clone()
dlogits[range(m), Yb] -= 1.0              # dlogits = probs - onehot(Y)
dlogits /= m                              # 因为 loss 用了 .mean()

dW2  = a1.T @ dlogits                      # 输出层权重
db2  = dlogits.sum(0)
da1  = dlogits @ W2.T
dz1  = da1 * (1.0 - a1**2)                 # tanh 导数 = 1 - tanh²

# --- BatchNorm 反向(见第 6 节)---
dW1  = emb_flat.T @ dhprebn
demb_flat = dhprebn @ W1.T
demb = demb_flat.reshape(emb.shape)
dC   = torch.zeros_like(C)
dC.index_add_(0, Xb.flatten(), demb.view(-1, demb.shape[-1]))  # 见 embedding笔记
```

### 两个高频记忆点

1. **`dlogits = (probs - onehot(Y)) / m`**:softmax+CE 的梯度就是"预测 - 真实",再除以 batch size(因为 mean)。这是整个反传里最优雅的一步。
2. **`tanh 导数 = 1 - a1²`**:直接用输出算,不用碰输入,省事。

### 验证手法:autograd 对拍

用 `.clone().detach().requires_grad_(True)` 复制一份参数,同样的输入跑一遍
`loss.backward()`,再逐个 `torch.allclose(手写梯度, xxx.grad, atol=1e-6)`。
02-3 里 6 个梯度全部 ✅,最大差异 ~1e-9,说明手写公式没错。

---

## 6. BatchNorm(02-3 核心新增)

### 6.1 动机

02-2 训练完 tanh 饱和严重:`|a1|>0.99` 的占比达 **23%**,这些神经元梯度≈0
(因为 tanh 导数 `1-a1²` 在 ±1 处→0),等于网络容量被浪费掉近 1/4。
BatchNorm 强制把进 tanh 前的分布拉回"均值0、方差1",直接解决饱和。

### 6.2 前向

```python
bnmean   = hprebn.mean(0, keepdim=True)              # 沿 batch 维求均值
bnvar    = hprebn.var(0, keepdim=True, unbiased=True)
bnvar_inv = (bnvar + 1e-5) ** -0.5                    # +1e-5 防除零
bnraw    = (hprebn - bnmean) * bnvar_inv              # 标准化到 N(0,1)
z1       = bngain * bnraw + bnbias                    # 再用可学习参数缩放平移
```

- `bngain/bnbias`:如果标准化"标准化过头"了,让网络能学着调回来。初始化 gain=1,bias=0(即恒等)。
- **为什么去掉 b1**:`hprebn = emb_flat@W1 + b1`,但下一步立刻减去 `bnmean`,
  `b1` 是常数会被完全减掉 → 加了也白加,直接删。

### 6.3 训练 / 推理不一致问题 + 滑动平均

BatchNorm 训练时用**当前 batch** 的均值方差,但推理时可能只有 1 个样本、算不出方差。
解决:训练过程中用滑动平均偷偷记住"整体"的统计量。

```python
# 训练循环里,每步更新(不参与梯度)
with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmean
    bnvar_running  = 0.999 * bnvar_running  + 0.001 * bnvar

# 推理/验证时改用 running 值,而不是当前 batch
bnraw = (hprebn - bnmean_running) / torch.sqrt(bnvar_running + 1e-5)
```

**动量 0.999**:每步只挪 0.1%,平滑地逼近全局统计量。太大跟不上分布,太小抖动大。

### 6.4 反向(化简后的解析式)

```python
dbngain = (dz1 * bnraw).sum(0, keepdim=True)
dbnbias = dz1.sum(0, keepdim=True)
dbnraw  = dz1 * bngain
n = batch_size
dhprebn = bngain * bnvar_inv / n * (
    n*dbnraw - dbnraw.sum(0) - n/(n-1) * bnraw * (dbnraw*bnraw).sum(0)
)
```

这一步不用死记,Karpathy 的推导思路是"逐节点展开再合并同类项"。
关键理解:**BatchNorm 让同一 batch 内的样本互相耦合**(因为共享均值方差),
所以 `dhprebn` 里出现了 `dbnraw.sum(0)` 这种跨样本的项——一个样本的梯度会受
同 batch 其他样本影响。这也是 BatchNorm 有轻微正则化效果的来源。

### 6.5 副作用

- **正则化**:batch 里的样本互相"污染"引入噪声,类似 dropout,能轻微抗过拟合。
- **代价**:batch 太小时统计量不稳;训练/推理行为不一致是常见 bug 源(务必用 running 值推理)。

---

## 7. 训练循环 & 优化技巧

```python
for i in range(iters):
    ix = torch.randint(0, X_train.shape[0], (batch_size,))  # mini-batch 采样
    ... 前向 ...
    ... 反向 ...
    cur_alpha = alpha if i < alpha_decay_start else alpha_decayed  # 学习率衰减
    W2 -= cur_alpha * dW2
    ...
```

### 7.1 Mini-batch(batch_size=32)

不用全量数据算梯度,每步随机抽 32 条。**优点**:快、能跳出局部极小;
**代价**:梯度有噪声(loss 曲线抖)。看趋势要 `lossi.view(-1,100).mean(1)` 做滑动平均。

### 7.2 学习率(alpha=0.1)

太大→震荡甚至发散;太小→收敛慢。找法:早期可以做 **lr sweep**
(从 1e-3 到 1 指数扫一遍,画 loss vs lr,取下降最快那段)。

### 7.3 学习率衰减(02-3 新增)

```python
alpha_decay_start = int(iters * 0.8)   # 后 20%
alpha_decayed = alpha / 10             # 0.1 → 0.01
```

**为什么**:训练后期 loss 在最优点附近震荡(步子太大反复越过),
降低学习率让它"精细着陆"。02-3 的 loss 曲线在 80% 处明显又下了一个台阶。

---

## 8. 诊断指标:训练完看什么、怎么改

| 指标 | 02-2 | 02-3 | 解读 & 改法 |
|---|---|---|---|
| **train / dev loss** | 2.2469 / 2.2580 | **2.2124 / 2.2276** | 差距 0.015 很小 → **欠拟合**,该加容量/加训练,不是加正则 |
| **update:data ratio** | W2 ≈ -1.7(偏高) | 同样 W2 偏高 | 理想 -3。W2 相对自身被更新过猛 → 给 W2 单独小 lr,或换 Adam |
| **tanh 饱和 \|a1\|>0.99** | 23% | **6%** | BatchNorm 生效。仍高就压 W1 增益 |
| **W2 正负计数** | ~49% 正 | ~49% 正 | 对称,健康 |
| **loss 曲线形状** | 收敛慢、噪声大 | 后期又降一截 | 加 batch_size / 加 lr 衰减 |

### 8.1 update:data ratio 深入

```python
ud_w1.append(torch.log10(alpha * dW1.std() / (W1.std()+1e-8)).item())
```

衡量"这一步把参数改变了百分之几"。理想 log10 值在 **-3**(每步动 0.1%)。
- 高于 -3(如 W2 的 -1.7)→ 更新太猛,可能震荡。
- 低于 -3 → 学得太慢,该加 lr。
- **注意**:这是比值,掩盖了是分子(梯度)还是分母(权重)的问题。建议额外分开记录 `grad.std()` 和 `weight.std()` 两条线。

### 8.2 过拟合 vs 欠拟合的判断树

```
train loss 高, dev loss 也高, 两者接近
   → 欠拟合:加容量(hidden_size↑ / embedding_dim↑ / block_size↑)、加训练步数、调 lr

train loss 低, dev loss 明显更高(差距拉大)
   → 过拟合:加正则(weight decay / dropout)、减小模型、加数据

两者都低且接近
   → 健康,可以停,或继续小步榨性能
```

**当前处于第一种(欠拟合)**,所以下一步优化方向是**加容量**,不是加正则。

---

## 9. 采样(用训练好的模型生成名字)

```python
@torch.no_grad()
def sample(n=10):
    for _ in range(n):
        out, context = [], [0]*block_size
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(bngain*((emb.reshape(1,-1)@W1 - bnmean_running)*bnvar_inv_running) + bnbias)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, 1).item()  # 按概率采样,不是取 argmax
            if ix == 0: break                          # 采到 '.' 就结束
            out.append(itos[ix]); context = context[1:]+[ix]
        print(''.join(out))
```

**要点**:
- 用 `multinomial` 按概率抽,不用 `argmax`(否则每次都生成同一个名字,没多样性)。
- 采样时 BatchNorm 也要用 `running` 统计量。
- **定性检查**:loss 数字好不代表生成质量好,一定要眼睛看几个采样结果,判断像不像真名字。

---

## 10. 训练时值得多收集的指标(盲区清单)

现有代码只记了 W1/W2/C 的 ud、a1 直方图、最终 train/dev loss。补充建议:

1. **dev loss 完整曲线**(每 N 步算一次)→ 看 train/dev 何时分叉,捕捉过拟合起点。
2. **bngain/bnbias 的 ud**→ 现在完全没监控 BatchNorm 自己的参数。
3. **grad.std() 与 weight.std() 分开记**→ 分清 ud 异常是哪一侧的锅。
4. **梯度 NaN/Inf 断言**:`assert torch.isfinite(dW1).all()`→ 爆炸第一时间报错。
5. **定期采样打印名字**(每 1~2k 步)→ 定性看学习进展。
6. **生成结果与训练集重复率**→ 检测"记忆"而非"泛化",模型变大后必看。
7. **perplexity = exp(loss)**→ 更直观、可横向比较。
8. **单步耗时**→ 网格搜索放大模型前先掂量时间成本。

---

## 11. 下一步优化路线图(按性价比排序)

1. **加容量**(当前欠拟合):hidden_size 200→300,embedding_dim 10→更高;或 block_size 3→更长。
2. **换 Adam**:自动抹平各层量级差异,直接解决 W2 的 ud 偏高问题。
3. **网格搜索超参数**(第一个 cell 的 TODO 还没做):lr × hidden_size × embedding_dim,用 dev loss 选最优。
4. **正则化**(等加容量后 dev 开始落后于 train 再上):weight decay 或 dropout。
5. **最后**才碰 test 集,报一次最终分数。

---

## 一句话总结

- **数据**:滑动窗口造样本,`.` 兼任起止符,三段划分且 test 只碰一次。
- **初始化**:输出层压小(初始 loss ≈ 3.3),隐藏层 Kaiming+tanh增益(激活方差≈1)。
- **前向**:softmax 减最大值防溢出;BatchNorm 拉平分布防 tanh 饱和。
- **反向**:`dlogits = (probs - onehot)/m`,`tanh'=1-a1²`,autograd 对拍验证。
- **训练**:mini-batch + lr 衰减;看 ud≈-3、饱和率、train/dev 差距三大指标。
- **调优**:欠拟合就加容量,过拟合才加正则;换 Adam 治 ud 不均;定性看采样。
