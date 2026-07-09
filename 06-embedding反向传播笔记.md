# Embedding 层反向传播:demb_flat → demb → dC

> makemore backprop ninja 系列,唯一不是标准矩阵求导的一步(涉及 reshape + lookup 的梯度回传)


---

## 前向传播回顾(具体形状示例)

设 `m=32`(batch size), `block_size=3`, `n_embd=10`, `vocab_size=27`

```python
Xb.shape         # (32, 3)      —— 32个样本,每个样本3个字符索引
emb = C[Xb]
emb.shape        # (32, 3, 10)  —— lookup,每个索引→10维向量

emb_flat = emb.view(32, 30)     # 展平:3个字符的embedding直接拼接(不是相加/平均)
emb_flat.shape   # (32, 30)
```

`emb_flat` 第 i 行结构:`[字符1的10维, 字符2的10维, 字符3的10维]` 拼成一条30维向量。

---

## 第一步:demb_flat → demb(reshape 的逆操作)

`.view()` 只是换一种方式看待同一堆数字,顺序和内容不变 → 反向传播直接reshape回去即可,**没有任何数学运算**:

```python
demb = demb_flat.view(32, 3, 10)   # 形状 = emb 的形状
```

**核心规则**:任何变量 X 的梯度 dX,形状永远和 X 本身相同。

---

## 第二步:demb → dC(lookup 操作的梯度,真正的难点)

### 关键对应关系

- `Xb[i, j]` = 第i个样本、第j个context位置用的**字符索引**(如 5)
- `demb[i, j]` = 对应位置的10维梯度,即"C矩阵第5行这次被使用后,应该往哪调整"

### 实现:for循环 + 累加

```python
dC = torch.zeros_like(C)   # (27, 10),先全0

for i in range(Xb.shape[0]):        # 遍历样本
    for j in range(Xb.shape[1]):    # 遍历context位置
        ix = Xb[i, j]
        dC[ix] += demb[i, j]        # 注意:+=,不是 =
```

向量化写法(等价,更快):
```python
dC = torch.zeros_like(C)
dC.index_add_(0, Xb.flatten(), demb.view(-1, demb.shape[-1]))
```

### 为什么必须用 `+=`(唯一易错点)

如果某个字符(如索引5)在整个batch里被用到7次(不同样本/不同context位置都出现过),说明 C 第5行在前向传播时被**读取了7次**,参与了7次不同计算,对loss都有贡献。

反向传播时,这7次各自产生的梯度必须**加总**,才是C第5行真正的总梯度。

- 用 `+=`(累加)→ 正确,7笔梯度贡献全部保留
- 用 `=`(覆盖)→ 错误,只留下最后一次的梯度,前面6次全部丢失

**类比**:C矩阵某一行是"账户余额",batch里多次用到该字符 = 多笔"存入交易",要的是总和,不是"只记最后一笔"。

---

## 验证

```python
dC_manual = torch.zeros_like(C)
for i in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        dC_manual[Xb[i,j]] += demb[i,j]

print(torch.allclose(dC_manual, C.grad))   # 应为 True
```

若对不上,最常见的bug:
1. 漏了 `+=`,写成了 `=`(覆盖而非累加)
2. 索引顺序写反,如 `Xb[j,i]` 而非 `Xb[i,j]`

---

## 一句话总结

- **demb_flat → demb**:纯reshape,形状对齐即可,无计算
- **demb → dC**:lookup的反向传播 = 把每次使用的梯度,按索引"送回"C矩阵对应行,**多次使用同一行 → 梯度必须累加**
