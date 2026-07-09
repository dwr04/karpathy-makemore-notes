"""
classlist.py

封装的神经网络基础层：Linear, BatchNorm1d, Tanh
以及一个根据后接非线性动态决定初始化gain的辅助函数 calculate_gain。

用法：
    from classlist import Linear, BatchNorm1d, Tanh, calculate_gain
"""

import torch


# ============================================================
# gain 查表：根据"这一层后面接的是什么非线性"决定初始化的gain
# 对应 torch.nn.init.calculate_gain 的思路
# ============================================================

def calculate_gain(nonlinearity):
    gains = {
        'linear': 1.0,
        'sigmoid': 1.0,
        'tanh': 5.0 / 3.0,
        'relu': 2.0 ** 0.5,
        'leaky_relu': (2.0 / (1 + 0.01 ** 2)) ** 0.5,
    }
    if nonlinearity not in gains:
        raise ValueError(f"未知的nonlinearity: {nonlinearity}，可选: {list(gains.keys())}")
    return gains[nonlinearity]


# ============================================================
# Linear
# ============================================================

class Linear:
    def __init__(self, fan_in, fan_out, bias=True, nonlinearity='linear', generator=None):
        """
        nonlinearity: 这一层输出之后接的是什么激活函数（'tanh'/'relu'/'linear'等），
                      用来决定初始化的gain。
        generator:    可选，不传就用全局默认随机状态；
                      传入固定种子的Generator可以保证可复现。
        """
        gain = calculate_gain(nonlinearity)
        if generator is not None:
            self.weight = torch.randn((fan_in, fan_out), generator=generator) * gain / fan_in ** 0.5
        else:
            self.weight = torch.randn((fan_in, fan_out)) * gain / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])


# ============================================================
# BatchNorm1d
# ============================================================

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.001):
        self.eps = eps
        self.momentum = momentum
        self.training = True   # True: 用batch统计量；False: 用running统计量（推理/评估时切换）

        # 可学习参数
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # running统计量，不参与梯度
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):

        if self.training:
            if x.ndim==2:
                dim=0
            elif x.ndim==3:
                dim=(0,1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True, unbiased=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        # 只在训练模式下更新running统计量，且不进入计算图
        if self.training:
            with torch.no_grad():
                # xmean/xvar 在3D输入时 keepdim=True 会带多余的singleton维度（如(1,1,dim)），
                # 用 view(-1) 压回 (dim,)，避免 running_mean/var 的 shape 被广播"带偏"
                xmean_flat = xmean.view(-1)
                xvar_flat = xvar.view(-1)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean_flat
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * xvar_flat

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


# ============================================================
# Tanh
# ============================================================

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []
    
class FlattenConsecutive:
    def __init__(self,n):
        self.n=n

    def __call__(self,x):
        B,T,C=x.shape
        x=x.view(B,T//self.n,C*self.n)
        if x.shape[1]==1:
            x=x.squeeze(1)
        self.out=x
        return self.out

    def parameters(self):
        return []
    
class Embedding:
    
    def __init__(self, num_embeddings, embedding_dim, generator=None):
        if generator is not None:
            self.weight = torch.randn((num_embeddings, embedding_dim), generator=generator)
        else:
            self.weight = torch.randn((num_embeddings, embedding_dim))
        
    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]
    
class Sequential:
    
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        # 把所有子层的参数列表拼接成一个大列表
        return [p for layer in self.layers for p in layer.parameters()]