# 本节主要介绍注意力评分函数
# q:查询 k:键 v:值
# f(q,(k,v)) = Σα(k,q)v
# α(k,q) = softmax(a(k,q)) 其中α为注意力评分函数
# 将注意力汇聚的输出计算可以作为值的加权平均，选择不同的注意力评分函数会带来不同的注意力汇聚操作。
# 当查询和键是不同长度的矢量时，可以使用可加性注意力评分函数。当它们的长度相同时，使用缩放的“点－积”注意力评分函数的计算效率更高。

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from  C10_1 import show_heatmap,show_multiple_heatmaps

def mask_softmax(x,valid_lens,mask=None):
    """

    :param x: 注意力分数张量，形状为 [batch_size, seq_len, seq_len]
    :param valid_lens: 有效范围
    :param mask: 掩蔽矩阵
    :return: 裁剪后进行softmax的值
    """
    batch_size, seq_len, _ = x.shape

    # 步骤1: 创建掩码矩阵
    # 使用torch.arange创建位置索引 [0, 1, 2, 3, ...]
    positions = torch.arange(seq_len, device=x.device)  # [seq_len]

    # 将valid_lens扩展为适合广播的形状
    # 原始: [batch_size] -> 扩展为: [batch_size, 1, seq_len]
    valid_lens_expanded = valid_lens.view(-1, 1, 1)  # [batch_size, 1, 1]
    positions_expanded = positions.view(1, 1, -1)  # [1, 1, seq_len]

    # 步骤2: 生成掩码 (True表示保留，False表示屏蔽)
    # 比较位置索引和有效长度，创建布尔掩码
    mask = positions_expanded < valid_lens_expanded  # [batch_size, 1, seq_len]

    # 扩展掩码维度以匹配x的形状 [batch_size, seq_len, seq_len]
    mask = mask.expand(-1, seq_len, -1)

    # 步骤3: 应用掩码到注意力分数
    # 将被屏蔽的位置填充为负无穷
    x_masked = x.masked_fill(~mask, -1e9)  # 使用 -1e9 近似负无穷

    # 步骤4: 应用softmax
    result = F.softmax(x_masked, dim=-1)

    return result

class additive_attention(nn.Module):
    """
    加性注意力 注意力评分函数为a(k,q) = W_v^T * tanh(W_q @ q + W_k @ k) 适用于查询和键的长度不同的情况
    :param W_v,W_q,W_k
    :param q:查询 形状是batch_size * seq_len * feature_size
    :param k:键 形状是batch_size * seq_len * feature_size
    :return: 加权后的查询-值的概率对，相关系数矩阵
    """
    def __init__(self,q_feature_size,k_feature_size,num_hiddens,dropout,**kwargs):
        super(additive_attention,self).__init__(**kwargs)
        #初始化参数
        self.W_q = nn.Linear(in_features=q_feature_size,out_features=num_hiddens,bias=False)
        self.W_k = nn.Linear(in_features=k_feature_size,out_features=num_hiddens,bias=False)
        self.W_v = nn.Linear(in_features=num_hiddens,out_features=1,bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,queries, keys, values, valid_lens,mask_softmax):
        #将查询与键的特征维度通过稠密层投射到同一维度
        queries = self.W_q(queries)
        keys = self.W_k(keys)

        #通过广播机制将每个查询与每个键值对相加，得到batch_size * 查询个数 * 键值对个数 * num_hiddens
        features = torch.tanh(queries.unsqueeze(2) + keys.unsqueeze(1))

        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = mask_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values) , self.attention_weights
class DotProductAttention(nn.Module):
    """
    缩放点积注意力 评分函数a(k,q) = (q^T @ k)/sqrt(d)，d: 查询和键的长度，适用于查询和键长度相同的情况
    :param
    :return
    """
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, mask_softmax, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = mask_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values),self.attention_weights


if __name__ == '__main__':
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.randn((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])

    batch_size, q_size, q_feature_size = queries.shape
    _,k_size,k_feature_size = keys.shape
    num_hiddens = 20
    figure_data_list = []

    #----------------------------------------------------------------
    #-----------------------加性注意力---------------------------------
    additive_attention_model =  additive_attention(
        q_feature_size = q_feature_size,
        k_feature_size = k_feature_size,
        num_hiddens = num_hiddens,
        dropout = 0.1
    )

    additive_attention_model.eval()

    # 前向传播
    with torch.no_grad():  # 不需要计算梯度
        output, attention_weights = additive_attention_model(
            queries=queries,
            keys=keys,
            values=values,
            valid_lens=valid_lens,
            mask_softmax = mask_softmax
        )

    print("输出形状:", output.shape)
    print("注意力权重形状:", attention_weights.shape)
    print(attention_weights)

    figure_data_list.append((attention_weights[0],'additive batch0',True,'Reds'))
    figure_data_list.append((attention_weights[1], 'additive batch1', True, 'Reds'))
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # -----------------------缩放点积注意力------------------------------
    queries = torch.normal(0, 1, (2, 1, 2)) #此处修改q的特征长度和k的相同

    DotProductAttention_model = DotProductAttention(
        dropout=0.2
    )

    DotProductAttention_model.eval()

    with torch.no_grad():
        DotProductAttention_model(
            queries=queries,
            keys=keys,
            values=values,
            valid_lens=valid_lens,
            mask_softmax=mask_softmax
        )

    print("输出形状:", output.shape)
    print("注意力权重形状:", attention_weights.shape)
    print(attention_weights)

    # 展示相关性热力图
    figure_data_list.append((attention_weights[0], 'DotProduct batch0', True, 'Reds'))
    figure_data_list.append((attention_weights[1], 'DotProduct batch1', True, 'Reds'))
    show_multiple_heatmaps(figure_data_list,(10,6))
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------









