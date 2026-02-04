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

import torch
import torch.nn.functional as F


def mask_softmax(x, valid_lens=None, mask=None):
    """
    实现Masked Softmax，支持各种形状的掩码

    参数:
    x: 注意力分数张量，形状为 [batch_size * num_heads, num_queries, num_keys]
        注意：在多头注意力中，x已经是重塑后的形状，batch维度已包含num_heads
    valid_lens: 有效长度，用于创建填充掩码，形状为 (batch_size,) 或 (batch_size, num_queries)
    mask: 外部传入的掩码，形状可以是：
          - 2D: (num_queries, num_keys)
          - 3D: (batch_size, num_queries, num_keys)
          - 4D: (batch_size, 1, 1, num_keys) 或 (batch_size, 1, num_queries, num_keys)

    返回:
    裁剪后进行softmax的值，形状与x相同
    """
    # x的形状：在MultiHeadAttention中，x已经是(batch_size * num_heads, num_queries, num_keys)
    # 我们需要通过mask的batch_size来推断num_heads
    batch_size_num_heads, num_queries, num_keys = x.shape

    # 处理外部传入的掩码
    if mask is not None:
        # 记录原始掩码形状
        original_mask_shape = mask.shape

        # 情况1：4维掩码 (从create_masks返回)
        if mask.dim() == 4:
            batch_size = mask.shape[0]  # 原始batch_size
            num_heads = batch_size_num_heads // batch_size

            # 确保能整除
            if batch_size_num_heads % batch_size != 0:
                raise ValueError(
                    f"掩码的batch_size={batch_size}无法整除x的batch_size={batch_size_num_heads}"
                )

            # 子情况1.1: (batch_size, 1, 1, num_keys) -> 用于编码器或编码器-解码器注意力
            if mask.shape[1] == 1 and mask.shape[2] == 1:
                # 扩展掩码：(batch_size, 1, 1, num_keys) -> (batch_size, num_heads, num_queries, num_keys)
                mask = mask.expand(-1, num_heads, num_queries, -1)
                # 重塑：(batch_size, num_heads, num_queries, num_keys) -> (batch_size * num_heads, num_queries, num_keys)
                mask = mask.reshape(batch_size_num_heads, num_queries, num_keys)

            # 子情况1.2: (batch_size, 1, num_queries, num_keys) -> 用于解码器自注意力
            elif mask.shape[1] == 1:
                # 扩展掩码：(batch_size, 1, num_queries, num_keys) -> (batch_size, num_heads, num_queries, num_keys)
                mask = mask.expand(-1, num_heads, -1, -1)
                # 重塑
                mask = mask.reshape(batch_size_num_heads, num_queries, num_keys)

            # 子情况1.3: (batch_size, num_queries, num_queries, num_keys) -> 不常见，但处理一下
            elif mask.shape[2] == num_queries:
                # 假设num_queries = num_keys
                mask = mask.squeeze(1) if mask.shape[1] == 1 else mask
                if mask.dim() == 4:
                    # 压缩到3D
                    mask = mask.mean(dim=1)  # 或者其他聚合方式
                mask = mask.expand(-1, num_heads, -1, -1).reshape(batch_size_num_heads, num_queries, num_keys)

        # 情况2：3维掩码 (batch_size, num_queries, num_keys)
        elif mask.dim() == 3:
            batch_size = mask.shape[0]
            num_heads = batch_size_num_heads // batch_size

            if batch_size_num_heads % batch_size == 0:
                # 扩展掩码到多头
                mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
                mask = mask.reshape(batch_size_num_heads, num_queries, num_keys)
            else:
                # 如果不能整除，假设mask已经是正确的形状
                if mask.shape != x.shape:
                    # 尝试广播
                    try:
                        mask = mask.expand_as(x)
                    except:
                        raise ValueError(
                            f"3D掩码形状{mask.shape}无法广播到x形状{x.shape}，且batch_size不匹配"
                        )

        # 情况3：2维掩码 (num_queries, num_keys)
        elif mask.dim() == 2:
            # 直接扩展到需要的形状
            mask = mask.unsqueeze(0).expand(batch_size_num_heads, -1, -1)

        else:
            raise ValueError(f"不支持的掩码维度: {mask.dim()}")

    # 初始化最终掩码
    final_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)

    # 添加外部掩码（如果有）
    if mask is not None:
        # 确保掩码形状匹配
        if mask.shape != x.shape:
            try:
                mask = mask.expand_as(x)
            except RuntimeError:
                raise ValueError(f"掩码形状{mask.shape}无法广播到x形状{x.shape}")

        # 注意：我们的掩码是True表示要屏蔽
        final_mask = final_mask | mask

    # 添加填充掩码（如果有valid_lens）
    if valid_lens is not None:
        # 确定batch_size
        if mask is not None and mask.dim() >= 3:
            # 从掩码推断batch_size
            batch_size = mask.shape[0] if mask.dim() == 3 else mask.shape[0]
        else:
            # 从x推断：假设x的第一维是batch_size * num_heads
            # 我们需要知道num_heads才能知道batch_size
            # 这里我们假设num_heads=1，或者使用有效长度推断
            batch_size = batch_size_num_heads

        # 确保valid_lens的形状正确
        if valid_lens.dim() == 1:
            # (batch_size,) -> (batch_size, 1, 1)
            valid_lens = valid_lens.view(-1, 1, 1)
            # 扩展到每个查询
            valid_lens = valid_lens.expand(-1, num_queries, 1)
        elif valid_lens.dim() == 2:
            # (batch_size, num_queries) -> (batch_size, num_queries, 1)
            valid_lens = valid_lens.unsqueeze(-1)

        # 创建位置索引
        positions = torch.arange(num_keys, device=x.device)
        positions = positions.view(1, 1, -1).expand(batch_size, num_queries, -1)

        # 创建填充掩码：位置 >= 有效长度的位置应该被屏蔽
        # 注意：valid_lens需要扩展到num_keys维度
        padding_mask = positions >= valid_lens.expand(-1, -1, num_keys)

        # 扩展填充掩码到多头（如果需要）
        if padding_mask.shape[0] < batch_size_num_heads:
            # 计算num_heads
            num_heads = batch_size_num_heads // padding_mask.shape[0]
            if batch_size_num_heads % padding_mask.shape[0] == 0:
                padding_mask = padding_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
                padding_mask = padding_mask.reshape(batch_size_num_heads, num_queries, num_keys)

        # 合并掩码
        final_mask = final_mask | padding_mask

    # 应用掩码：将被屏蔽的位置设置为负无穷
    x_masked = x.masked_fill(final_mask, -1e9)

    # 应用softmax
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

    def forward(self,queries, keys, values, valid_lens, mask):
        #将查询与键的特征维度通过稠密层投射到同一维度
        queries = self.W_q(queries)
        keys = self.W_k(keys)

        #通过广播机制将每个查询与每个键值对相加，得到batch_size * 查询个数 * 键值对个数 * num_hiddens
        features = torch.tanh(queries.unsqueeze(2) + keys.unsqueeze(1))

        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = mask_softmax(scores, valid_lens, mask)
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
    def forward(self, queries, keys, values, mask=None, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = mask_softmax(scores, valid_lens = valid_lens, mask = mask)
        return torch.bmm(self.dropout(self.attention_weights), values), self.attention_weights


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
            mask = None
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
            mask = None
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









