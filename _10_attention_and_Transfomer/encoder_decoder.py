#从头实现transfomer
import torch
import torch.nn as nn
from C10_5 import MultiHeadAttention
from C10_6 import position_ecoder
from C10_1 import show_heatmap,show_multiple_heatmaps

#1.基于位置的前馈网络 FNN
class FFN(nn.Module):
    """
    基于位置的前馈网络(Feed-Forward Network,FFN)
    目的在于将自注意力输出的特征进行进一步处理, 增加模型的表达能力, 提供额外的非线性

    """
    def __init__(self,input,num_hiddens,output,**kwargs):
        super(FFN,self).__init__(**kwargs)
        #一般来说,input=output,而num_hiddens应该比他们大(例如4倍)
        self.layer1 = nn.Linear(in_features=input,out_features=num_hiddens)
        self.layer2 = nn.Linear(in_features=num_hiddens,out_features=output)

    def forward(self,x):
        return self.layer2(torch.relu(self.layer1(x)))


#2.残差连接和层规范化(add&norm)
class add_norm(nn.Module):
    """
        残差连接和层规范化(add & norm)
        解决梯度消失/爆炸问题，使深层网络可训练
        稳定训练过程，加速收敛
        实现稳定高效的深度网络训练
    """
    def __init__(self,normalized_shape,dropout,**kwargs):
        super(add_norm,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self,x,y):
        return self.ln(self.dropout(y)+x) #x,y形状应该一致


#3.transfomer encoder
class transfomer_cell(nn.Module):
    """
    x -> MultiHeadAttention -> y -> add&norm(dropout(y)+x) -> x' -> FFN -> y' -> add&norm(dropout(y')+x') -> x"
      -> next transfomer_cell
    """
    def __init__(self,
                 num_heads,dropout,d_model,
                 **kwargs
    ):
        super(transfomer_cell,self).__init__(**kwargs)
        #初始化多头注意力模型,一般来说q_size = k_size = v_size = num_hiddens = d_model
        self.multiheadAttention = MultiHeadAttention(
            num_heads,
            dropout,
            num_hiddens = d_model,
            q_size = d_model,
            k_size = d_model,
            v_size = d_model,
            bias = False
        )
        #初始化FFN
        self.ffn = FFN(input = d_model,num_hiddens = 4*d_model,output = d_model) #这里直接采用经典的涉及方式,简化参数
        #初始化add & norm
        self.addnorm = add_norm(normalized_shape = d_model,dropout = dropout)

    def forward(self,q, k, v, valid_len=None, mask=None):
        residual1 = q
        y,self.attention_weights = self.multiheadAttention(q,k,v,valid_len = valid_len, mask = mask) #编码过程不需要掩码
        x2 = self.addnorm(x=residual1,y=y)

        residual2 = x2
        y2 = self.ffn(x2)
        output = self.addnorm(x=residual2,y=y2)
        return output,self.attention_weights

class transfomer_encoder(nn.Module):
    """
    基于transfomer_cell构建transfomer_encoder
    """
    def __init__(self,
                 num_layers,vocab_size,
                 num_heads, dropout, d_model,
                 **kwargs
    ):
        super(transfomer_encoder,self).__init__(**kwargs)
        self.sequentia = nn.Sequential()
        self.embeded = nn.Embedding(vocab_size,d_model)
        self.pos_encoding = position_ecoder(dropout,num_hiddens=d_model,max_len=1000)
        for i in range(num_layers):
            self.sequentia.add_module(
                "block" + str(i),
                transfomer_cell(num_heads,dropout,d_model)
            )
    def forward(self,x,valid_len=None,mask=None):
        x = self.embeded(x)
        x = self.pos_encoding(x)
        self.attention_weight = []
        i = 1
        for blk in self.sequentia:
            x,attention_weight = blk(x,x,x,valid_len=valid_len, mask=mask)
            self.attention_weight.append(
                (attention_weight,f"enocde block{i}",True,"Reds")
            )
            i += 1

        return x,self.attention_weight

class transfomer_decoder_cell(nn.Module):
    """
    transfomer decoder cell:
    包含解码器自注意力、“编码器-解码器”注意力和基于位置的前馈网络
    cell部分依旧可用，此时cell的输入变成了 解码器自注意力的输出作为q,编码器的输出作为k,v
    流程如下：
    输入x -> 掩码自注意力 -> add&norm -> 编码器-解码器注意力（其中k和v来自编码器输出，q来自上一个add&norm的输出） -> add&norm -> FFN -> add&norm -> 输出
    """
    def __init__(
            self,
            num_heads,
            dropout,
            d_model,
            **kwargs
    ):
        super(transfomer_decoder_cell,self).__init__(**kwargs)
        self.multiattention1 = MultiHeadAttention(
            num_heads = num_heads,
            dropout = dropout,
            num_hiddens = d_model,
            q_size = d_model,
            k_size = d_model,
            v_size = d_model,
            bias=False
        ) #编码器自注意力模块

        self.add_norm1 = add_norm(
            normalized_shape = d_model,
            dropout = dropout
        )

        self.cell = transfomer_cell(
            num_heads = num_heads,
            dropout = dropout,
            d_model = d_model,
        ) #cell 依旧可用，仅修改q为解码器自注意力输出,k,v为encoder输出,维持d_model一致

    def forward(self,decoder_input, encoder_output,
                mask=None,valid_len=None, #decoder输入序列的掩蔽方式
                encoder_mask=None, encoder_valid_len=None #原encoder序列的掩蔽方式
        ):
        # 实际上 q为编码器输出 k=v=解码器自注意力输出
        residual1 = decoder_input
        y1,self.attention_weights = self.multiattention1(decoder_input,decoder_input,decoder_input,valid_len=valid_len,mask=mask)
        x2 = self.add_norm1(x=residual1,y=y1) #编码器自注意力输出

        # --- 第2步：编码器-解码器注意力 + FFN ---
        # 使用复用的transformer_cell
        # q = x2 (解码器自注意力输出)
        # k = v = encoder_output (编码器输出)
        output, self.cross_attention_weights = self.cell(
            q = x2, k = encoder_output, v = encoder_output, valid_len = encoder_valid_len, mask = encoder_mask
        )

        return output, self.attention_weights, self.cross_attention_weights


class transfomer_decoder(nn.Module):
    """
    基于transformer decoder cell 构建的 transfomer decoder
    """
    def __init__(
            self,
            vocab_size,
            num_layers,
            num_heads, dropout, d_model,
            **kwargs
    ):
        super(transfomer_decoder,self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pos_encoding = position_ecoder(dropout, num_hiddens=d_model, max_len=1000)
        self.sequential = nn.Sequential()
        for i in range(num_layers):
            self.sequential.add_module(
                "block" + str(i),
                transfomer_decoder_cell(num_heads,dropout,d_model)
            )

        # self.output_layer = nn.Linear(d_model, vocab_size),解码器中不需要输出拓展成vocab_size

    def forward(self,
                decoder_input, encoder_output,
                tgt_mask=None, tgt_valid_len=None, # 解码器掩码
                memory_mask=None, memory_valid_len=None # 原编码器掩码
        ):
        self.attention_weights = []
        self.cross_attention_weights = []
        q = self.embedding(decoder_input)
        q = self.pos_encoding(q)
        i = 1
        for blk in self.sequential:
            q, attention_weights, cross_attention_weights = blk(
                decoder_input=q, encoder_output=encoder_output,
                mask=tgt_mask, valid_len=tgt_valid_len,
                encoder_mask=memory_mask, encoder_valid_len=memory_valid_len
            )
            self.attention_weights.append(
                (attention_weights,f"decoder block{i}",True,"Reds")
            )
            self.cross_attention_weights.append(
                (cross_attention_weights, f"decoder block{i}", True, "Reds")
            )
            i += 1
        output = q
        return output,self.attention_weights,self.cross_attention_weights


def create_masks(src_seq=None, tgt_seq=None, pad_token_id=0, device='cpu'):
    """
    创建Transformer所需的所有掩码

    参数:
    src_seq: 源序列，形状为 (batch_size, src_len)
    tgt_seq: 目标序列，形状为 (batch_size, tgt_len)
    pad_token_id: 填充标记ID

    返回:
    src_mask: 源序列填充掩码，用于编码器，形状为 (batch_size, 1, 1, src_len)
    tgt_mask: 目标序列掩码（前瞻+填充），用于解码器自注意力，形状为 (batch_size, 1, tgt_len, tgt_len)
    memory_mask: 编码器输出掩码，用于解码器交叉注意力，形状为 (batch_size, 1, 1, src_len)
    src_valid_len: 源序列有效长度，形状为 (batch_size,)
    tgt_valid_len: 目标序列有效长度，形状为 (batch_size,)

    支持单独更新tgt_seq的mask,能够满足推理工作的需求
    """
    result = {
        'src_mask': None,
        'tgt_mask': None,
        'memory_mask': None,
        'src_valid_len': None,
        'tgt_valid_len': None
    }

    if src_seq is not None:
        batch_size, src_len = src_seq.shape

        # 创建源序列填充掩码（用于编码器）
        src_mask = (src_seq == pad_token_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,src_len]

        # 编码器输出掩码（用于解码器交叉注意力）
        memory_mask = src_mask.clone()

        # 源序列有效长度
        src_valid_len = (src_seq != pad_token_id).sum(dim=1)  # [B]

        # 更新结果
        result['src_mask'] = src_mask
        result['memory_mask'] = memory_mask
        result['src_valid_len'] = src_valid_len

    if tgt_seq is not None:
        batch_size, tgt_len = tgt_seq.shape

        # 创建目标序列填充掩码
        tgt_padding_mask = (tgt_seq == pad_token_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,tgt_len]

        # 创建前瞻掩码（上三角矩阵）
        look_ahead_mask = torch.triu(
            torch.ones((tgt_len, tgt_len), device=device),
            diagonal=1
        ).bool().unsqueeze(0).unsqueeze(0)  # [1,1,tgt_len,tgt_len]

        # 扩展到批次维度
        look_ahead_mask = look_ahead_mask.expand(batch_size, -1, -1, -1)  # [B,1,tgt_len,tgt_len]

        # 扩展填充掩码到注意力矩阵形状
        tgt_padding_mask_expanded = tgt_padding_mask.expand(-1, -1, tgt_len, -1)  # [B,1,tgt_len,tgt_len]

        # 合并目标序列掩码（填充+前瞻）
        tgt_mask = tgt_padding_mask_expanded | look_ahead_mask

        # 目标序列有效长度
        tgt_valid_len = (tgt_seq != pad_token_id).sum(dim=1)  # [B]

        # 更新结果
        result['tgt_mask'] = tgt_mask
        result['tgt_valid_len'] = tgt_valid_len

    return result


def Pad(x, seq_len, pad_token_id=0):
    """
    将序列填充或截断到指定长度

    参数:
    x: 输入序列，形状为 (batch_size, current_len)
    seq_len: 目标序列长度
    pad_token_id: 填充标记的ID（默认0）

    返回:
    填充或截断后的序列，形状为 (batch_size, seq_len)
    """
    # 输入验证
    if seq_len <= 0:
        raise ValueError(f"seq_len必须是正整数，当前为{seq_len}")

    if x.dim() != 2:
        raise ValueError(f"输入x必须是2维张量，当前维度为{x.dim()}")

    batch_size, x_len = x.shape

    if x_len == seq_len:
        return x
    elif x_len > seq_len:
        # 截断到前seq_len个token
        return x[:, :seq_len].clone()
    else:
        pad_seq = torch.full(
            size=(batch_size, seq_len - x_len),
            fill_value=pad_token_id,
            dtype=x.dtype,  # 保持数据类型一致
            device=x.device  # 保持设备一致
        )
        return torch.cat((x, pad_seq), dim=-1)

def add_special_tokens(seq, bos_token_id=1, eos_token_id=2):
    """添加特殊标记"""
    # 添加<bos>和<eos>
    return torch.cat([
        torch.full((seq.size(0), 1), bos_token_id, device=seq.device),
        seq,
        torch.full((seq.size(0), 1), eos_token_id, device=seq.device)
    ], dim=1)

def test_encoder_decoder():
    """
        标准流程：原序列，目标序列根据seq_len填充 -> 生成掩码 -> encoder -> decoder
    """
    num_layers = 6
    vocab_size = 100000
    num_heads = 64
    dropout = 0.2
    d_model = 256 # d_model 需要能够被num_heads整除
    batch_size = 1
    seq_len = 64 # 不足64用0填充
    pad_token_id = 0 # 用0填充

    x1 = torch.randint(0, vocab_size, (batch_size, 30)) #源序列   30<64 需要填充
    x2 = torch.randint(0, vocab_size, (batch_size, 84)) #目标序列 84>64 需要截断 在注意力权重矩阵上表现的很清楚,见test_result
    #填充
    x1 = Pad(x=x1,seq_len=seq_len,pad_token_id=pad_token_id)
    x2 = Pad(x=x2,seq_len=seq_len,pad_token_id=pad_token_id)

    #创建相关掩码
    masks = create_masks(x1, x2,pad_token_id = pad_token_id, device = 'cpu')

    #编码器
    encoder = transfomer_encoder(
        num_layers, vocab_size,
        num_heads, dropout, d_model,
    )

    encoder.eval()

    with torch.no_grad():
        output_encoder, encoder_attention_weights = encoder(x1,valid_len=masks['src_valid_len'],mask=masks['src_mask'])

    print("transfomer encoder output shape:",output_encoder.shape)

    #解码器
    decoder = transfomer_decoder(
        vocab_size,num_layers,
        num_heads, dropout, d_model,
    )

    decoder.eval()
    with torch.no_grad():
        output_decoder,decoder_attention_weights,cross_attention_weights = decoder(
            decoder_input = x2, encoder_output = output_encoder,
            tgt_mask=masks['tgt_mask'], tgt_valid_len=masks['tgt_valid_len'],  # 解码器掩码
            memory_mask=masks['memory_mask'], memory_valid_len=masks['src_valid_len']  # 原编码器掩码
        )
    print("transfomer decoder output shape:", output_decoder.shape)

    # shape:num_layers,(attention_weights,'name',grid,'color')
    # attention_weights shape: batch_size*num_heads , seq_len(q_size), seq_len(k_size)
    #show_multiple_heatmaps(data_list, figsize=(10, 6), layout=None,
                        # title=None, hide_ticks=False, filename=None):
    show_multiple_heatmaps(
        encoder_attention_weights,
        figsize=(8,6),
        title="encoder_attention_weights",
        filename="test_result/encoder_attention_weights",
        hide_ticks=True
    )
    show_multiple_heatmaps(
        decoder_attention_weights,
        figsize = (8, 6),
        title = "decoder_attention_weights",
        filename = "test_result/decoder_attention_weights",
        hide_ticks = True
    ) #可以看到上三角部分全为0,这是因为进行了掩蔽
    show_multiple_heatmaps(
        cross_attention_weights,
        figsize = (8, 6),
        title = "cross_attention_weights",
        filename = "test_result/cross_attention_weights",
        hide_ticks = True
    )


if __name__ == "__main__":
    test_encoder_decoder()










