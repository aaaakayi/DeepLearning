# 多头注意力机制
# 将同一个q,k,v进行不同的线性变换后进行注意力汇聚，将这些注意力合并在一起然后经过MLP得到更复杂的表示
# 数学表达 H_i = f(W_qi,W_ki,W_vi,q,k,v) 最后得到W0 * ΣH
# q,k,v形状： batch_size , seq_len , features_size
# 这里就是把原本的特征（假设features_size = 12 ,num_head = 3)则每个头均分12/3 = 4个特征
import torch
import torch.nn as nn
from  C10_3 import DotProductAttention,mask_softmax
from C10_1 import show_multiple_heatmaps
#实现多头注意力机制
class MultiHeadAttention(nn.Module):
    #简化计算成本，
    def __init__(self,
                 num_heads,
                 dropout,
                 num_hiddens,
                 q_size,
                 k_size,
                 v_size,
                 bias=False
        ):
        super(MultiHeadAttention,self).__init__()
        self.num_heads = num_heads
        self.attention =  DotProductAttention(dropout) # 注意力汇聚函数

        #需要投影到一个维度
        self.Wq = nn.Linear(q_size,num_hiddens,bias=bias)
        self.Wk = nn.Linear(k_size,num_hiddens,bias=bias)
        self.Wv = nn.Linear(v_size,num_hiddens,bias=bias)

        self.Wo = nn.Linear(num_hiddens,num_hiddens,bias=bias)


    def forward(self,q,k,v,valid_len=None,mask=None):
        # 首先要把数据改成 batch_size*num_heads , seq_len , num_hiddens/num_heads
        # 经过线性变化再拆分，然后进行注意力汇聚
        q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)

        q = Unpack_base_head(q,self.num_heads)
        k = Unpack_base_head(k, self.num_heads)
        v = Unpack_base_head(v, self.num_heads)

        # 此时q,k,v形状变成 batch_size * num_heads,seq_len,num_hiddens/num_heads
        # valid_len需要匹配成相应的形状
        if valid_len is not None:
            valid_len = torch.repeat_interleave(valid_len,repeats=self.num_heads, dim=0)

        #注意力汇聚
        output,self.attention_weights = self.attention(
            queries = q,
            keys = k,
            values = v,
            mask = mask,
            valid_lens = valid_len
        )

        output = Merge_base_head(output,self.num_heads)
        output = self.Wo(output)

        #流程：q,k,v -> 线性变化投射到同一维度 -> 根据head拆分 -> valid_len(如果需要)根据拆分后的情况调整 -> 注意力汇聚 -> 根据head合并输出 -> 输出经过线性变换输出
        return output,self.attention_weights


def Unpack_base_head(x,num_heads):
    #根据头进行拆分
    x = x.reshape(x.shape[0],x.shape[1],num_heads,-1) # shape -> batch_size,seq_len,num_heads,features_size/num_heads
    x = x.permute(0,2,1,3) #shape -> batch_size,num_heads,seq_len,features_size/num_heads
    return x.reshape(-1,x.shape[2],x.shape[3]) #shape -> batch_size * num_heads,seq_len,features_size/num_heads

def Merge_base_head(x,num_heads):
    #根据头进行合并，此时根据线性变化，x shape -> batch_size * num_heads,seq_len,num_hiddens/num_heads
    x = x.reshape(-1,num_heads,x.shape[1],x.shape[2]) # shape -> batch_size,num_heads,seq_len,num_hiddens
    x = x.permute(0,2,3,1) # shape -> batch_size,seq_len,num_hiddens,num_heads
    return x.reshape(x.shape[0],x.shape[1],-1) #shape -> batch_size,seq_len,num_hiddens * num_heads


if __name__ == "__main__":
    q = torch.randn((2,4,10),device='cpu')
    k = torch.randn((2,10,20),device='cpu')
    v = torch.randn((2,10,20),device='cpu')
    valid_len = torch.tensor((2,3))
    multi_head = MultiHeadAttention(
        num_heads = 5,
        dropout = 0.02,
        num_hiddens = 50,
        q_size = q.shape[2],
        k_size = k.shape[2],
        v_size = v.shape[2],
        bias=False
    )

    multi_head.eval()

    with torch.no_grad():
        output,attention_weights = multi_head(
            q, k, v, valid_len
        )

    print(output.shape) # shape : batch_size * q_seq_len * num_hiddens: 2,4,50
    print(attention_weights.shape) # shape: batch_size*num_heads , q_seq_len , k_seq_len
    attention_weights_list = []
    list = (0,1,2,5,6,7)
    for num in list:
        attention_weights_list.append(
            (attention_weights[num],f'Sample {num}', True, 'Reds')
        )

    show_multiple_heatmaps(attention_weights_list)
