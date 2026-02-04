# self-attention
# q,k,v来源于同一个输入
import torch
import torch.nn as nn

from C10_5 import MultiHeadAttention
from C10_1 import show_heatmap,show_multiple_heatmaps

class position_ecoder(nn.Module):
    def __init__(self,dropout,num_hiddens,max_len=1000):
        super(position_ecoder,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1,max_len,num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)



if __name__ == "__main__":
    num_heads = 5
    dropout = 0.2
    num_hiddens = 100
    x = torch.randn((2,5,20),device='cpu') #q,k,v都是同一个东西
    valid_len = torch.tensor((2,3))
    _,_,features_size = x.shape
    model = MultiHeadAttention(
        num_heads = num_heads,
        dropout = dropout,
        num_hiddens = num_hiddens,
        q_size = features_size,
        k_size = features_size,
        v_size = features_size,
        bias=False
    )

    model.eval()

    with torch.no_grad():
        output,attention_weights = model(x,x,x,valid_len)

    print(output.shape)
    attention_weights_list = []
    list = (0, 1, 2, 5, 6, 7)
    for num in list:
        attention_weights_list.append(
            (attention_weights[num], f'Sample {num}', True, 'Reds')
        )

    show_multiple_heatmaps(attention_weights_list)
    position = position_ecoder(
        dropout=dropout,
        num_hiddens=num_hiddens,
    )
    output_position = position(output)
    print((output_position - output)[0].shape)


