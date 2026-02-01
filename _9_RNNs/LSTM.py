#LSTM,类似于GRU，实际上LSTM进化出来的GRU，但比GRU更复杂

import torch
import torch.nn as nn
from _8_RNN import get_vocab

vocab = get_vocab()

#从头实现LSTM
class LSTM_layer(nn.Module):
    """
        一个简单的LSTM_layer
    """
    def __init__(
            self,
            vocab,
            input_size,
            num_hiddens,
            device
    ):
        super(LSTM_layer,self).__init__()
        self.device = device
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.input_size = input_size
        self.num_hiddens = num_hiddens

    def get_parameter(self) -> "Parameter":
        # 输入门 I 参数：Wxi,Whi,bi
        # 遗忘门 F 参数：Wxf,Whf,bf
        # 输出门 O 参数：Wxo,Who,bo
        # 候选记忆门 C_ 参数：Wxc,Whc,bc
        input_size = self.input_size
        num_hiddens = self.num_hiddens

        device = self.device
        def three():
            return (
                (torch.randn((input_size, num_hiddens), device=device) * 0.01),
                (torch.randn((num_hiddens, num_hiddens), device=device) * 0.01),
                (torch.zeros((num_hiddens), device=device) * 0.01)
            )

        # 初始化参数
        Wxi, Whi, bi = three()
        Wxf, Whf, bf = three()
        Wxo, Who, bo = three()
        Wxc, Whc, bc = three()

        Parameter = [Wxi, Whi, bi, Wxf, Whf, bf, Wxo, Who, bo, Wxc, Whc, bc]

        #设置梯度
        for param in Parameter:
            param.requires_grad_(True)

        return Parameter

    def forward(self,x):
        # x的形状应该是 batch_size * seq_len * vocab_size(或者embed_size)

        batch_size,seq_len,_ = x.shape

        #初始化H0
        H = torch.zeros((batch_size,self.num_hiddens),device=self.device)
        #初始化记忆元C
        C = torch.zeros((batch_size,self.num_hiddens),device=self.device)
        [Wxi, Whi, bi, Wxf, Whf, bf, Wxo, Who, bo, Wxc, Whc, bc] = self.get_parameter()
        outputs = []
        for T in range(seq_len):
            x_T = x[:,T,:]

            I = torch.sigmoid(x_T @ Wxi + H @ Whi + bi) #输入门I
            F = torch.sigmoid(x_T @ Wxf + H @ Whf + bf) #遗忘门F
            O = torch.sigmoid(x_T @ Wxo + H @ Who + bo) #输出门O
            C_ = torch.tanh(x_T @ Wxc + H @ Whc + bc) #候选记忆门 C_

            C = F * C + I * C_ #更新记忆元
            H = O * torch.tanh(C) #更新H
            outputs.append(H) #储存每一步的H
        outputs = torch.stack(outputs, dim=1)
        return outputs,(H,)

class LSTM_model(nn.Module):
    def __init__(
            self,
            vocab,
            device,
            embed_size
    ):
        super(LSTM_model,self).__init__()
        self.embed_size = embed_size
        self.vocab_size = len(vocab)

        self.embedding = nn.Embedding(len(vocab),embed_size)

        self.layer1 = LSTM_layer(
            vocab=vocab,
            device=device,
            input_size=self.embed_size,
            num_hiddens=256,
        )

        self.layer2 = LSTM_layer(
            vocab=vocab,
            device=device,
            input_size = 256,
            num_hiddens=128,
        )

        self.layer3 = LSTM_layer(
            vocab=vocab,
            device=device,
            input_size = 128,
            num_hiddens=64,
        )

        self.dense = nn.Linear(in_features=64,out_features=self.vocab_size) #稠密层输出，无需自定义输出层参数，原LSTM输出层就相当于一个稠密层

    def forward(self,x):
        #要求x 形状为 batch_size * seq_len

        embeded = self.embedding(x)

        x1, _ = self.layer1(embeded)
        x1 = torch.relu(x1)

        x2, _ = self.layer2(x1)
        x2 = torch.relu(x2)

        x3, (h,) = self.layer3(x2)
        x3 = torch.relu(x3)

        output = self.dense(x3)
        output = nn.functional.softmax(output, dim=-1)

        return output, (h,)

if __name__ == '__main__':
    batch_size = 32
    num_hiddens = 256
    vocab = get_vocab()  # 导入字典
    seq_len = 10
    vocab_size = len(vocab)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 生成测试数据
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    model = LSTM_model(
        vocab=vocab,
        device='cpu',
        embed_size=100
    )

    outputs, (h,) = model(x)

    print(outputs.shape)  # 预期结果 batch_size * seq_len * vocab_size
    print(h.shape)  # 预期结果 batch_size * 64(layer3的num_hiddens)
    # 现在该模型可以用于正常的训练了，同时也可以使用LSTMlayer来自定义新的模型










