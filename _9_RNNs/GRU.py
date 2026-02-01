#GRU 有重置门(R) 更新门(z) 候选隐状态
#重置门 更新门和普通的隐状态计算没什么区别 (sigmoid激活)
#候选隐状态等于 更新门逐元素乘以上一步隐状态 然后计算 (tanh激活)
#最终隐状态 Hadamard(H(t-1),Z)+Hadamard(候选隐状态,Z) 其中Hadamard是逐元素相乘的意思

import torch #这一节开始换成pytorch编写代码
import torch.nn as nn

#手动实现GRU模块
from _8_RNN import get_vocab


def get_params(vocab_size,num_hiddens,device):
    #初始化参数,输入的x形状为seq_n * vocab_size
    #重置门(R) 参数 Wxr Whr br
    #更新门(z) 参数 Wxz Whz bz
    #候选隐状态 参数 Wxh Whh bh
    #最终隐状态无需定义新参数

    input_size = output_size = vocab_size

    def normal(shape):
        return torch.randn(size=shape,device=device)*0.01

    def three():
        return (
            normal((input_size,num_hiddens)),
            normal((num_hiddens,num_hiddens)),
            torch.zeros((num_hiddens),device=device)
        )

    Wxr,Whr,br = three()
    Wxz, Whz, bz = three()
    Wxh, Whh, bh = three()

    #输出层参数
    Wx = normal((num_hiddens,output_size))
    bx = torch.zeros((output_size),device=device)

    params = [Wxr,Whr,br,Wxz, Whz,bz,Wxh, Whh,bh,Wx,bx]

    for param in params:
        param.requires_grad_(True)

    return params

def get_gru_init(batch_size,num_hiddens,device):
    #初始化H0
    return torch.zeros((batch_size,num_hiddens),device=device)


def forward(x, batch_size, vocab_size, num_hiddens, device):
    params = get_params(vocab_size, num_hiddens, device)
    Wxr, Whr, br, Wxz, Whz, bz, Wxh, Whh, bh, Wx, bx = params

    # 初始化隐状态
    H = get_gru_init(batch_size, num_hiddens,device=device)
    outputs = []

    # 按时间步循环
    for t in range(x.shape[1]):
        # 当前时间步的输入，形状为(batch_size, vocab_size)
        X_t = x[:, t, :]
        # 重置门
        R = torch.sigmoid(X_t @ Wxr + H @ Whr + br)
        # 更新门
        Z = torch.sigmoid(X_t @ Wxz + H @ Whz + bz)
        # 候选隐状态
        H_tilde = torch.tanh(X_t @ Wxh + (R * H) @ Whh + bh)
        # 当前时间步的隐状态
        H = Z * H + (1 - Z) * H_tilde
        # 输出
        Y_t = H @ Wx + bx  # 形状为(batch_size, vocab_size)
        outputs.append(Y_t)

    # 将输出列表转换为张量，形状为(seq_len, batch_size, vocab_size)
    # 然后转置为(batch_size, seq_len, vocab_size)
    outputs = torch.stack(outputs, dim=1)
    return outputs, (H,)


#按上述思路手动整合成一个完整的模型
class GRU_layer(nn.Module):
    # 构建一个GRU单层模块
    def __init__(
            self,
            input_size,  # 输入维度
            hidden_size,  # 隐藏层维度
            device,
    ):
        super(GRU_layer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

    def get_parameter(self):
        input_size = self.input_size
        num_hiddens = self.hidden_size

        def normal(shape):
            return torch.randn(size=shape, device=self.device) * 0.01

        def three():
            return (
                normal((input_size, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros((num_hiddens), device=self.device)
            )

        Wxr, Whr, br = three()
        Wxz, Whz, bz = three()
        Wxh, Whh, bh = three()

        # 注意：这里不包含输出层参数Wx,bx！
        params = [Wxr, Whr, br, Wxz, Whz, bz, Wxh, Whh, bh]

        for param in params:
            param.requires_grad_(True)

        self.params = params

    def forward(self, x):
        self.get_parameter()
        Wxr, Whr, br, Wxz, Whz, bz, Wxh, Whh, bh = self.params

        batch_size, seq_len, input_dim = x.shape

        # 初始化隐状态
        H = torch.zeros((batch_size, self.hidden_size), device=self.device)
        outputs = []

        # 按时间步循环
        for t in range(seq_len):
            X_t = x[:, t, :]
            # 重置门
            R = torch.sigmoid(X_t @ Wxr + H @ Whr + br)
            # 更新门
            Z = torch.sigmoid(X_t @ Wxz + H @ Whz + bz)
            # 候选隐状态
            H_tilde = torch.tanh(X_t @ Wxh + (R * H) @ Whh + bh)
            # 当前时间步的隐状态
            H = Z * H + (1 - Z) * H_tilde
            # 存储每个时间步的隐藏状态
            outputs.append(H)

        # 输出形状: (batch_size, seq_len, hidden_size)
        outputs = torch.stack(outputs, dim=1)
        return outputs, (H,)


class GRU_model(nn.Module):
    # 使用GRU_layer构建一个GRU模型
    def __init__(self, vocab, device, embed, dropout=0.02):
        super(GRU_model, self).__init__()
        self.embed = embed
        self.vocab_size = len(vocab)
        self.device = device

        # 嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.embed)

        # GRU层（只输出隐藏状态）
        self.layer1 = GRU_layer(
            input_size=self.embed,  # 嵌入维度
            hidden_size=256,
            device=device
        )

        self.layer2 = GRU_layer(
            input_size=256,  # 第一层隐藏状态维度
            hidden_size=128,
            device=device
        )

        self.layer3 = GRU_layer(
            input_size=128,  # 第二层隐藏状态维度
            hidden_size=64,
            device=device
        )

        # 输出层：将隐藏状态转换为词汇表大小
        self.output_layer = nn.Linear(64, self.vocab_size)

    def forward(self, x):
        if x.dim() == 2:
            x = self.embedding(x)
        elif x.dim() != 3:
            print("输入了错误的形状")
            return None

        # 第一层GRU输出隐藏状态
        x1, _ = self.layer1(x)  # 形状: (batch_size, seq_len, 256)
        x1 = torch.relu(x1)

        # 第二层GRU
        x2, _ = self.layer2(x1)  # 形状: (batch_size, seq_len, 128)
        x2 = torch.relu(x2)

        # 第三层GRU
        x3, h = self.layer3(x2)  # 形状: (batch_size, seq_len, 64)
        x3 = torch.relu(x3)

        # 通过输出层得到最终输出
        output = self.output_layer(x3)  # 形状: (batch_size, seq_len, vocab_size)
        output = nn.functional.softmax(output, dim=-1)

        return output, h

if __name__ == '__main__':
    batch_size = 32
    num_hiddens = 256
    vocab = get_vocab() #导入字典
    seq_len =  10
    vocab_size = len(vocab)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #生成测试数据
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    model = GRU_model(
        vocab=vocab,
        device='cpu',
        embed=100
    )

    outputs,(h,) = model(x)

    print(outputs.shape) #预期结果 batch_size * seq_len * vocab_size
    print(h.shape) #预期结果 batch_size * 64(layer3的num_hiddens)
    #现在该模型可以用于正常的训练了，同时也可以使用GRUlayer来自定义新的模型











