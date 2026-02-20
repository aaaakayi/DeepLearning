# Nadaraya-Watson核回归
# 本节的主要目的在于探索根据所给值如何进行注意力汇聚，一种简单的方法就是引入核
# 核是与当前输入x有关的，利用它可以进行不带参数的注意力汇聚以及带参数的注意力汇聚
# 

import torch
import torch.nn as nn
from C10_1 import show_heatmap
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
def plot_multiple_xy_mappings(data_pairs, xlabel='X', ylabel='Y', title='Multiple XY Mappings'):
    import matplotlib.pyplot as plt
    """
    绘制多对x-y映射在同一图上

    参数:
    data_pairs: list of tuples, 每个元组格式为 (x_data, y_data, label, color, linewidth)
                x_data: x轴数据 (list/array)
                y_data: y轴数据 (list/array)
                label: 线条标签 (str)
                color: 线条颜色 (str, 可选)
                linewidth: 线宽 (float, 可选)
    """
    plt.figure(figsize=(10, 6))

    for pair in data_pairs:
        x = pair[0]
        y = pair[1]
        label = pair[2] if len(pair) > 2 else None
        color = pair[3] if len(pair) > 3 else None
        linewidth = pair[4] if len(pair) > 4 else 1.5

        plt.plot(x, y, label=label, color=color, linewidth=linewidth)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if any(len(pair) > 2 and pair[2] is not None for pair in data_pairs):
        plt.legend()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def train_model(x,y,n_train,n_test,x_test):
    X_tile = x.repeat((n_train, 1))
    Y_tile = y.repeat((n_train, 1))
    keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

    net = NWKernelRegression()
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)

    for epoch in range(5):
        trainer.zero_grad()
        l = loss(net(x, keys, values), y)
        l.sum().backward()
        trainer.step()
        print(f'epoch {epoch + 1}, loss {float(l.sum().detach()):.6f}')

    keys = x.repeat((n_test, 1))
    values = y.repeat((n_test, 1))
    y_hat = net(x_test, keys, values).unsqueeze(1).detach()

    return y_hat,net.attention_weights

if __name__ == '__main__':
    # 先构造一对x -> y的映射
    n_train = 50  # 训练样本数
    x, _ = torch.sort(torch.rand(n_train) * 5)  # 排序后的训练样本

    def f(x):
        return 2 * torch.sin(x) + x ** 0.8


    y = f(x) + torch.normal(0.0, 0.5, (n_train,))  # 加上一些随机噪声

    x_test = torch.arange(0, 5, 0.1)  # 测试样本
    y_truth = f(x_test)  # 测试样本的真实输出
    n_test = len(x_test)  # 测试样本数

    # 最简单的情况，把y的平均值作为y_pred
    y_hat = torch.repeat_interleave(y.mean(), n_test)

    #基于高斯核的非参数注意力汇聚
    X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    attention_weights = nn.functional.softmax(-(X_repeat - x) ** 2 / 2, dim=1)
    y_Nadaraya_Watson = torch.matmul(attention_weights, y)

    #参数参与的注意力汇聚
    y_Nadaraya_Watson_with_w,net_attention_weights = train_model(x,y,n_train,n_test,x_test)



    data = [
        (x_test,y_truth,'true','red',2),
        (x_test,y_hat, 'y_hat', 'green', 2),
        (x_test,y_Nadaraya_Watson,'Nadaraya-Watson','blue',2),
        (x_test,y_Nadaraya_Watson_with_w, 'Nadaraya_Watson_with_w', 'yellow', 2)
    ]

    plot_multiple_xy_mappings(data)

    #查看heatmap
    show_heatmap(attention_weights=attention_weights)
    show_heatmap(net_attention_weights)