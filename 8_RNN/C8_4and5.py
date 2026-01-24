#RNN *即使时间步向前递进，循环神经网络始终使用一组同样数量的参数 故产生了循环的感觉，因为参数一直是固定的所以共用一个神经网络，就像循环一样
#这个神经网络应该应该有两个输出，两个输出。输入：h(t-1)和x(t)，输出：h(t)和O(t)
#RNN像存在一个潜在的网络层的MLP,他这一个潜在的网络层每一次像前传播都会更新，并且影响这一步的输出。
import tensorflow as tf

#从头实现RNN
def get_param(vocab_size,num_hiddens):
    # 得到神经网络的参数(随机)
    """
    :param vocab_size: 字典大小s
    :param num_hiddens: 隐藏层大小
    :return:
    """
    #输入和输出都不能脱离字典
    output_dim = input_dim = vocab_size

    #隐藏层参数
    W_xh = tf.Variable(tf.random.normal((input_dim,num_hiddens)),dtype=tf.float32)
    W_hh = tf.Variable(tf.random.normal((num_hiddens,num_hiddens)),dtype=tf.float32) #隐藏层到隐藏层的权重
    b_h = tf.Variable(tf.zeros((num_hiddens,)),dtype=tf.float32)

    #输出层参数
    W_xo = tf.Variable(tf.random.normal((num_hiddens,output_dim)),dtype=tf.float32)
    b_o = tf.Variable(tf.zeros((output_dim,)),dtype=tf.float32)

    return [W_xh,W_hh,b_h,W_xo,b_o]

def init_rnn_state(batch_size,num_hiddens):
    #初始化RNN时生成h(0)，全部用0填充,返回元组方便与其它一致
    return (tf.zeros((batch_size, num_hiddens)), )


def rnn_forward(x,vocab_size,num_hiddens):
    """

    :param x: 输入时间步 结构为[batch_size, seq_length]
    假设这里的x为[[1,2,3],[4,5,6]] batch_size=2,imput_dim=vocal_size

    :param vocab_size: 字典大小
    :param num_hiddens: 隐藏层大小
    :return: 返回输出和h
    """
    #进行一次简单的时间步向前传播,这里不使用激活函数
    batch_size = x.shape[0]
    seq_length = x.shape[1]
    [W_xh, W_hh, b_h, W_xo, b_o] = get_param(vocab_size,num_hiddens)#得到参数

    h0,  = init_rnn_state(batch_size=batch_size,num_hiddens=num_hiddens) #初始化隐藏层

    h_temp = h0
    output = []
    for i in range(seq_length):
        x_t = x[:,i]
        x_t = tf.one_hot(x_t,vocab_size)
        #计算每一个时间的结果
        h_temp = x_t @ W_xh + h_temp @ W_hh + b_h
        output.append(h_temp @ W_xo + b_o)

    output = tf.stack(output, axis=0)
    return output,h_temp

def test_my_rnn():
    #简单的测试函数
    x = tf.reshape(tf.range(6), [2, 3])
    print(x)

    output, h = rnn_forward(x, 10, 10)
    print(output, h)


#把函数整合成一个类
class my_RNN():
    def __init__(self,vocab_size,num_hiddens,x):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.x = x

    def call(self):
        return rnn_forward(self.x,self.vocab_size,self.num_hiddens)
    #上面的函数可以整合到底下来

#简单的测试
x = tf.reshape(tf.range(6), [2, 3])
rnn = my_RNN(10,10,x)
print(rnn.call())



