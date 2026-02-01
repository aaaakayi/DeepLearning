#使用tensorflow keras-API来轻松创建RNN
import tensorflow as tf
from C8_2 import get_vocab
num_hiddens = 256
batch_size = 32
vocab = get_vocab()
vocab_size = len(vocab)
num_steps = 10

class RNNModel(tf.keras.layers.Layer):
    def __init__(self,num_hiddens,vocab_size,**kwargs):
        super(RNNModel,self).__init__(**kwargs)

        #隐藏层
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.SimpleRNNCell(
                num_hiddens,
                kernel_initializer='glorot_uniform',
                activation='tanh'
            ),
            return_sequences=True,
            return_state=True
        )

        #稠密层
        self.dense = tf.keras.layers.Dense(
            units = vocab_size,
            activation ='softmax'
        )

        #设置词汇表
        self.vocab = None

    def call(self,X):
        Y,new_status = self.rnn(X) #X的形状是batch_size,seq_len,vocal_len
        out = self.dense(tf.reshape(Y, (-1, Y.shape[-1])))
        return out,new_status

    def begin_state(self, *args, **kwargs):
        return self.rnn.cell.get_initial_state(*args, **kwargs)

    def get_vocab(self,vocab):
        self.vocab = vocab


def output_txt(output,batch_size,seq_len,vocab_size,vocab):
    output = tf.reshape(output,(batch_size,seq_len,vocab_size))

    indices = tf.argmax(output,axis=-1) #shape : batch_size * seq_len

    indices_np = indices.numpy()

    construct_token = []
    for i in range(batch_size):
        # 将索引列表转换为token列表
        tokens = vocab.to_tokens(indices_np[i].tolist())
        construct_token.append(tokens)

    return construct_token


#生成一个示例数据
X = tf.random.uniform((batch_size,10),maxval=vocab_size,dtype=tf.int32)
X = tf.one_hot(X,vocab_size)

#进行一次前向传播，采用默认参数
net = RNNModel(num_hiddens=num_hiddens,vocab_size=vocab_size)
out,new_status = net(X)

print(out) #原始输出

#将输出转化回字符
construct_tokens = output_txt(
    output=out,
    batch_size=batch_size,
    seq_len=num_steps,
    vocab_size=vocab_size,
    vocab=vocab
)
construct_tokens_str = ' '.join(construct_tokens[1])
print(construct_tokens_str) #打印第一个的结果


