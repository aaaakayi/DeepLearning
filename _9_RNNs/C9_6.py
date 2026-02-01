#encoder and decoder
from torch import nn
import torch
from _8_RNN import get_vocab

#@save
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

#@save
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

#@save
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

#------------------------------------------------------------
#------------------实现一个简单的基于RNN的encoder-decoder架构-----

class RNNencoder(Encoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0,**kwargs):
        super(RNNencoder,self).__init__()
        self.embeded = nn.Embedding(vocab_size,embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self,x):
        x = self.embeded(x)
        x = x.permute(1, 0, 2)  # 转置为 (num_steps, batch_size, embed_size) 以适应RNN
        output, state = self.rnn(x)
        return output, state

class RNNDecoder(Decoder):
   def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
       super(RNNDecoder, self).__init__()
       self.embedding = nn.Embedding(vocab_size, embed_size)
       self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
       self.dense = nn.Linear(num_hiddens, vocab_size)

   def init_state(self, enc_outputs, *args):
       # enc_outputs是(输出, 隐状态)的元组，我们取隐状态作为解码器的初始状态
       return enc_outputs[1]

   def forward(self, X, state):
       X = self.embedding(X).permute(1, 0, 2)  # 转为 (num_steps, batch_size, embed_size)
       # 这里为了简单，我们假设编码器的最后一个隐状态是上下文向量，并重复到每个时间步
       # 实际上，注意力机制会将编码器的所有输出用于每个时间步
       context = state[-1].repeat(X.shape[0], 1, 1)  # 取最后一层的隐状态，并重复
       X_and_context = torch.cat((X, context), 2)
       output, state = self.rnn(X_and_context, state)
       output = self.dense(output).permute(1, 0, 2)  # 转回 (batch_size, num_steps, vocab_size)
       return output, state

if __name__ == '__main__':
    vocab = get_vocab()
    encoder = RNNencoder(len(vocab),100,256,3)
    decoder = RNNDecoder(len(vocab),100,256,3)
    model = EncoderDecoder(encoder=encoder,decoder=decoder)


