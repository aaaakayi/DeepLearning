from .C8_2 import read_txt_to_lines,tokenize,Vocab

def get_vocab():
    str = r'C:/Users/Aumu/PycharmProjects/DeepLearning/_8_RNN/timemachine.txt'
    lines = read_txt_to_lines(str)

    # 词元(token)化
    token = tokenize(lines, token='word')

    # 建立字典
    vocab = Vocab(token)
    return vocab
