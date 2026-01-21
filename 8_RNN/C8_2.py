#文本是最常见的序列数据
#文本预处理：文本 -> 词元，词元(token)是最小的单位
#给每个词元按照出现频率从小到大给一个索引，这个词元(token)到索引的映射叫做字典
import collections
import re

def read_txt_to_lines(str):
    # 先把数据一行一行的导入一个列表中,同时清理数据只留下小写字母
    with open(str,'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+',' ',line).strip().lower() for line in lines]

def tokenize(lines, token='word'):
    # 词元(token)化
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print("错误token类型：" + token)


class Vocab:  #@save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)




if __name__ == '__main__':
    #先把数据一行一行的导入一个列表中,同时清理数据只留下小写字母
    str = 'timemachine.txt'
    lines = read_txt_to_lines(str)

    # 词元(token)化
    token = tokenize(lines,token='word')

    # 建立字典
    vocab = Vocab(token)
    print(list(vocab.token_to_idx.items())[:10])

    #预处理标准流程：数据导入列表 -> 词元化 -> 建立字典映射

