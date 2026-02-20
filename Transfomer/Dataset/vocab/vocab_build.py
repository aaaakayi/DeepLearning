import os
from collections import Counter
import json


class VocabularyBuilder:
    """词汇表构建器"""

    def __init__(self, min_freq=2, max_size=50000, special_tokens=None):
        """
        参数:
            min_freq: 最小词频
            max_size: 最大词汇表大小
            special_tokens: 特殊标记
        """
        if special_tokens is None:
            special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']

        self.min_freq = min_freq
        self.max_size = max_size
        self.special_tokens = special_tokens
        self.idx2word = {}
        self.word2idx = {}

    def build_from_tok_file(self, filepath, lang='en'):
        """从分词文件构建词汇表"""
        print(f"从 {filepath} 构建 {lang} 词汇表...")

        # 统计词频
        word_counter = Counter()

        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 按空格分词（tok文件已经是分词格式）
                tokens = line.strip().split()
                word_counter.update(tokens)

        print(f"总共有 {len(word_counter)} 个不同的词元")

        # 过滤低频词
        filtered_words = [(word, count) for word, count in word_counter.items()
                          if count >= self.min_freq]
        filtered_words.sort(key=lambda x: x[1], reverse=True)

        print(f"过滤后（min_freq={self.min_freq}）: {len(filtered_words)} 个词")

        # 构建词汇表
        self._create_vocab(filtered_words)

        return word_counter

    def __len__(self):
        return len(self.word2idx)

    def _create_vocab(self, filtered_words):
        """创建词汇表映射"""
        # 添加特殊标记
        for i, token in enumerate(self.special_tokens):
            self.word2idx[token] = i
            self.idx2word[i] = token

        # 添加普通词（限制最大数量）
        max_regular = self.max_size - len(self.special_tokens)
        for word, count in filtered_words[:max_regular]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"词汇表大小: {len(self.word2idx)} (特殊标记: {len(self.special_tokens)})")

    def save_vocab(self, save_path):
        """保存词汇表"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'special_tokens': self.special_tokens,
            'min_freq': self.min_freq,
            'max_size': self.max_size
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        print(f"词汇表已保存到: {save_path}")

    @classmethod
    def load_vocab(cls, load_path):
        """加载词汇表"""
        with open(load_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        builder = cls(
            min_freq=vocab_data['min_freq'],
            max_size=vocab_data['max_size'],
            special_tokens=vocab_data['special_tokens']
        )
        builder.word2idx = vocab_data['word2idx']
        builder.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}

        return builder

    def encode(self, text, add_bos=False, add_eos=False, max_len=None):
        """将文本编码为索引"""
        # 分词（如果已经是分词文本，直接split）
        tokens = text.split() if isinstance(text, str) else text

        # 添加特殊标记
        if add_bos:
            tokens = ['<bos>'] + tokens
        if add_eos:
            tokens = tokens + ['<eos>']

        # 转换为索引
        indices = []
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx['<unk>'])

        # 填充或截断
        if max_len:
            if len(indices) < max_len:
                indices = indices + [self.word2idx['<pad>']] * (max_len - len(indices))
            else:
                indices = indices[:max_len]

        return indices

    def decode(self, indices, remove_special=True):
        """将索引解码为文本"""
        tokens = []
        for idx in indices:
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if remove_special and word in self.special_tokens:
                    continue
                tokens.append(word)
            else:
                tokens.append('<unk>')

        return ' '.join(tokens)
def get_vocab():
    vocab_zh_path = r"C:/Users/Aumu/PycharmProjects/DeepLearning/data/en-zh/tok/ted_train_en-zh.tok.zh"
    vocab_en_path = r"C:/Users/Aumu/PycharmProjects/DeepLearning/data/en-zh/tok/ted_train_en-zh.tok.en"

    vocab_zh_save_path = r"C:/Users/Aumu/PycharmProjects/DeepLearning/Transfomer/Dataset/vocab/vocab_zh.json"
    vocab_en_save_path = r"C:/Users/Aumu/PycharmProjects/DeepLearning/Transfomer/Dataset/vocab/vocab_en.json"

    if os.path.exists(vocab_zh_save_path):
        print("中文字典已存在")
        vocab_zh = VocabularyBuilder.load_vocab(vocab_zh_save_path)
    else:
        print(f"中文字典不存在,正在从{vocab_zh_path}中构建...")
        vocab_zh = VocabularyBuilder()
        vocab_zh.build_from_tok_file(
            filepath=vocab_zh_path,
            lang='zh'
        )
        vocab_zh.save_vocab(vocab_zh_save_path)

    if os.path.exists(vocab_en_save_path):
        print("英文字典已存在")
        vocab_en = VocabularyBuilder.load_vocab(vocab_en_save_path)
    else:
        print(f"英文字典不存在,正在从{vocab_en_path}中构建...")
        vocab_en = VocabularyBuilder()
        vocab_en.build_from_tok_file(
            filepath=vocab_en_path,
            lang='en'
        )
        vocab_en.save_vocab(vocab_en_save_path)

    return vocab_zh,vocab_en

if __name__ == "__main__":
    #构建我们需要的中英文词汇表

    vocab_zh, vocab_en = get_vocab()

    print(len(vocab_en))
    print(len(vocab_zh))

    text = "我 是 一个 人"
    seq = vocab_zh.encode(text, add_bos=True, add_eos=True)
    print(f"编码: {seq}")

    decoded = vocab_zh.decode(seq)
    print(f"解码: {decoded}")




