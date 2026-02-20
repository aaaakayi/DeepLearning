import torch
import os
from .vocab.vocab_build import VocabularyBuilder
from torch.utils.data import Dataset, DataLoader


#构建数据集

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_vocab, tgt_vocab, max_len=50,limit_len = False, max_sentences = 20000):
        """
        src_file: 源语言文件路径（英文tok文件）
        tgt_file: 目标语言文件路径（中文tok文件）
        src_vocab: 英文词汇表
        tgt_vocab: 中文词汇表
        max_len: 最大序列长度
        """
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

        # 读取数据
        self.src_sentences = []
        self.tgt_sentences = []

        sentence_count = 0  # 计数器

        with open(src_file, 'r', encoding='utf-8') as f_src, \
                open(tgt_file, 'r', encoding='utf-8') as f_tgt:

            for src_line, tgt_line in zip(f_src, f_tgt):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()

                # 跳过空行
                if not src_line or not tgt_line:
                    continue

                self.src_sentences.append(src_line)
                self.tgt_sentences.append(tgt_line)
                sentence_count += 1

                # 达到最大句子数时立即停止
                if sentence_count >= max_sentences and limit_len:
                    break

            print(f"已读取 {sentence_count} 个句子对（最大限制: {max_sentences}）")

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        # 获取原始句子（已分词，空格分隔）
        src_text = self.src_sentences[idx]  # 英文
        tgt_text = self.tgt_sentences[idx]  # 中文

        # 编码为索引
        src_indices = self.src_vocab.encode(
            src_text,
            add_bos=True,
            add_eos=True,
            max_len=self.max_len
        )

        tgt_indices = self.tgt_vocab.encode(
            tgt_text,
            add_bos=True,
            add_eos=True,
            max_len=self.max_len
        )

        return {
            'src': torch.tensor(src_indices, dtype=torch.long),  # 源序列
            'tgt': torch.tensor(tgt_indices, dtype=torch.long),  # 目标序列
            'src_text': src_text,  # 原始文本（调试用）
            'tgt_text': tgt_text  # 原始文本（调试用）
        }


def collate_fn(batch,pad_token_id = 0):
    """处理变长序列，填充到相同长度"""
    # 找到批次中的最大长度
    src_max_len = max(len(item['src']) for item in batch)
    tgt_max_len = max(len(item['tgt']) for item in batch)

    # 初始化批处理张量
    batch_size = len(batch)
    src_batch = torch.full((batch_size, src_max_len),fill_value=pad_token_id, dtype=torch.long)
    tgt_batch = torch.full(size=(batch_size, tgt_max_len),fill_value=pad_token_id, dtype=torch.long)

    # 填充数据
    for i, item in enumerate(batch):
        src_len = len(item['src'])
        tgt_len = len(item['tgt'])

        src_batch[i, :src_len] = item['src']
        tgt_batch[i, :tgt_len] = item['tgt']

    return src_batch, tgt_batch


def get_train_data():
    train_data_path =r"C:/Users/Aumu/PycharmProjects/DeepLearning/Transfomer/Dataset/data/train_data.pt"

    if os.path.exists(train_data_path):
        print("训练数据存在")
        train_dataset = torch.load(train_data_path, weights_only=False)
    else:
        src_file = r"C:/Users/Aumu/PycharmProjects/DeepLearning/data/en-zh/tok/ted_train_en-zh.tok.en"
        tgt_file = r"C:/Users/Aumu/PycharmProjects/DeepLearning/data/en-zh/tok/ted_train_en-zh.tok.zh"
        print(f"训练数据不存在,从{src_file}和{tgt_file}中构建")

        train_dataset = TranslationDataset(
            src_file=src_file,
            tgt_file=tgt_file,
            src_vocab=vocab_en,
            tgt_vocab=vocab_zh,
            max_len=50
        )
        torch.save(train_dataset, train_data_path)

    return train_dataset


if __name__ == "__main__":
    # 加载中英文字典
    vocab_zh = VocabularyBuilder.load_vocab("vocab/vocab_zh.json")
    vocab_en = VocabularyBuilder.load_vocab("vocab/vocab_en.json")

    train_dataset = get_train_data()

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn  # 处理填充
    )

    # 测试一个批次
    for src_batch, tgt_batch in train_loader:
        print(f"批次形状: src={src_batch.shape}, tgt={tgt_batch.shape}")

        # 解码第一句查看
        src_sample = src_batch[0]
        tgt_sample = tgt_batch[0]

        src_text = vocab_en.decode(src_sample.tolist())
        tgt_text = vocab_zh.decode(tgt_sample.tolist())

        print(f"英文: {src_text}")
        print(f"中文: {tgt_text}")
        break
