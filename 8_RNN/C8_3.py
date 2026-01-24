#考虑从原始文本开始构建神经网络可以用于训练的特征和标签
#借鉴了统计学领域马尔可夫模型用特定长度、最新历史子序列建模完整序列特征的思想
#比如一个序列1，2，3，4，5，6 特征可以是2，3，4，标签就是3，4，5
#有两种方法生成特征，随机采样和顺序采样
from typing import Iterator
import tensorflow as tf
import random
class LanguageModelDataGenerator():
    def __init__(self,corpus,batch_size,num_size,shuffle=True):
        """

        :param corpus: 用于分割的语料
        :param batch_size: 批次数
        :param num_size: 序列长度
        :param shuffle:是否打乱，即是否随机取样
        """
        self.corpus = corpus
        self.batch_size = batch_size
        self.num_size = num_size
        self.shuffle = shuffle

        # 返回可能的序列数，比如corpus长度为10，num_size=3,只有7个序列，第八个（8，9，10）没有对应的标签所以不计入
        self.sample_size = len(self.corpus) - self.num_size

    def __len__(self):
        #返回可能的批次数，像下取整舍去凑不齐的情况
        return self.sample_size // self.batch_size

    def __iter__(self) -> Iterator[tuple[tf.Tensor(), tf.Tensor()]]:
        index = list(range(self.sample_size))
        if self.shuffle:
            #随机打乱
            random.shuffle(index)

        for i in range(0,self.sample_size,self.batch_size):
            batch_indices = index[i:i + self.batch_size]

            features = [self.corpus[idx: idx + self.num_size]
                        for idx in batch_indices]
            labels = [self.corpus[idx + 1: idx + self.num_size + 1]
                      for idx in batch_indices]

            yield tf.constant(features), tf.constant(labels)

corpus = [1,2,3,4,5,6,7,8,9,10]
llm = LanguageModelDataGenerator(
    corpus=corpus,
    batch_size=2,
    num_size=3,
    shuffle=False
)

for batch_size, (features, labels) in enumerate(llm):
    print(f"batch: {batch_size}")
    print(f"features:{features.numpy().tolist()}")
    print(f"labels:{labels.numpy().tolist()}")