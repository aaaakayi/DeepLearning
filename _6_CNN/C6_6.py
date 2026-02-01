#LeNet 一个经典的卷积神经网络,是最早发布的卷积神经网络之一
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def LeNet1():
    # conv2d -> pooling -> conv2d -> pooling -> MLP
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(filters=6,kernel_size=5,activation='sigmoid',padding='same'),
            tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(filters=12, kernel_size=5, activation='sigmoid'),
            tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120,activation='sigmoid'),
            tf.keras.layers.Dense(84,activation='sigmoid'),
            tf.keras.layers.Dense(10)
        ],name='LeNet1'
    )

def LeNet():
    # conv2d -> pooling -> conv2d -> pooling -> MLP
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(25,activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(15,activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ],name='LeNet'
    )
def trans_to_tensor(data):
    """将400维向量转换为4D张量 (批量大小, 20, 20, 1)"""
    data = (data-data.mean())/data.std()
    # 先重塑为 (样本数, 20, 20)
    reshaped = data.reshape(-1, 20, 20)
    # 添加通道维度，变成 (样本数, 20, 20, 1)
    return reshaped[..., np.newaxis].astype('float32')


def plot_training_history(history):
    import matplotlib.pyplot as plt
    """绘制训练过程中的损失和准确率曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制损失曲线
    ax1.plot(history.history['loss'], label='train loss')
    ax1.plot(history.history['val_loss'], label='test loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('train and test loss')
    ax1.legend()
    ax1.grid(True)

    # 绘制准确率曲线
    ax2.plot(history.history['accuracy'], label='train prediction')
    ax2.plot(history.history['val_accuracy'], label='test prediction')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('train and test prediction')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 因为我没有d2l包，把数据集改成一个手写数字的数据集
    X = np.load('X.npy')  # X.shape = (5000,400) 5000张20*20的灰度图像
    y = np.load('y.npy')  # y.shape = (5000,1) 标签
    X = (X - X.mean()) / X.std()
    #X = trans_to_tensor(X) #转化成张量形式 h * w * 通道数
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42) #划分训练集和预测集

    # 进行一次训练 需要的输入数据为 batch_size * h * w * 通道数
    model = LeNet()


    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    #model.build(input_shape=(None, 20, 20, 1))
    #model.summary #可以查看模型的参数

    history = model.fit(
        X_train, y_train,
        batch_size=128,  # 批大小
        epochs=100,  # 训练轮数
        validation_data=(X_test, y_test),  # 验证数据
        verbose=1  # 显示训练进度
    )

    # 评估模型
    print("\n评估模型...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"验证集损失: {test_loss:.4f}")
    print(f"验证集准确率: {test_accuracy:.4f}")

    #输出图像
    plot_training_history(history)

"""
100个epoch后 accuracy: 0.0986 - loss: 2.3026 - val_accuracy: 0.1033 - val_loss: 2.3026
loss=2.3026是多分类任务里一个很经典的错误，证明所有的可能性都是0.1，loss一直等于这个值不变，那就说明某一层的输出全0
证明模型不合适,实际上sigmoid容易造成梯度消失导致出现这个问题
"""
