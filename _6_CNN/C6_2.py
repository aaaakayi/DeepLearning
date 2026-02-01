import tensorflow as tf
#以下代码基于通道数为1的情况
def corr2d(X,y):
    #互相关运算的简单实现
    #X为输入张量，y为卷积核
    h1, w1 = y.shape
    h2, w2 = X.shape
    re = tf.Variable(tf.zeros((X.shape[0] - h1 + 1, X.shape[1] - w1 + 1)))
    for i in range(h2-h1+1):
        for j in range(w2-w1+1):
            re[i,j].assign(tf.reduce_sum(X[i:i+h1,j:j+w1]*y))
    return re

def test_corr2d():
    X = tf.constant([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = tf.constant([[0.0, 1.0], [2.0, 3.0]])
    print(corr2d(X, K))

class Conv2D_Dense(tf.keras.layers.Layer):
    #简单的卷积层，定义卷积核与偏置，前向传播使用corr2d实现
    def __init__(self,kernel_size,activation=None):
        super().__init__()
        self.kernel_size = kernel_size
        if activation == None:
            self.activation = lambda x:x
        else:
            self.activation = activation

    def build(self,input_shape):
        initializer = tf.random_normal_initializer
        self.kernel = self.add_weight(
            name='k_matrix',
            shape = self.kernel_size,
            initializer = initializer,
            trainable = True
        )
        self.bias = self.add_weight(
            name='bias',
            shape = [1,],
            initializer = initializer,
            trainable = True
        )

    def call(self,X):
        #期望返回一个张量,X为输入，卷积核的参数是在模型训练中修改的
        #self.weights为卷积核，self.bias为偏置，大小需要在build中传输入kernel_size构建N*N的卷积核与偏置
        return self.activation(corr2d(X,self.kernel) + self.bias)

def test_Conv2D_Dense():
    # 现在有一个6*8的矩阵
    X = tf.Variable(tf.ones((6, 8)))
    X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
    print(X)

    # 实例化一个卷积层,卷积核为1,2大小，使用默认参数
    Dense = Conv2D_Dense(kernel_size=[1, 2])
    Y = Dense(X)
    print(Y)

    # 修改卷积核参数
    new_kernel = tf.constant([[1.0, -1.0]], dtype=tf.float32)
    new_bias = tf.constant([0], dtype=tf.float32)
    Dense.kernel.assign(new_kernel)
    Dense.bias.assign(new_bias)
    Y = Dense(X)
    print(Y)
    #验证修改后的卷积核是否准确
    K = tf.constant([[1.0, -1.0]])
    Y = corr2d(X, K)
    print(Y)


#课后练习题

#1.构建一个具有对角线边缘的图像X。
#1.1如果将本节中举例的卷积核K应用于X，会发生什么情况？
    #答：代码如下。
#1.2如果转置X会发生什么？
    #答：与1问相同，对角阵的转账还是对角阵
#1.3如果转置K会发生什么？
    #答：代码如下。
def  test1_1():
    X = tf.Variable(tf.ones((6, 6)))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if (i == j):
                X[i, j].assign(1)
            else:
                X[i, j].assign(0)
    model = Conv2D_Dense(kernel_size=[1, 2])
    model(X)
    model.kernel.assign([[1.0, -1.0]])
    model.bias.assign([0])
    Y = model(X)
    print(Y)

    model = Conv2D_Dense(kernel_size=[2, 1])
    model(X)
    model.kernel.assign([[1.0],[-1.0]]) #转置成一个两行一列的数据
    model.bias.assign([0])
    Y = model(X)
    print(Y)



