#多通道CNN，一般都是多通道的，像每个RGB输入图像具有3*h*w的形状
#本章所给数据采用 通道数 * h * w，需要转置
import tensorflow as tf

def corr2d_multi_in(X,k):
    #多输入通道的互相关功能的简单实现，只有一个输出通道。
    #X为数据，k为卷积核
    #z为通道的维数
    X = tf.transpose(X, [1, 2, 0])
    k = tf.transpose(k, [1, 2, 0])

    h_in,w_in,c_in = X.shape
    h_k,w_k,c_k = k.shape

    output_h = h_in - h_k + 1
    output_w = w_in - w_k + 1
    re = tf.Variable(tf.zeros((output_h, output_w)))
    for i in range(output_h):
        for j in range(output_w):
            window = X[i:i+h_k,j:j+w_k,:]
            temp = tf.reduce_sum(window * k)
            re[i,j].assign(temp)
    return re

def test_corr2d_multi_in():
    X = tf.constant(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        ]
    )  # X是一个2 * 3  通道数为2的数据（因为底下的k限制了通道数最多为2)
    K = tf.constant(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[1.0, 2.0], [3.0, 4.0]]
        ]
    )
    X = tf.transpose(X, [1, 2, 0])
    K = tf.transpose(K, [1, 2, 0])
    print(corr2d_multi_in(X, K))

#多输入多输出通道，每一个输出通道对应一个 输入通道 * h * w的卷积核，合起来就是输出通道 * 输入通道 * h * w的卷积核
def corr2d_multi_in_and_out(X,k):
    #k是一个输出通道 * 输入通道 * h * w的卷积核

    c_in, h_in, w_in = X.shape
    c_k, h_k, w_k= k[0].shape
    output_h = h_in - h_k + 1
    output_w = w_in - w_k + 1

    re = tf.Variable((tf.zeros((k.shape[0],output_h,output_w))))

    for i in range(k.shape[0]):
        k_i = k[i]
        temp = corr2d_multi_in(X, k_i)
        re[i,:,:].assign(temp)

    return re

def test_corr2d_multi_in_and_out():
    X = tf.constant(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        ]
    )
    K = tf.constant(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[1.0, 2.0], [3.0, 4.0]]
        ]
    )
    K = tf.stack((K, K + 1, K + 2), 0)
    print(corr2d_multi_in_and_out(X, K))

#1*1卷积核



