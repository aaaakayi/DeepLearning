import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#定义网络
def net():
    #一个简单的模型，也可以按自己的需求修改
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(15, activation='relu'),
            tf.keras.layers.Dense(1)
        ]
    )

def model_net(net,train_features,train_labels,test_features, test_labels):
    # 得到一个训练好的模型
    #基于train_features,train_labels,test_features, test_labels来训练与评估
    loss = tf.keras.losses.MeanSquaredError()

    model = net()
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=['mae']
    )

    history = model.fit(
        train_features, train_labels,
        batch_size=16,
        epochs=50,
        validation_data=(test_features, test_labels),
        verbose=1
    )
    return model

#单步预测
def draw_onestep(model_net,train_features,train_labels):
    train_predictions = model_net.predict(train_features)

    # 1. 预测值与真实值对比（训练集）
    plt.figure(figsize=(10, 5))

    # 展平数组以便绘图
    train_labels_flat = train_labels.numpy().flatten()
    train_predictions_flat = train_predictions.flatten()

    # 绘制前600个点
    n_plot = 600
    plt.plot(range(n_plot), train_labels_flat[:n_plot],
             'bo-', label='True', alpha=0.7)
    plt.plot(range(n_plot), train_predictions_flat[:n_plot],
    'rs--', label = 'Predcit', alpha = 0.7)

    plt.xlabel('sample')
    plt.ylabel('number')
    plt.title('predcit and true')
    plt.legend()
    plt.grid(True)
    plt.show()

#k步预测
#使用我们预测出来的数去预测之后的数据
def kstep_pre(model_net,T,n_train,x,tau):
    #model_net:训练好的模型，T:总序列数，n_train:对照数据，x:数据
    kstep_pre = tf.Variable(tf.zeros(T))
    kstep_pre[:n_train + 1].assign(x[:n_train + 1].numpy())
    for i in range(n_train + 1, T):
        # n_train=600 现在是601 则kstep_pre[i-tau:i]为从597：601为597，598，599，600
        kstep_pre[i].assign(
            tf.reshape(model_net(tf.reshape(kstep_pre[i - tau:i], (1, -1))), ())
        )

    plt.figure(figsize=(10, 5))

    # 展平数组以便绘图
    kstep_pre_flat = kstep_pre.numpy().flatten()
    # = x.flatten()
    n_plot = T
    plt.plot(range(n_plot), x[:n_plot],
             'bo-', label='True', alpha=0.7)
    plt.plot(range(n_plot), kstep_pre_flat[:n_plot],
             'rs--', label='predict', alpha=0.7)

    plt.xlabel('sample')
    plt.ylabel('number')
    plt.title('predict and true')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # ----------------数据准备------------
    T = 1000
    time = tf.range(1, T + 1, dtype=tf.float32)
    x_have_noise = tf.sin(0.01 * time) + tf.random.normal([T], 0, 0.2) #加入了随机噪声的数据
    x_no_noise = tf.sin(0.01 * time) #没有噪声的数据

    x = x_no_noise #可以选择是否有噪声来观察模型结果

    tau = 128 #tau值可以自己调节，数学意义为以过去几个的数据来预测下一个的 建议依次设定为4，32，64，128测试不同的效果
    features = tf.Variable(tf.zeros((T - tau, tau)))
    for i in range(tau):
        features[:, i].assign(x[i: T - tau + i])

    # 这是标签,简单来说就是features的某一行，比如第一行其实是x[1],x[2],x[3],x[4],而laberls此时对应的数字是X[5]
    # 用前四个预测下一个
    labels = tf.reshape(x[tau:], (-1, 1))

    n_train = 600
    train_features = features[:n_train]
    train_labels = labels[:n_train]
    test_features = features[n_train:]
    test_labels = labels[n_train:]
    # ------------------------------------------
    #得到训练好的模型
    model = model_net(net,train_features,train_labels,test_features, test_labels)

    #单步预测
    draw_onestep(model,train_features,train_labels)

    #k步预测
    kstep_pre(model,T,n_train,x,tau=tau)


