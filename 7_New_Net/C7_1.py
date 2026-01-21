import tensorflow as tf

def AlexNet():
    return tf.keras.models.Sequential([
        # 这里使用一个11*11的更大窗口来捕捉对象。
        # 同时，步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        tf.keras.layers.Dense(10)
    ])

#定义VCG块
def VGG_block(num_convs,num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(num_channels, kernel_size=3, padding='same', activation='relu')
        )
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
def VGG(conv_arch):
    #conv_arch示例，conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = tf.keras.models.Sequential()
    #卷积层部分
    for (nums_conv,num_channels) in conv_arch:
        net.add(VGG_block(nums_conv,num_channels))
    #MLP部分
    net.add(
        tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10)
        ]
    )
    )
    return net

def NiN_block(num_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(num_channels,kernel_size,strides=strides,padding=padding,activation='relu'),
            #接两个1*1的卷积层
            tf.keras.layers.Conv2D(num_channels, kernel_size=1, activation='relu'),
            tf.keras.layers.Conv2D(num_channels, kernel_size=1, activation='relu')
        ]
    )

def NiN():
    #NiN去除了全连接层
    return tf.keras.models.Sequential(
        [
            NiN_block(96, kernel_size=11, strides=4, padding='valid'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            NiN_block(256, kernel_size=5, strides=1, padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            NiN_block(384, kernel_size=3, strides=1, padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            tf.keras.layers.Dropout(0.5),
            # 标签类别数是10
            NiN_block(10, kernel_size=3, strides=1, padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Reshape((1, 1, 10)),
            # 将四维的输出转成二维的输出，其形状为(批量大小,10)
            tf.keras.layers.Flatten(),
        ]
    )