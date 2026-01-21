import tensorflow as tf
import numpy as np
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)

net.save_weights('mlp.weights.h5')

clone = MLP()
clone(X)
clone.load_weights('mlp.weights.h5')
Y_clone = clone(X)
print(Y == Y_clone)

x = tf.constant([1, 2, 3])
print(x.device