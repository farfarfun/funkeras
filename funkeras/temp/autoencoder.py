import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.dense_1 = Dense(128, activation='tanh')
        self.dense_2 = Dense(64, activation='tanh')
        self.dense_3 = Dense(32, activation='tanh')
        self.dense_4 = Dense(2, activation='sigmoid')

    def call(self, inputs, training=False, **kwargs):
        output = self.dense_1(inputs)
        output = self.dense_2(output)
        output = self.dense_3(output)
        output = self.dense_4(output)
        return output


class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense_1 = Dense(16, activation='tanh')
        self.dense_2 = Dense(64, activation='tanh')
        self.dense_3 = Dense(128, activation='tanh')
        self.dense_4 = Dense(784, activation='sigmoid')

    def call(self, inputs, training=False, **kwargs):
        output = self.dense_1(inputs)
        output = self.dense_2(output)
        output = self.dense_3(output)
        output = self.dense_4(output)
        return output


class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs, training=False, **kwargs):
        output = self.encoder(inputs, training)
        output = self.decoder(output, training)
        return output


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = np.reshape(X_train, (-1, 784)) / 255.
batch_size = 512
optimizer = tf.keras.optimizers.Adam(lr=0.001)
model = AutoEncoder()
sample = np.reshape(X_test[:5], (5, 784))

for step in range(10000):
    true_image = X_train[np.random.choice(X_train.shape[0], batch_size)]
    with tf.GradientTape() as tape:
        pred_image = model(true_image, training=True)
        loss = tf.reduce_mean(tf.square(pred_image - true_image))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if step % 200 == 0: print("=> %4d loss %.4f" % (step, loss))
    if step % 1000 == 0:
        pred_image = model(sample, training=False)
        pred_image = np.reshape(pred_image, (5, 28, 28))
        show_image = np.concatenate(pred_image[:5], -1)
        plt.tight_layout()
        plt.imshow(show_image)
        plt.savefig("%d.png" % ((step + 1) // 2000))

"""
visualize embedding in 2D
"""
sample = np.reshape(X_test[:5000], (5000, 784))
label = y_test[:5000]
embeddings = model.encoder(sample, training=False)

fig, ax = plt.subplots()
X, Y = embeddings[:, 0].numpy(), embeddings[:, 1].numpy()
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
for x, y, l in zip(X, Y, label):
    c = cm.rainbow(int(255 * l / 9))
    ax.text(x, y, l, color=c)
    # plt.plot(x,y, '.', c=c)
plt.axis('off')
plt.legend()
plt.tight_layout()
plt.savefig("embedding.png")
plt.show()
