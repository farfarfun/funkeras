from notekeras.backend import keras


# 自定义 Model 的方法写线性回归
class Linear(keras.layers.Layer):
    def __init__(self, input_shape):
        super(Linear, self).__init__()
        # Dense: 全连接层
        self.name = "liner-001"
        self.dense = keras.layers.Dense(name="line-dense1",
                                        units=32,  # 输出张量的维度
                                        activation=None,  # 激活函数, 包括 tf.nn.relu 、 tf.nn.tanh 和 tf.nn.sigmoid
                                        )
        self.dense2 = keras.layers.Dense(name="line-dense2",
                                         units=32,  # 输出张量的维度
                                         activation=None,  # 激活函数, 包括 tf.nn.relu 、 tf.nn.tanh 和 tf.nn.sigmoid
                                         )

    def build(self, input_shape):
        super(Linear, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input, **kwargs):
        output = self.dense(input)
        output = self.dense2(output)
        return output


# define documents
docs = ['Well done!', 'Good work', 'Great effort', 'nice work', 'Excellent!',
        'Weak', 'Poor effort!', 'not good', 'poor work', 'Could have done better.'
        ]
# define class labels
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
# integer encode the documents
vocab_size = 50
encoded_docs = [keras.preprocessing.text.one_hot(d, vocab_size) for d in docs]
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
# define the model

model = keras.models.Sequential()
model.add(keras.layers.embeddings.Embedding(vocab_size, 8, input_length=max_length))
model.add(keras.layers.Flatten())
# model.add(Linear())
model.add(keras.layers.Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))

vocab = keras.Input(shape=(4,), name='input_inner')
emb = keras.layers.embeddings.Embedding(vocab_size, 8, input_length=max_length)(vocab)
flat = keras.layers.Flatten()(emb)
liner = Linear((32,))(flat)

d1 = keras.layers.Dense(1, activation='sigmoid')(liner)

model2 = keras.models.Model([vocab], d1)

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model2.summary())
# fit the model
model2.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model2.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))
