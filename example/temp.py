import numpy as np
import tensorflow as tf

features, labels = (np.random.sample((6, 3)),  # 模拟6组数据，每组数据3个特征
                    np.random.sample((6, 1)))  # 模拟6组数据，每组数据对应一个标签，注意两者的维数必须匹配

print((features, labels))  # 输出下组合的数据
data = tf.data.Dataset.from_tensor_slices((features, labels))
print(data)  # 输出张量的信息
