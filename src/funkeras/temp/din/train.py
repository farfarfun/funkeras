import os
import pickle

import numpy as np
from farlog import getLogger
from tqdm import tqdm

from funkeras.backend import keras
from funkeras.temp.din.data_generator import DataInput, TestData
from funkeras.temp.din.model import din

logger = getLogger("funkeras")

Callback = keras.callbacks.Callback

# 没有可移植的默认值（依赖本地预处理好的电商数据集），通过环境变量传入，
# 避免在 import 阶段就依赖作者本机路径；默认值仅用于提示。
data_root = os.environ.get(
    "FUNKERAS_DIN_DATA_ROOT", os.path.join(os.getcwd(), "data", "electronics")
)

batch_size = 64

# 以下全局变量在 main() 中赋值，模块导入阶段不触发任何 I/O 或训练。
train_model = None
label_model = None
train_D = None
test_D = None
evaluator = None


# 定义sigmoid函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def calc_auc(raw_arr):
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d: d[2])
    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0]  # noclick
        tp2 += record[1]  # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2
    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5
    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None


# 定义回调函数
class Evaluate(Callback):
    """回调评估和保存模型"""

    def __init__(self):
        self.acc = []
        self.best_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.acc.append(acc)
        if acc > self.best_acc:
            self.best_acc = acc
            train_model.save_weights("./best_model.weight")
        logger.info('acc: {:.4f}, best acc: {:.4f}'.format(acc, self.best_acc))

    def evaluate(self):
        score = []  # 记录实际和预测的结果
        # 取一个batch*20的数据
        np.random.shuffle(test_D.test_set)
        batch_valid_data = test_D.test_set[:(batch_size * 20)]
        for row in tqdm(batch_valid_data):
            label = row[-1]
            logits = label_model.predict(row[:-1])  # (batch_size, 1)  array([[0.3211818]], dtype=float32)
            pred = sigmoid(logits)[0][0]

            if label > 0.5:
                score.append([0, 1, pred])
            else:
                score.append([1, 0, pred])

        # 计算AUC
        auc = calc_auc(score)
        logger.info("TEST --> auc: {}".format(auc))
        return auc


def main() -> None:
    """构建 DIN 模型并在 ``data_root`` 指向的本地数据集上训练。

    数据集路径来自环境变量 ``FUNKERAS_DIN_DATA_ROOT``（未设置时默认为
    ``<当前目录>/data/electronics``，该目录通常不存在，需要用户预先准备好
    `din/README.md` 描述的电商数据集），不会在模块导入阶段触发任何 I/O。
    """
    global train_model, label_model, train_D, test_D, evaluator

    train_model, label_model = din(item_count=63001, cate_count=801, hidden_units=128)
    keras.utils.plot_model(train_model, 'model.png', show_shapes=True)

    with open('{}/raw_data/dataset.pkl'.format(data_root), 'rb') as f:
        pickle.load(f)  # train_set
        pickle.load(f)  # test_set
        pickle.load(f)  # cate_list
        pickle.load(f)  # user_count, item_count, cate_count

    train_D = DataInput(file="{}/paddle_train.txt".format(data_root), batch_size=batch_size)
    test_D = TestData(file="{}/paddle_test.txt".format(data_root))

    evaluator = Evaluate()

    # 定义ModelCheckpoint、EarlyStopping和TensorBoard
    # Using tensorboard callbacks
    # tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    train_model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=len(train_D),
                              callbacks=[evaluator],
                              epochs=2)


if __name__ == "__main__":
    main()
