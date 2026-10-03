import os
from pathlib import Path

from funkeras.models.vgg.vgg16 import VGG16
from funkeras.models.vgg.vgg19 import VGG19

#from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input

from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import Model

# 使用 VGG16 提取特征

# 默认使用仓库自带的示例图片（example/yolo/data/images/kite.jpg），可通过
# FUNKERAS_VGG_IMAGE 环境变量覆盖，避免依赖作者本机路径。
DEFAULT_IMAGE_PATH = Path(__file__).resolve().parents[1] / "yolo/data/images/kite.jpg"
IMAGE_PATH = os.environ.get("FUNKERAS_VGG_IMAGE", str(DEFAULT_IMAGE_PATH))


def vgg16_test():
    model = VGG16(weights='imagenet', include_top=False)

    img = image.load_img(IMAGE_PATH, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    print(features)


def vgg19_test():
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('block4_pool').output)

    img = image.load_img(IMAGE_PATH, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    block4_pool_features = model.predict(x)
    print(block4_pool_features)


if __name__ == '__main__':
    vgg16_test()
    # vgg19_test()
