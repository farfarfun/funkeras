from notekeras.model.vgg.vgg16 import VGG16
from notekeras.model.vgg.vgg19 import VGG19

#from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input

from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import Model

# 使用 VGG16 提取特征


def vgg16_test():
    model = VGG16(weights='imagenet', include_top=False)

    img_path = '/root/workspace/notechats/notekeras/example/yolo/data/images/kite.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    print(features)


def vgg19_test():
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('block4_pool').output)

    img_path = '/root/workspace/notechats/notekeras/example/yolo/data/images/kite.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    block4_pool_features = model.predict(x)
    print(block4_pool_features)


vgg16_test()
# vgg19_test()
