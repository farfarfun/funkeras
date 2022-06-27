import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image

import notekeras.model.resnet as res
from notekeras.model.temp import preprocess_input


def run1():
    model = ResNet50(include_top=False)
    img = image.load_img('./image/man.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, version=1)  # or version=2
    preds = model.predict(x)
    print('Predicted:', preds)


def run2():
    model = res.ResNet50(include_top=False)
    model.load_weight()

    img = image.load_img('./image/man.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, version=1)  # or version=2
    preds = model.predict(x)
    print('Predicted:', preds)


run1()
run2()
