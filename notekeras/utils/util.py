import os
from collections import namedtuple
from functools import reduce

import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from notekeras.backend import backend as K
from notekeras.backend import keras
from notekeras.layers import Extract, MaskedGlobalMaxPool1D
from notekeras.models.loader import (load_trained_model_from_checkpoint,
                                     load_vocabulary)
from notekeras.tokenizer import Tokenizer
from PIL import Image

__all__ = [
    'POOL_NSP', 'POOL_MAX', 'POOL_AVE',
    'get_checkpoint_paths', 'extract_embeddings_generator', 'extract_embeddings', 'compose', 'get_random_data'
]

POOL_NSP = 'POOL_NSP'
POOL_MAX = 'POOL_MAX'
POOL_AVE = 'POOL_AVE'


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line,
                    input_shape,
                    random=True,
                    max_boxes=20,
                    jitter=.3,
                    hue=.1,
                    sat=1.5,
                    val=1.5,
                    proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes: box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    函数嵌套 compose(f,g,k) = k(g(f))
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def get_checkpoint_paths(model_path):
    CheckpointPaths = namedtuple('CheckpointPaths', ['config', 'checkpoint', 'vocab'])
    config_path = os.path.join(model_path, 'bert_config.json')
    checkpoint_path = os.path.join(model_path, 'bert_model.ckpt')
    vocab_path = os.path.join(model_path, 'vocab.txt')
    return CheckpointPaths(config_path, checkpoint_path, vocab_path)


def extract_embeddings_generator(model,
                                 texts,
                                 poolings=None,
                                 vocabs=None,
                                 cased=False,
                                 batch_size=4,
                                 cut_embed=True,
                                 output_layer_num=1):
    """Extract embeddings from texts.

    :param model: Path to the checkpoint or built model without MLM and NSP.
    :param texts: Iterable texts.
    :param poolings: Pooling methods. Word embeddings will be returned if it is None.
                     Otherwise concatenated pooled embeddings will be returned.
    :param vocabs: A dict should be provided if model is built.
    :param cased: Whether it is cased for tokenizer.
    :param batch_size: Batch size.
    :param cut_embed: The computed embeddings will be cut based on their input lengths.
    :param output_layer_num: The number of layers whose outputs will be concatenated as a single output.
                             Only available when `model` is a path to checkpoint.
    :return: A list of numpy arrays representing the embeddings.
    """
    if isinstance(model, (str, type(u''))):
        paths = get_checkpoint_paths(model)
        model = load_trained_model_from_checkpoint(
            config_file=paths.config,
            checkpoint_file=paths.checkpoint,
            output_layer_num=output_layer_num,
        )
        vocabs = load_vocabulary(paths.vocab)

    seq_len = K.int_shape(model.outputs[0])[1]
    tokenizer = Tokenizer(vocabs, cased=cased)

    def _batch_generator():
        tokens, segments = [], []

        def _pad_inputs():
            if seq_len is None:
                max_len = max(map(len, tokens))
                for i in range(len(tokens)):
                    tokens[i].extend([0] * (max_len - len(tokens[i])))
                    segments[i].extend([0] * (max_len - len(segments[i])))
            return [np.array(tokens), np.array(segments)]

        for text in texts:
            if isinstance(text, (str, type(u''))):
                token, segment = tokenizer.encode(text, max_len=seq_len)
            else:
                token, segment = tokenizer.encode(text[0], text[1], max_len=seq_len)
            tokens.append(token)
            segments.append(segment)
            if len(tokens) == batch_size:
                yield _pad_inputs()
                tokens, segments = [], []
        if len(tokens) > 0:
            yield _pad_inputs()

    if poolings is not None:
        if isinstance(poolings, (str, type(u''))):
            poolings = [poolings]
        outputs = []
        for pooling in poolings:
            if pooling == POOL_NSP:
                outputs.append(Extract(index=0, name='Pool-NSP')(model.outputs[0]))
            elif pooling == POOL_MAX:
                outputs.append(MaskedGlobalMaxPool1D(name='Pool-Max')(model.outputs[0]))
            elif pooling == POOL_AVE:
                outputs.append(keras.layers.GlobalAvgPool1D(name='Pool-Ave')(model.outputs[0]))
            else:
                raise ValueError('Unknown pooling method: {}'.format(pooling))
        if len(outputs) == 1:
            outputs = outputs[0]
        else:
            outputs = keras.layers.Concatenate(name='Concatenate')(outputs)
        model = keras.models.Model(inputs=model.inputs, outputs=outputs)

    for batch_inputs in _batch_generator():
        outputs = model.predict(batch_inputs)
        for inputs, output in zip(batch_inputs[0], outputs):
            if poolings is None and cut_embed:
                length = 0
                for i in range(len(inputs) - 1, -1, -1):
                    if inputs[i] != 0:
                        length = i + 1
                        break
                output = output[:length]
            yield output


def extract_embeddings(model,
                       texts,
                       poolings=None,
                       vocabs=None,
                       cased=False,
                       batch_size=4,
                       cut_embed=True,
                       output_layer_num=1):
    """Extract embeddings from texts.

    :param model: Path to the checkpoint or built model without MLM and NSP.
    :param texts: Iterable texts.
    :param poolings: Pooling methods. Word embeddings will be returned if it is None.
                     Otherwise concatenated pooled embeddings will be returned.
    :param vocabs: A dict should be provided if model is built.
    :param cased: Whether it is cased for tokenizer.
    :param batch_size: Batch size.
    :param cut_embed: The computed embeddings will be cut based on their input lengths.
    :param output_layer_num: The number of layers whose outputs will be concatenated as a single output.
                             Only available when `model` is a path to checkpoint.
    :return: A list of numpy arrays representing the embeddings.
    """
    return [embedding for embedding in extract_embeddings_generator(
        model, texts, poolings, vocabs, cased, batch_size, cut_embed, output_layer_num
    )]
