import numpy as np

from notekeras.model.loader import get_checkpoint_config
from notekeras.model.loader import load_trained_model_from_checkpoint, PreTrainedList, load_vocabulary
from notekeras.tokenizer import Tokenizer

config = get_checkpoint_config(info=PreTrainedList.chinese_base)
model = load_trained_model_from_checkpoint(config.config, config.checkpoint, training=True, seq_len=None)

model.summary(line_length=120)

token_dict = load_vocabulary(config.vocab)
token_dict_inv = {v: k for k, v in token_dict.items()}

tokenizer = Tokenizer(token_dict)
text = '数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科'
tokens = tokenizer.tokenize(text)
tokens[1] = tokens[2] = '[MASK]'
print('Tokens:', tokens)

indices = np.array([[token_dict[token] for token in tokens]])
segments = np.array([[0] * len(tokens)])
masks = np.array([[0, 1, 1] + [0] * (len(tokens) - 3)])

predicts = model.predict([indices, segments, masks])[0].argmax(axis=-1).tolist()
print('Fill with: ', list(map(lambda x: token_dict_inv[x], predicts[0][1:3])))

sentence_1 = '数学是利用符号语言研究數量、结构、变化以及空间等概念的一門学科。'
sentence_2 = '从某种角度看屬於形式科學的一種。'
print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))
indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2)
masks = np.array([[0] * len(indices)])

predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]
print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))

sentence_2 = '任何一个希尔伯特空间都有一族标准正交基。'
print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))
indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2)
masks = np.array([[0] * len(indices)])

predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]
print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))
