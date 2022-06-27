


## ScaledDotProductAttention
[参考论文](https://arxiv.org/pdf/1706.03762.pdf)

$$\text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V$$

## SeqSelfAttention
[参考论文](https://arxiv.org/pdf/1806.01264.pdf)

### multiplicative
$$e_{t, t'} = x_t^T W_a x_{t'} + b_a$$

$$a_{t} = \text{softmax}(e_t)$$

$$l_t = \sum_{t'} a_{t, t'} x_{t'}$$

### additive
$$h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)$$

$$e_{t, t'} = W_a h_{t, t'} + b_a$$

$$a_{t} = \text{softmax}(e_t)$$

$$l_t = \sum_{t'} a_{t, t'} x_{t'}$$


## SeqWeightedAttention
[参考论文](https://arxiv.org/pdf/1708.00524.pdf)

$$Y = \text{softmax}(XW + b) X$$


## MultiHeadAttention
[参考论文](https://arxiv.org/pdf/1706.03762.pdf)




