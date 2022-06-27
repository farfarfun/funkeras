import tensorflow as tf
from notekeras import ops
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import Dense, Layer

from .utils import (deserialize_kwarg, is_keras_kwarg, is_layer_kwarg,
                    serialize_kwarg)


class Pool(Layer):
    r"""
    A general class for pooling layers.

    You can extend this class to create custom implementations of pooling layers.

    Any extension of this class must implement the `call(self, inputs)` and
    `config(self)` methods.

    **Arguments**:

    - ``**kwargs`: additional keyword arguments specific to Keras' Layers, like
    regularizers, initializers, constraints, etc.
    """

    def __init__(self,
                 activation=None,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.bias = None

        kwargs.update({
            activation: activation,
            use_bias: use_bias,
            kernel_initializer: kernel_initializer,
            bias_initializer: bias_initializer,
            kernel_regularizer: kernel_regularizer,
            bias_regularizer: bias_regularizer,
            activity_regularizer: activity_regularizer,
            kernel_constraint: kernel_constraint,
            bias_constraint: bias_constraint,
        })
        super().__init__(
            **{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})
        self.kwargs_keys = []
        for key in kwargs:
            if is_layer_kwarg(key):
                attr = kwargs[key]
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        keras_config = {}
        for key in self.kwargs_keys:
            keras_config[key] = serialize_kwarg(key, getattr(self, key))
        return {**base_config, **keras_config, **self.config}

    @property
    def config(self):
        return {}


class DiffPool(Pool):
    r"""
    A DiffPool layer from the paper

    > [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804)<br>
    > Rex Ying et al.

    **Mode**: batch.

    This layer computes a soft clustering \(\S\) of the input graphs using a GNN,
    and reduces graphs as follows:
    $$
        \S = \textrm{GNN}(\A, \X); \\
        \A' = \S^\top \A \S; \X' = \S^\top \X;
    $$

    where GNN consists of one GraphConv layer with softmax activation.
    Two auxiliary loss terms are also added to the model: the _link prediction
    loss_
    $$
        \big\| \A - \S\S^\top \big\|_F
    $$
    and the _entropy loss_
    $$
        - \frac{1}{N} \sum\limits_{i = 1}^{N} \S \log (\S).
    $$

    The layer also applies a 1-layer GCN to the input features, and returns
    the updated graph signal (the number of output channels is controlled by
    the `channels` parameter).
    The layer can be used without a supervised loss, to compute node clustering
    simply by minimizing the two auxiliary losses.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `([batch], n_nodes, n_nodes)`;

    **Output**

    - Reduced node features of shape `([batch], K, channels)`;
    - Reduced adjacency matrix of shape `([batch], K, K)`;
    - If `return_mask=True`, the soft clustering matrix of shape `([batch], n_nodes, K)`.

    **Arguments**

    - `k`: number of nodes to keep;
    - `channels`: number of output channels (if None, the number of output
    channels is assumed to be the same as the input);
    - `return_mask`: boolean, whether to return the cluster assignment matrix;
    - `kernel_initializer`: initializer for the weights;
    - `kernel_regularizer`: regularization applied to the weights;
    - `kernel_constraint`: constraint applied to the weights;
    """

    def __init__(
        self,
        k,
        channels=None,
        return_mask=False,
        activation=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):

        super().__init__(
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs
        )
        self.k = k
        self.channels = channels
        self.return_mask = return_mask

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        F = input_shape[0][-1]

        if self.channels is None:
            self.channels = F

        self.kernel_emb = self.add_weight(
            shape=(F, self.channels),
            name="kernel_emb",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        self.kernel_pool = self.add_weight(
            shape=(F, self.k),
            name="kernel_pool",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        super().build(input_shape)

    def call(self, inputs):
        if len(inputs) == 3:
            X, A, I = inputs
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X, A = inputs
            I = None

        N = K.shape(A)[-1]
        # Check if the layer is operating in mixed or batch mode
        mode = ops.autodetect_mode(X, A)
        self.reduce_loss = mode in (ops.MIXED, ops.BATCH)

        # Get normalized adjacency
        if K.is_sparse(A):
            I_ = tf.sparse.eye(N, dtype=A.dtype)
            A_ = tf.sparse.add(A, I_)
        else:
            I_ = tf.eye(N, dtype=A.dtype)
            A_ = A + I_
        fltr = ops.normalize_A(A_)

        # Node embeddings
        Z = K.dot(X, self.kernel_emb)
        Z = ops.modal_dot(fltr, Z)
        if self.activation is not None:
            Z = self.activation(Z)

        # Compute cluster assignment matrix
        S = K.dot(X, self.kernel_pool)
        S = ops.modal_dot(fltr, S)
        S = activations.softmax(S, axis=-1)  # softmax applied row-wise

        # Link prediction loss
        S_gram = ops.modal_dot(S, S, transpose_b=True)
        if mode == ops.MIXED:
            A = tf.sparse.to_dense(A)[None, ...]
        if K.is_sparse(A):
            # A/tf.norm(A) - S_gram/tf.norm(S_gram)
            LP_loss = tf.sparse.add(A, -S_gram)
        else:
            LP_loss = A - S_gram
        LP_loss = tf.norm(LP_loss, axis=(-1, -2))
        if self.reduce_loss:
            LP_loss = K.mean(LP_loss)
        self.add_loss(LP_loss)

        # Entropy loss
        entr = tf.negative(
            tf.reduce_sum(tf.multiply(S, K.log(S + K.epsilon())), axis=-1)
        )
        entr_loss = K.mean(entr, axis=-1)
        if self.reduce_loss:
            entr_loss = K.mean(entr_loss)
        self.add_loss(entr_loss)

        # Pooling
        X_pooled = ops.modal_dot(S, Z, transpose_a=True)
        A_pooled = ops.matmul_at_b_a(S, A)

        output = [X_pooled, A_pooled]

        if I is not None:
            I_mean = tf.math.segment_mean(I, I)
            I_pooled = ops.repeat(I_mean, tf.ones_like(I_mean) * self.k)
            output.append(I_pooled)

        if self.return_mask:
            output.append(S)

        return output

    @property
    def config(self):
        return {
            "k": self.k,
            "channels": self.channels,
            "return_mask": self.return_mask,
        }


class GlobalPool(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        self.pooling_op = None
        self.batch_pooling_op = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = "disjoint"
        else:
            if len(input_shape) == 2:
                self.data_mode = "single"
            else:
                self.data_mode = "batch"
        super().build(input_shape)

    def call(self, inputs):
        if self.data_mode == "disjoint":
            X = inputs[0]
            I = inputs[1]
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X = inputs

        if self.data_mode == "disjoint":
            return self.pooling_op(X, I)
        else:
            return self.batch_pooling_op(
                X, axis=-2, keepdims=(self.data_mode == "single")
            )

    def compute_output_shape(self, input_shape):
        if self.data_mode == "single":
            return (1,) + input_shape[-1:]
        elif self.data_mode == "batch":
            return input_shape[:-2] + input_shape[-1:]
        else:
            # Input shape is a list of shapes for X and I
            return input_shape[0]

    def get_config(self):
        return super().get_config()


class GlobalSumPool(GlobalPool):
    """
    A global sum pooling layer. Pools a graph by computing the sum of its node
    features.

    **Mode**: single, disjoint, mixed, batch.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);

    **Output**

    - Pooled node features of shape `(batch, n_node_features)` (if single mode, shape will
    be `(1, n_node_features)`).

    **Arguments**

    None.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pooling_op = tf.math.segment_sum
        self.batch_pooling_op = tf.reduce_sum


class GlobalAvgPool(GlobalPool):
    """
    An average pooling layer. Pools a graph by computing the average of its node
    features.

    **Mode**: single, disjoint, mixed, batch.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);

    **Output**

    - Pooled node features of shape `(batch, n_node_features)` (if single mode, shape will
    be `(1, n_node_features)`).

    **Arguments**

    None.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pooling_op = tf.math.segment_mean
        self.batch_pooling_op = tf.reduce_mean


class GlobalMaxPool(GlobalPool):
    """
    A max pooling layer. Pools a graph by computing the maximum of its node
    features.

    **Mode**: single, disjoint, mixed, batch.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);

    **Output**

    - Pooled node features of shape `(batch, n_node_features)` (if single mode, shape will
    be `(1, n_node_features)`).

    **Arguments**

    None.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pooling_op = tf.math.segment_max
        self.batch_pooling_op = tf.reduce_max


class GlobalAttentionPool(GlobalPool):
    r"""
    A gated attention global pooling layer from the paper

    > [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493)<br>
    > Yujia Li et al.

    This layer computes:
    $$
        \X' = \sum\limits_{i=1}^{N} (\sigma(\X \W_1 + \b_1) \odot (\X \W_2 + \b_2))_i
    $$
    where \(\sigma\) is the sigmoid activation function.

    **Mode**: single, disjoint, mixed, batch.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);

    **Output**

    - Pooled node features of shape `(batch, channels)` (if single mode,
    shape will be `(1, channels)`).

    **Arguments**

    - `channels`: integer, number of output channels;
    - `bias_initializer`: initializer for the bias vectors;
    - `kernel_regularizer`: regularization applied to the kernel matrices;
    - `bias_regularizer`: regularization applied to the bias vectors;
    - `kernel_constraint`: constraint applied to the kernel matrices;
    - `bias_constraint`: constraint applied to the bias vectors.
    """

    def __init__(
        self,
        channels,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        super().build(input_shape)
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )
        self.features_layer = Dense(
            self.channels, name="features_layer", **layer_kwargs
        )
        self.attention_layer = Dense(
            self.channels, activation="sigmoid", name="attn_layer", **layer_kwargs
        )
        self.built = True

    def call(self, inputs):
        if self.data_mode == "disjoint":
            X, I = inputs
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X = inputs
        inputs_linear = self.features_layer(X)
        attn = self.attention_layer(X)
        masked_inputs = inputs_linear * attn
        if self.data_mode in {"single", "batch"}:
            output = K.sum(masked_inputs, axis=-2,
                           keepdims=self.data_mode == "single")
        else:
            output = tf.math.segment_sum(masked_inputs, I)

        return output

    def compute_output_shape(self, input_shape):
        if self.data_mode == "single":
            return (1,) + (self.channels,)
        elif self.data_mode == "batch":
            return input_shape[:-2] + (self.channels,)
        else:
            output_shape = input_shape[0]
            output_shape = output_shape[:-1] + (self.channels,)
            return output_shape

    def get_config(self):
        config = {
            "channels": self.channels,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "bias_constraint": self.bias_constraint,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalAttnSumPool(GlobalPool):
    r"""
    A node-attention global pooling layer. Pools a graph by learning attention
    coefficients to sum node features.

    This layer computes:
    $$
        \alpha = \textrm{softmax}( \X \a); \\
        \X' = \sum\limits_{i=1}^{N} \alpha_i \cdot \X_i
    $$
    where \(\a \in \mathbb{R}^F\) is a trainable vector. Note that the softmax
    is applied across nodes, and not across features.

    **Mode**: single, disjoint, mixed, batch.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);

    **Output**

    - Pooled node features of shape `(batch, n_node_features)` (if single mode, shape will
    be `(1, n_node_features)`).

    **Arguments**

    - `attn_kernel_initializer`: initializer for the attention weights;
    - `attn_kernel_regularizer`: regularization applied to the attention kernel
    matrix;
    - `attn_kernel_constraint`: constraint applied to the attention kernel
    matrix;
    """

    def __init__(
        self,
        attn_kernel_initializer="glorot_uniform",
        attn_kernel_regularizer=None,
        attn_kernel_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.attn_kernel_initializer = initializers.get(
            attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(
            attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = "disjoint"
            F = input_shape[0][-1]
        else:
            if len(input_shape) == 2:
                self.data_mode = "single"
            else:
                self.data_mode = "batch"
            F = input_shape[-1]
        # Attention kernels
        self.attn_kernel = self.add_weight(
            shape=(F, 1),
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
            name="attn_kernel",
        )
        self.built = True

    def call(self, inputs):
        if self.data_mode == "disjoint":
            X, I = inputs
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X = inputs
        attn_coeff = K.dot(X, self.attn_kernel)
        attn_coeff = K.squeeze(attn_coeff, -1)
        attn_coeff = K.softmax(attn_coeff)
        if self.data_mode == "single":
            output = K.dot(attn_coeff[None, ...], X)
        elif self.data_mode == "batch":
            output = K.batch_dot(attn_coeff, X)
        else:
            output = attn_coeff[:, None] * X
            output = tf.math.segment_sum(output, I)

        return output

    def get_config(self):
        config = {
            "attn_kernel_initializer": self.attn_kernel_initializer,
            "attn_kernel_regularizer": self.attn_kernel_regularizer,
            "attn_kernel_constraint": self.attn_kernel_constraint,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SortPool(Layer):
    r"""
    A SortPool layer as described by
    [Zhang et al](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf).
    This layers takes a graph signal \(\mathbf{X}\) and returns the topmost k
    rows according to the last column.
    If \(\mathbf{X}\) has less than k rows, the result is zero-padded to k.

    **Mode**: single, disjoint, batch.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);

    **Output**

    - Pooled node features of shape `(batch, k, n_node_features)` (if single mode, shape will
    be `(1, k, n_node_features)`).

    **Arguments**

    - `k`: integer, number of nodes to keep;
    """

    def __init__(self, k):
        super(SortPool, self).__init__()
        k = int(k)
        if k <= 0:
            raise ValueError("K must be a positive integer")
        self.k = k

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = "disjoint"
            self.F = input_shape[0][-1]
        else:
            if len(input_shape) == 2:
                self.data_mode = "single"
            else:
                self.data_mode = "batch"
            self.F = input_shape[-1]

    def call(self, inputs):
        if self.data_mode == "disjoint":
            X, I = inputs
            X = ops.disjoint_signal_to_batch(X, I)
        else:
            X = inputs
            if self.data_mode == "single":
                X = tf.expand_dims(X, 0)

        N = tf.shape(X)[-2]
        sort_perm = tf.argsort(X[..., -1], direction="DESCENDING")
        X_sorted = tf.gather(X, sort_perm, axis=-2, batch_dims=1)

        def truncate():
            _X_out = X_sorted[..., : self.k, :]
            return _X_out

        def pad():
            padding = [[0, 0], [0, self.k - N], [0, 0]]
            _X_out = tf.pad(X_sorted, padding)
            return _X_out

        X_out = tf.cond(tf.less_equal(self.k, N), truncate, pad)

        if self.data_mode == "single":
            X_out = tf.squeeze(X_out, [0])
            X_out.set_shape((self.k, self.F))
        elif self.data_mode == "batch" or self.data_mode == "disjoint":
            X_out.set_shape((None, self.k, self.F))

        return X_out

    def get_config(self):
        config = {
            "k": self.k,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if self.data_mode == "single":
            return self.k, self.F
        elif self.data_mode == "batch" or self.data_mode == "disjoint":
            return input_shape[0], self.k, self.F


layers = {
    "sum": GlobalSumPool,
    "avg": GlobalAvgPool,
    "max": GlobalMaxPool,
    "attn": GlobalAttentionPool,
    "attn_sum": GlobalAttnSumPool,
    "sort": SortPool,
}


def get(identifier):
    if identifier not in layers:
        raise ValueError(
            "Unknown identifier {}. Available: {}".format(
                identifier, list(layers.keys())
            )
        )
    else:
        return layers[identifier]


class TopKPool(Pool):
    r"""
    A gPool/Top-K layer from the papers

    > [Graph U-Nets](https://arxiv.org/abs/1905.05178)<br>
    > Hongyang Gao and Shuiwang Ji

    and

    > [Towards Sparse Hierarchical Graph Classifiers](https://arxiv.org/abs/1811.01287)<br>
    > Cătălina Cangea et al.

    **Mode**: single, disjoint.

    This layer computes the following operations:
    $$
        \y = \frac{\X\p}{\|\p\|}; \;\;\;\;
        \i = \textrm{rank}(\y, K); \;\;\;\;
        \X' = (\X \odot \textrm{tanh}(\y))_\i; \;\;\;\;
        \A' = \A_{\i, \i}
    $$

    where \( \textrm{rank}(\y, K) \) returns the indices of the top K values of
    \(\y\), and \(\p\) is a learnable parameter vector of size \(F\). \(K\) is
    defined for each graph as a fraction of the number of nodes.
    Note that the the gating operation \(\textrm{tanh}(\y)\) (Cangea et al.)
    can be replaced with a sigmoid (Gao & Ji).

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);

    **Output**

    - Reduced node features of shape `(ratio * n_nodes, n_node_features)`;
    - Reduced adjacency matrix of shape `(ratio * n_nodes, ratio * n_nodes)`;
    - Reduced graph IDs of shape `(ratio * n_nodes, )` (only in disjoint mode);
    - If `return_mask=True`, the binary pooling mask of shape `(ratio * n_nodes, )`.

    **Arguments**

    - `ratio`: float between 0 and 1, ratio of nodes to keep in each graph;
    - `return_mask`: boolean, whether to return the binary mask used for pooling;
    - `sigmoid_gating`: boolean, use a sigmoid gating activation instead of a
        tanh;
    - `kernel_initializer`: initializer for the weights;
    - `kernel_regularizer`: regularization applied to the weights;
    - `kernel_constraint`: constraint applied to the weights;
    """

    def __init__(
        self,
        ratio,
        return_mask=False,
        sigmoid_gating=False,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):
        super().__init__(
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs
        )
        self.ratio = ratio
        self.return_mask = return_mask
        self.sigmoid_gating = sigmoid_gating
        self.gating_op = K.sigmoid if self.sigmoid_gating else K.tanh

    def build(self, input_shape):
        self.F = input_shape[0][-1]
        self.N = input_shape[0][0]
        self.kernel = self.add_weight(
            shape=(self.F, 1),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        super().build(input_shape)

    def call(self, inputs):
        if len(inputs) == 3:
            X, A, I = inputs
            self.data_mode = "disjoint"
        else:
            X, A = inputs
            I = tf.zeros(tf.shape(X)[:1])
            self.data_mode = "single"
        if K.ndim(I) == 2:
            I = I[:, 0]
        I = tf.cast(I, tf.int32)

        A_is_sparse = K.is_sparse(A)

        # Get mask
        y = self.compute_scores(X, A, I)
        N = K.shape(X)[-2]
        indices = ops.segment_top_k(y[:, 0], I, self.ratio)
        indices = tf.sort(indices)  # required for ordered SparseTensors
        mask = ops.indices_to_mask(indices, N)

        # Multiply X and y to make layer differentiable
        features = X * self.gating_op(y)

        axis = (
            0 if len(K.int_shape(A)) == 2 else 1
        )  # Cannot use negative axis in tf.boolean_mask
        # Reduce X
        X_pooled = tf.gather(features, indices, axis=axis)

        # Reduce A
        if A_is_sparse:
            A_pooled, _ = ops.gather_sparse_square(A, indices, mask=mask)
        else:
            A_pooled = tf.gather(A, indices, axis=axis)
            A_pooled = tf.gather(A_pooled, indices, axis=axis + 1)

        output = [X_pooled, A_pooled]

        # Reduce I
        if self.data_mode == "disjoint":
            I_pooled = tf.gather(I, indices)
            output.append(I_pooled)

        if self.return_mask:
            output.append(mask)

        return output

    def compute_scores(self, X, A, I):
        return K.dot(X, K.l2_normalize(self.kernel))

    @property
    def config(self):
        return {
            "ratio": self.ratio,
            "return_mask": self.return_mask,
            "sigmoid_gating": self.sigmoid_gating,
        }


class SAGPool(TopKPool):
    r"""
    A self-attention graph pooling layer (SAG) from the paper

    > [Self-Attention Graph Pooling](https://arxiv.org/abs/1904.08082)<br>
    > Junhyun Lee et al.

    **Mode**: single, disjoint.

    This layer computes the following operations:
    $$
        \y = \textrm{GNN}(\A, \X); \;\;\;\;
        \i = \textrm{rank}(\y, K); \;\;\;\;
        \X' = (\X \odot \textrm{tanh}(\y))_\i; \;\;\;\;
        \A' = \A_{\i, \i}
    $$

    where \( \textrm{rank}(\y, K) \) returns the indices of the top K values of
    \(\y\), and \(\textrm{GNN}\) consists of one GraphConv layer with no
    activation. \(K\) is defined for each graph as a fraction of the number of
    nodes.

    This layer temporarily makes the adjacency matrix dense in order to compute
    \(\A'\).
    If memory is not an issue, considerable speedups can be achieved by using
    dense graphs directly.
    Converting a graph from sparse to dense and back to sparse is an expensive
    operation.

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);

    **Output**

    - Reduced node features of shape `(ratio * n_nodes, n_node_features)`;
    - Reduced adjacency matrix of shape `(ratio * n_nodes, ratio * n_nodes)`;
    - Reduced graph IDs of shape `(ratio * n_nodes, )` (only in disjoint mode);
    - If `return_mask=True`, the binary pooling mask of shape `(ratio * n_nodes, )`.

    **Arguments**

    - `ratio`: float between 0 and 1, ratio of nodes to keep in each graph;
    - `return_mask`: boolean, whether to return the binary mask used for pooling;
    - `sigmoid_gating`: boolean, use a sigmoid gating activation instead of a
        tanh;
    - `kernel_initializer`: initializer for the weights;
    - `kernel_regularizer`: regularization applied to the weights;
    - `kernel_constraint`: constraint applied to the weights;
    """

    def __init__(
        self,
        ratio,
        return_mask=False,
        sigmoid_gating=False,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):
        super().__init__(
            ratio,
            return_mask=return_mask,
            sigmoid_gating=sigmoid_gating,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs
        )

    def compute_scores(self, X, A, I):
        scores = K.dot(X, self.kernel)
        scores = ops.modal_dot(A, scores)
        return scores
