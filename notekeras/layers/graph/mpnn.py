import copy
import inspect
import warnings

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from notekeras import ops
from notekeras.ops import deserialize_scatter, dot, serialize_scatter
from scipy import sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     GRUCell, Layer, PReLU)
from tensorflow.keras.models import Sequential

from .utils import *


class MessagePassing(Layer):
    r"""
    A general class for message passing networks from the paper

    > [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)<br>
    > Justin Gilmer et al.

    **Mode**: single, disjoint.

    **This layer and all of its extensions expect a sparse adjacency matrix.**

    This layer computes:
    $$
        \x_i' = \gamma \left( \x_i, \square_{j \in \mathcal{N}(i)} \,
        \phi \left(\x_i, \x_j, \e_{j \rightarrow i} \right) \right),
    $$

    where \( \gamma \) is a differentiable update function, \( \phi \) is a
    differentiable message function, \( \square \) is a permutation-invariant
    function to aggregate the messages (like the sum or the average), and
    \(\E_{ij}\) is the edge attribute of edge i-j.

    By extending this class, it is possible to create any message-passing layer
    in single/disjoint mode.

    **API**

    ```python
    propagate(x, a, e=None, **kwargs)
    ```
    Propagates the messages and computes embeddings for each node in the graph. <br>
    Any `kwargs` will be forwarded as keyword arguments to `message()`,
    `aggregate()` and `update()`.

    ```python
    message(x, **kwargs)
    ```
    Computes messages, equivalent to \(\phi\) in the definition. <br>
    Any extra keyword argument of this function will be populated by
    `propagate()` if a matching keyword is found. <br>
    Use `self.get_i()` and  `self.get_j()` to gather the elements using the
    indices `i` or `j` of the adjacency matrix. Equivalently, you can access
    the indices themselves via the `index_i` and `index_j` attributes.

    ```python
    aggregate(messages, **kwargs)
    ```
    Aggregates the messages, equivalent to \(\square\) in the definition. <br>
    The behaviour of this function can also be controlled using the `aggregate`
    keyword in the constructor of the layer (supported aggregations: sum, mean,
    max, min, prod). <br>
    Any extra keyword argument of this function will be  populated by
    `propagate()` if a matching keyword is found.

    ```python
    update(embeddings, **kwargs)
    ```
    Updates the aggregated messages to obtain the final node embeddings,
    equivalent to \(\gamma\) in the definition. <br>
    Any extra keyword argument of this function will be  populated by
    `propagate()` if a matching keyword is found.

    **Arguments**:

    - `aggregate`: string or callable, an aggregation function. This flag can be
    used to control the behaviour of `aggregate()` wihtout re-implementing it.
    Supported aggregations: 'sum', 'mean', 'max', 'min', 'prod'.
    If callable, the function must have the signature `foo(updates, indices, n_nodes)`
    and return a rank 2 tensor with shape `(n_nodes, ...)`.
    - `kwargs`: additional keyword arguments specific to Keras' Layers, like
    regularizers, initializers, constraints, etc.
    """

    def __init__(self,
                 aggregate="sum",
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

        self.msg_signature = inspect.signature(self.message).parameters
        self.agg_signature = inspect.signature(self.aggregate).parameters
        self.upd_signature = inspect.signature(self.update).parameters
        self.agg = deserialize_scatter(aggregate)

    def call(self, inputs, **kwargs):
        x, a, e = self.get_inputs(inputs)
        return self.propagate(x, a, e)

    def build(self, input_shape):
        self.built = True

    def propagate(self, x, a, e=None, **kwargs):
        self.n_nodes = tf.shape(x)[-2]
        self.index_i = a.indices[:, 1]
        self.index_j = a.indices[:, 0]

        # Message
        msg_kwargs = self.get_kwargs(x, a, e, self.msg_signature, kwargs)
        messages = self.message(x, **msg_kwargs)

        # Aggregate
        agg_kwargs = self.get_kwargs(x, a, e, self.agg_signature, kwargs)
        embeddings = self.aggregate(messages, **agg_kwargs)

        # Update
        upd_kwargs = self.get_kwargs(x, a, e, self.upd_signature, kwargs)
        output = self.update(embeddings, **upd_kwargs)

        return output

    def message(self, x, **kwargs):
        return self.get_j(x)

    def aggregate(self, messages, **kwargs):
        return self.agg(messages, self.index_i, self.n_nodes)

    def update(self, embeddings, **kwargs):
        return embeddings

    def get_i(self, x):
        return tf.gather(x, self.index_i, axis=-2)

    def get_j(self, x):
        return tf.gather(x, self.index_j, axis=-2)

    def get_kwargs(self, x, a, e, signature, kwargs):
        output = {}
        for k in signature.keys():
            if signature[k].default is inspect.Parameter.empty or k == "kwargs":
                pass
            elif k == "x":
                output[k] = x
            elif k == "a":
                output[k] = a
            elif k == "e":
                output[k] = e
            elif k in kwargs:
                output[k] = kwargs[k]
            else:
                raise ValueError(
                    "Missing key {} for signature {}".format(k, signature))

        return output

    @staticmethod
    def get_inputs(inputs):
        if len(inputs) == 3:
            x, a, e = inputs
            assert K.ndim(e) in (2, 3), "E must have rank 2 or 3"
        elif len(inputs) == 2:
            x, a = inputs
            e = None
        else:
            raise ValueError(
                "Expected 2 or 3 inputs tensors (X, A, E), got {}.".format(len(inputs)))
        assert K.ndim(x) in (2, 3), "X must have rank 2 or 3"
        assert K.is_sparse(a), "A must be a SparseTensor"
        assert K.ndim(a) == 2, "A must have rank 2"

        return x, a, e

    def get_config(self):
        mp_config = {"aggregate": serialize_scatter(self.agg)}
        keras_config = {}
        for key in self.kwargs_keys:
            keras_config[key] = serialize_kwarg(key, getattr(self, key))
        base_config = super().get_config()

        return {**base_config, **keras_config, **mp_config, **self.config}

    @property
    def config(self):
        return {}

    @staticmethod
    def preprocess(a):
        return a


class AGNNConv(MessagePassing):
    r"""
    An Attention-based Graph Neural Network (AGNN) from the paper

    > [Attention-based Graph Neural Network for Semi-supervised Learning](https://arxiv.org/abs/1803.03735)<br>
    > Kiran K. Thekumparampil et al.

    **Mode**: single, disjoint, mixed.

    **This layer expects a sparse adjacency matrix.**

    This layer computes:
    $$
        \X' = \P\X
    $$
    where
    $$
        \P_{ij} = \frac{
            \exp \left( \beta \cos \left( \x_i, \x_j \right) \right)
        }{
            \sum\limits_{k \in \mathcal{N}(i) \cup \{ i \}}
            \exp \left( \beta \cos \left( \x_i, \x_k \right) \right)
        }
    $$
    and \(\beta\) is a trainable parameter.

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`.

    **Output**

    - Node features with the same shape of the input.

    **Arguments**

    - `trainable`: boolean, if True, then beta is a trainable parameter.
    Otherwise, beta is fixed to 1;
    - `activation`: activation function;
    """

    def __init__(self, trainable=True, aggregate="sum", activation=None, **kwargs):
        super().__init__(aggregate=aggregate, activation=activation, **kwargs)
        self.trainable = trainable

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if self.trainable:
            self.beta = self.add_weight(
                shape=(1,), initializer="ones", name="beta")
        else:
            self.beta = K.constant(1.0)
        self.built = True

    def call(self, inputs, **kwargs):
        x, a, _ = self.get_inputs(inputs)
        x_norm = K.l2_normalize(x, axis=-1)
        output = self.propagate(x, a, x_norm=x_norm)
        output = self.activation(output)

        return output

    def message(self, x, x_norm=None):
        x_j = self.get_j(x)
        x_norm_i = self.get_i(x_norm)
        x_norm_j = self.get_j(x_norm)
        alpha = self.beta * tf.reduce_sum(x_norm_i * x_norm_j, axis=-1)

        if len(tf.shape(alpha)) == 2:
            alpha = tf.transpose(alpha)  # For mixed mode
        alpha = ops.unsorted_segment_softmax(alpha, self.index_i, self.n_nodes)
        if len(tf.shape(alpha)) == 2:
            alpha = tf.transpose(alpha)  # For mixed mode
        alpha = alpha[..., None]

        return alpha * x_j

    @property
    def config(self):
        return {
            "trainable": self.trainable,
        }


class CrystalConv(MessagePassing):
    r"""
    A crystal graph convolutional layer from the paper

    > [Crystal Graph Convolutional Neural Networks for an Accurate and
    Interpretable Prediction of Material Properties](https://arxiv.org/abs/1710.10324)<br>
    > Tian Xie and Jeffrey C. Grossman

    **Mode**: single, disjoint, mixed.

    **This layer expects a sparse adjacency matrix.**

    This layer computes:
    $$
        \x_i' = \x_i + \sum\limits_{j \in \mathcal{N}(i)} \sigma \left( \z_{ij}
        \W^{(f)} + \b^{(f)} \right) \odot \g \left( \z_{ij} \W^{(s)} + \b^{(s)}
        \right)
    $$
    where \(\z_{ij} = \X_i \| \X_j \| \E_{ij} \), \(\sigma\) is a sigmoid
    activation, and \(g\) is the activation function (defined by the `activation`
    argument).

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`.
    - Edge features of shape `(num_edges, n_edge_features)`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `activation`: activation function;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(
        self,
        channels,
        aggregate="sum",
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            aggregate=aggregate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels

    def build(self, input_shape):
        assert len(input_shape) >= 2
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )
        self.dense_f = Dense(
            self.channels, activation="sigmoid", **layer_kwargs)
        self.dense_s = Dense(
            self.channels, activation=self.activation, **layer_kwargs)

        self.built = True

    def message(self, x, e=None):
        x_i = self.get_i(x)
        x_j = self.get_j(x)

        to_concat = [x_i, x_j]
        if e is not None:
            to_concat += [e]
        z = K.concatenate(to_concat, axis=-1)
        output = self.dense_s(z) * self.dense_f(z)

        return output

    def update(self, embeddings, x=None):
        return x + embeddings

    @property
    def config(self):
        return {"channels": self.channels}


class EdgeConv(MessagePassing):
    r"""
    An edge convolutional layer from the paper

    > [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829)<br>
    > Yue Wang et al.

    **Mode**: single, disjoint, mixed.

    **This layer expects a sparse adjacency matrix.**

    This layer computes for each node \(i\):
    $$
        \x_i' = \sum\limits_{j \in \mathcal{N}(i)} \textrm{MLP}\big( \x_i \|
        \x_j - \x_i \big)
    $$
    where \(\textrm{MLP}\) is a multi-layer perceptron.

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `mlp_hidden`: list of integers, number of hidden units for each hidden
    layer in the MLP (if None, the MLP has only the output layer);
    - `mlp_activation`: activation for the MLP layers;
    - `activation`: activation function;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(
        self,
        channels,
        mlp_hidden=None,
        mlp_activation="relu",
        aggregate="sum",
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            aggregate=aggregate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.mlp_activation = activations.get(mlp_activation)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

        self.mlp = Sequential(
            [
                Dense(channels, self.mlp_activation, **layer_kwargs)
                for channels in self.mlp_hidden
            ]
            + [
                Dense(
                    self.channels,
                    self.activation,
                    use_bias=self.use_bias,
                    **layer_kwargs
                )
            ]
        )

        self.built = True

    def message(self, x, **kwargs):
        x_i = self.get_i(x)
        x_j = self.get_j(x)
        return self.mlp(K.concatenate((x_i, x_j - x_i)))

    @property
    def config(self):
        return {
            "channels": self.channels,
            "mlp_hidden": self.mlp_hidden,
            "mlp_activation": self.mlp_activation,
        }


class GatedGraphConv(MessagePassing):
    r"""
    A gated graph convolutional layer from the paper

    > [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493)<br>
    > Yujia Li et al.

    **Mode**: single, disjoint, mixed.

    **This layer expects a sparse adjacency matrix.**

    This layer computes \(\x_i' = \h^{(L)}_i\) where:
    $$
    \begin{align}
        & \h^{(0)}_i = \x_i \| \mathbf{0} \\
        & \m^{(l)}_i = \sum\limits_{j \in \mathcal{N}(i)} \h^{(l - 1)}_j \W \\
        & \h^{(l)}_i = \textrm{GRU} \left(\m^{(l)}_i, \h^{(l - 1)}_i \right) \\
    \end{align}
    $$
    where \(\textrm{GRU}\) is a gated recurrent unit cell.

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`; note that
    `n_node_features` must be smaller or equal than `channels`.
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `n_layers`: integer, number of iterations with the GRU cell;
    - `activation`: activation function;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(
        self,
        channels,
        n_layers,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.n_layers = n_layers

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.n_layers, self.channels, self.channels),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.rnn = GRUCell(
            self.channels,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            use_bias=self.use_bias,
        )
        self.built = True

    def call(self, inputs):
        x, a, _ = self.get_inputs(inputs)
        F = K.int_shape(x)[-1]

        to_pad = self.channels - F
        ndims = len(tf.shape(x)) - 1
        output = tf.pad(x, [[0, 0]] * ndims + [[0, to_pad]])
        for i in range(self.n_layers):
            m = tf.matmul(output, self.kernel[i])
            m = self.propagate(m, a)
            output = self.rnn(m, [output])[0]

        output = self.activation(output)
        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
            "n_layers": self.n_layers,
        }


class GeneralConv(MessagePassing):
    r"""
    A general convolutional layer from the paper

    > [Design Space for Graph Neural Networks](https://arxiv.org/abs/2011.08843)<br>
    > Jiaxuan You et al.

    **Mode**: single, disjoint, mixed.

    **This layer expects a sparse adjacency matrix.**

    This layer computes:
    $$
        \x_i' = \mathrm{Agg} \left( \left\{ \mathrm{Act} \left( \mathrm{Dropout}
        \left( \mathrm{BN} \left( \x_j \W + \b \right) \right) \right),
        j \in \mathcal{N}(i) \right\} \right)
    $$

    where \( \mathrm{Agg} \) is an aggregation function for the messages,
    \( \mathrm{Act} \) is an activation function, \( \mathrm{Dropout} \)
    applies dropout to the node features, and \( \mathrm{BN} \) applies batch
    normalization to the node features.

    This layer supports the PReLU activation via the 'prelu' keyword.

    The default parameters of this layer are selected according to the best
    results obtained in the paper, and should provide a good performance on
    many node-level and graph-level tasks, without modifications.
    The defaults are as follows:

    - 256 channels
    - Batch normalization
    - No dropout
    - PReLU activation
    - Sum aggregation

    If you are uncertain about which layers to use for your GNN, this is a
    safe choice. Check out the original paper for more specific configurations.

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `batch_norm`: bool, whether to use batch normalization;
    - `dropout`: float, dropout rate;
    - `aggregate`: string or callable, an aggregation function. Supported
    aggregations: 'sum', 'mean', 'max', 'min', 'prod'.
    - `activation`: activation function. This layer also supports the
    advanced activation PReLU by passing `activation='prelu'`.
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(
        self,
        channels=256,
        batch_norm=True,
        dropout=0.0,
        aggregate="sum",
        activation="prelu",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            aggregate=aggregate,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm
        if activation == "prelu" or "prelu" in kwargs:
            self.activation = PReLU()
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.dropout = Dropout(self.dropout_rate)
        if self.use_batch_norm:
            self.batch_norm = BatchNormalization()
        self.kernel = self.add_weight(
            shape=(input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.built = True

    def call(self, inputs, **kwargs):
        x, a, _ = self.get_inputs(inputs)

        # TODO: a = add_self_loops(a)

        x = tf.matmul(x, self.kernel)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.activation(x)

        return self.propagate(x, a)

    @property
    def config(self):
        config = {
            "channels": self.channels,
        }
        if self.activation.__class__.__name__ == "PReLU":
            config["prelu"] = True

        return config


class GINConv(MessagePassing):
    r"""
    A Graph Isomorphism Network (GIN) from the paper

    > [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826)<br>
    > Keyulu Xu et al.

    **Mode**: single, disjoint, mixed.

    **This layer expects a sparse adjacency matrix.**

    This layer computes for each node \(i\):
    $$
        \x_i' = \textrm{MLP}\big( (1 + \epsilon) \cdot \x_i + \sum\limits_{j
        \in \mathcal{N}(i)} \x_j \big)
    $$
    where \(\textrm{MLP}\) is a multi-layer perceptron.

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `epsilon`: unnamed parameter, see the original paper and the equation
    above.
    By setting `epsilon=None`, the parameter will be learned (default behaviour).
    If given as a value, the parameter will stay fixed.
    - `mlp_hidden`: list of integers, number of hidden units for each hidden
    layer in the MLP (if None, the MLP has only the output layer);
    - `mlp_activation`: activation for the MLP layers;
    - `activation`: activation function;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(
        self,
        channels,
        epsilon=None,
        mlp_hidden=None,
        mlp_activation="relu",
        aggregate="sum",
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            aggregate=aggregate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.epsilon = epsilon
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.mlp_activation = activations.get(mlp_activation)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

        self.mlp = Sequential(
            [
                Dense(channels, self.mlp_activation, **layer_kwargs)
                for channels in self.mlp_hidden
            ]
            + [
                Dense(
                    self.channels,
                    self.activation,
                    use_bias=self.use_bias,
                    **layer_kwargs
                )
            ]
        )

        if self.epsilon is None:
            self.eps = self.add_weight(
                shape=(1,), initializer="zeros", name="eps")
        else:
            # If epsilon is given, keep it constant
            self.eps = K.constant(self.epsilon)

        self.built = True

    def call(self, inputs, **kwargs):
        x, a, _ = self.get_inputs(inputs)
        output = self.mlp((1.0 + self.eps) * x + self.propagate(x, a))

        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
            "epsilon": self.epsilon,
            "mlp_hidden": self.mlp_hidden,
            "mlp_activation": self.mlp_activation,
        }


class GraphSageConv(MessagePassing):
    r"""
    A GraphSAGE layer from the paper

    > [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)<br>
    > William L. Hamilton et al.

    **Mode**: single, disjoint, mixed.

    **This layer expects a sparse adjacency matrix.**

    This layer computes:
    $$
        \X' = \big[ \textrm{AGGREGATE}(\X) \| \X \big] \W + \b; \\
        \X' = \frac{\X'}{\|\X'\|}
    $$
    where \( \textrm{AGGREGATE} \) is a function to aggregate a node's
    neighbourhood. The supported aggregation methods are: sum, mean,
    max, min, and product.

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`.

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;
    - `aggregate_op`: str, aggregation method to use (`'sum'`, `'mean'`,
    `'max'`, `'min'`, `'prod'`);
    - `activation`: activation function;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.

    """

    def __init__(
        self,
        channels,
        aggregate="mean",
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            aggregate=aggregate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            shape=(2 * input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        x, a, _ = self.get_inputs(inputs)
        a = ops.add_self_loops(a)

        aggregated = self.propagate(x, a)
        output = K.concatenate([x, aggregated])
        output = K.dot(output, self.kernel)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        output = K.l2_normalize(output, axis=-1)
        if self.activation is not None:
            output = self.activation(output)

        return output

    @property
    def config(self):
        return {"channels": self.channels}


class TAGConv(MessagePassing):
    r"""
    A Topology Adaptive Graph Convolutional layer (TAG) from the paper

    > [Topology Adaptive Graph Convolutional Networks](https://arxiv.org/abs/1710.10370)<br>
    > Jian Du et al.

    **Mode**: single, disjoint, mixed.

    **This layer expects a sparse adjacency matrix.**

    This layer computes:
    $$
        \Z = \sum\limits_{k=0}^{K} \D^{-1/2}\A^k\D^{-1/2}\X\W^{(k)}
    $$

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `K`: the order of the layer (i.e., the layer will consider a K-hop
    neighbourhood for each node);
    - `activation`: activation function;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(
        self,
        channels,
        K=3,
        aggregate="sum",
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            aggregate=aggregate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.K = K
        self.linear = Dense(
            channels,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
        )

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True

    def call(self, inputs, **kwargs):
        x, a, _ = self.get_inputs(inputs)
        edge_weight = a.values

        output = [x]
        for k in range(self.K):
            output.append(self.propagate(x, a, edge_weight=edge_weight))
        output = K.concatenate(output)

        return self.linear(output)

    def message(self, x, edge_weight=None):
        x_j = self.get_j(x)
        return edge_weight[:, None] * x_j

    @property
    def config(self):
        return {
            "channels": self.channels,
        }

    @staticmethod
    def preprocess(a):
        return normalized_adjacency(a)
