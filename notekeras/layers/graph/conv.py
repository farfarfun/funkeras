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


class Conv(Layer):
    r"""
    A general class for convolutional layers.

    You can extend this class to create custom implementations of GNN layers
    that use standard matrix multiplication instead of the gather-scatter
    approach of MessagePassing.

    This is useful if you want to create layers that support dense inputs,
    batch and mixed modes, or other non-standard processing. No checks are done
    on the inputs, to allow for maximum flexibility.

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

    def call(self, inputs, **kwargs):
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

    @staticmethod
    def preprocess(a):
        return a


class DiffuseFeatures(Layer):
    r"""
    Utility layer calculating a single channel of the diffusional convolution.

    The procedure is based on [https://arxiv.org/abs/1707.01926](https://arxiv.org/abs/1707.01926)

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Normalized adjacency or attention coef. matrix \(\hat \A \) of shape
    `([batch], n_nodes, n_nodes)`; Use DiffusionConvolution.preprocess to normalize.

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to \(1\).

    **Arguments**

    - `num_diffusion_steps`: How many diffusion steps to consider. \(K\) in paper.
    - `kernel_initializer`: initializer for the weights;
    - `kernel_regularizer`: regularization applied to the kernel vectors;
    - `kernel_constraint`: constraint applied to the kernel vectors;
    """

    def __init__(
        self,
        num_diffusion_steps,
        kernel_initializer,
        kernel_regularizer,
        kernel_constraint,
        **kwargs
    ):
        super(DiffuseFeatures, self).__init__(**kwargs)

        self.K = num_diffusion_steps
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

    def build(self, input_shape):
        # Initializing the kernel vector (R^K) (theta in paper)
        self.kernel = self.add_weight(
            shape=(self.K,),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

    def call(self, inputs):
        x, a = inputs

        # Calculate diffusion matrix: sum kernel_k * Attention_t^k
        # tf.polyval needs a list of tensors as the coeff. thus we
        # unstack kernel
        diffusion_matrix = tf.math.polyval(tf.unstack(self.kernel), a)

        # Apply it to X to get a matrix C = [C_1, ..., C_F] (n_nodes x n_node_features)
        # of diffused features
        diffused_features = tf.matmul(diffusion_matrix, x)

        # Now we add all diffused features (columns of the above matrix)
        # and apply a non linearity to obtain H:,q (eq. 3 in paper)
        H = tf.math.reduce_sum(diffused_features, axis=-1)

        # H has shape ([batch], n_nodes) but as it is the sum of columns
        # we reshape it to ([batch], n_nodes, 1)
        return tf.expand_dims(H, -1)


class APPNPConv(Conv):
    r"""
    The APPNP operator from the paper

    > [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997)<br>
    > Johannes Klicpera et al.

    **Mode**: single, disjoint, mixed, batch.

    This layer computes:
    $$
        \Z^{(0)} = \textrm{MLP}(\X); \\
        \Z^{(K)} = (1 - \alpha) \hat \D^{-1/2} \hat \A \hat \D^{-1/2} \Z^{(K - 1)} +
                   \alpha \Z^{(0)},
    $$
    where \(\alpha\) is the teleport probability, \(\textrm{MLP}\) is a
    multi-layer perceptron, and \(K\) is defined by the `propagations` argument.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Modified Laplacian of shape `([batch], n_nodes, n_nodes)`; can be computed with
    `spektral.utils.convolution.gcn_filter`.

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;
    - `alpha`: teleport probability during propagation;
    - `propagations`: number of propagation steps;
    - `mlp_hidden`: list of integers, number of hidden units for each hidden
    layer in the MLP (if None, the MLP has only the output layer);
    - `mlp_activation`: activation for the MLP layers;
    - `dropout_rate`: dropout rate for Laplacian and MLP layers;
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
        alpha=0.2,
        propagations=1,
        mlp_hidden=None,
        mlp_activation="relu",
        dropout_rate=0.0,
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
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.alpha = alpha
        self.propagations = propagations
        self.mlp_activation = activations.get(mlp_activation)
        self.dropout_rate = dropout_rate

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
        mlp_layers = []
        for i, channels in enumerate(self.mlp_hidden):
            mlp_layers.extend(
                [
                    Dropout(self.dropout_rate),
                    Dense(channels, self.mlp_activation, **layer_kwargs),
                ]
            )
        mlp_layers.append(Dense(self.channels, "linear", **layer_kwargs))
        self.mlp = Sequential(mlp_layers)
        self.built = True

    def call(self, inputs):
        x, a = inputs

        mlp_out = self.mlp(x)
        z = mlp_out
        for k in range(self.propagations):
            z = (1 - self.alpha) * ops.modal_dot(a, z) + self.alpha * mlp_out
        output = self.activation(z)

        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
            "alpha": self.alpha,
            "propagations": self.propagations,
            "mlp_hidden": self.mlp_hidden,
            "mlp_activation": activations.serialize(self.mlp_activation),
            "dropout_rate": self.dropout_rate,
        }

    @staticmethod
    def preprocess(a):
        return gcn_filter(a)


class ARMAConv(Conv):
    r"""
    An Auto-Regressive Moving Average convolutional layer (ARMA) from the paper

    > [Graph Neural Networks with convolutional ARMA filters](https://arxiv.org/abs/1901.01343)<br>
    > Filippo Maria Bianchi et al.

    **Mode**: single, disjoint, mixed, batch.

    This layer computes:
    $$
        \X' = \frac{1}{K} \sum\limits_{k=1}^K \bar\X_k^{(T)},
    $$
    where \(K\) is the order of the ARMA\(_K\) filter, and where:
    $$
        \bar \X_k^{(t + 1)} =
        \sigma \left(\tilde \A \bar \X^{(t)} \W^{(t)} + \X \V^{(t)} \right)
    $$
    is a recursive approximation of an ARMA\(_1\) filter, where
    \( \bar \X^{(0)} = \X \)
    and
    $$
        \tilde \A =  \D^{-1/2} \A \D^{-1/2}.
    $$

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Normalized and rescaled Laplacian of shape `([batch], n_nodes, n_nodes)`; can be
    computed with `spektral.utils.convolution.normalized_laplacian` and
    `spektral.utils.convolution.rescale_laplacian`.

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;
    - `order`: order of the full ARMA\(_K\) filter, i.e., the number of parallel
    stacks in the layer;
    - `iterations`: number of iterations to compute each ARMA\(_1\) approximation;
    - `share_weights`: share the weights in each ARMA\(_1\) stack.
    - `gcn_activation`: activation function to compute each ARMA\(_1\)
    stack;
    - `dropout_rate`: dropout rate for skip connection;
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
        order=1,
        iterations=1,
        share_weights=False,
        gcn_activation="relu",
        dropout_rate=0.0,
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
        self.iterations = iterations
        self.order = order
        self.share_weights = share_weights
        self.gcn_activation = activations.get(gcn_activation)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Create weights for parallel stacks
        # self.kernels[k][i] refers to the k-th stack, i-th iteration
        self.kernels = []
        for k in range(self.order):
            kernel_stack = []
            current_shape = F
            for i in range(self.iterations):
                kernel_stack.append(
                    self.create_weights(
                        current_shape, F, self.channels, "ARMA_GCS_{}{}".format(
                            k, i)
                    )
                )
                current_shape = self.channels
                if self.share_weights and i == 1:
                    # No need to continue because all weights will be shared
                    break
            self.kernels.append(kernel_stack)
        self.built = True

    def call(self, inputs):
        x, a = inputs

        output = []
        for k in range(self.order):
            output_k = x
            for i in range(self.iterations):
                output_k = self.gcs([output_k, x, a], k, i)
            output.append(output_k)
        output = K.stack(output, axis=-1)
        output = K.mean(output, axis=-1)
        output = self.activation(output)

        return output

    def create_weights(self, input_dim, input_dim_skip, channels, name):
        """
        Creates a set of weights for a GCN with skip connections.
        :param input_dim: dimension of the input space
        :param input_dim_skip: dimension of the input space for the skip connection
        :param channels: dimension of the output space
        :param name: name of the layer
        :return:
            - kernel_1, from input space of the layer to output space
            - kernel_2, from input space of the skip connection to output space
            - bias, bias vector on the output space if use_bias=True, None otherwise.
        """
        kernel_1 = self.add_weight(
            shape=(input_dim, channels),
            name=name + "_kernel_1",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        kernel_2 = self.add_weight(
            shape=(input_dim_skip, channels),
            name=name + "_kernel_2",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            bias = self.add_weight(
                shape=(channels,),
                name=name + "_bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            bias = None
        return kernel_1, kernel_2, bias

    def gcs(self, inputs, stack, iteration):
        """
        Creates a graph convolutional layer with a skip connection.
        :param inputs: list of input Tensors, namely
            - input node features
            - input node features for the skip connection
            - normalized adjacency matrix;
        :param stack: int, current stack (used to retrieve kernels);
        :param iteration: int, current iteration (used to retrieve kernels);
        :return: output node features.
        """
        x, x_skip, a = inputs

        iter = 1 if self.share_weights and iteration >= 1 else iteration
        kernel_1, kernel_2, bias = self.kernels[stack][iter]

        output = K.dot(x, kernel_1)
        output = ops.modal_dot(a, output)

        skip = K.dot(x_skip, kernel_2)
        skip = Dropout(self.dropout_rate)(skip)
        output += skip

        if self.use_bias:
            output = K.bias_add(output, bias)
        output = self.gcn_activation(output)

        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
            "iterations": self.iterations,
            "order": self.order,
            "share_weights": self.share_weights,
            "gcn_activation": activations.serialize(self.gcn_activation),
            "dropout_rate": self.dropout_rate,
        }

    @staticmethod
    def preprocess(a):
        return normalized_adjacency(a, symmetric=True)


class ChebConv(Conv):
    r"""
    A Chebyshev convolutional layer from the paper

    > [Convolutional Neural Networks on Graphs with Fast Localized Spectral
  Filtering](https://arxiv.org/abs/1606.09375)<br>
    > Michaël Defferrard et al.

    **Mode**: single, disjoint, mixed, batch.

    This layer computes:
    $$
        \X' = \sum \limits_{k=0}^{K - 1} \T^{(k)} \W^{(k)}  + \b^{(k)},
    $$
    where \( \T^{(0)}, ..., \T^{(K - 1)} \) are Chebyshev polynomials of \(\tilde \L\)
    defined as
    $$
        \T^{(0)} = \X \\
        \T^{(1)} = \tilde \L \X \\
        \T^{(k \ge 2)} = 2 \cdot \tilde \L \T^{(k - 1)} - \T^{(k - 2)},
    $$
    where
    $$
        \tilde \L =  \frac{2}{\lambda_{max}} \cdot (\I - \D^{-1/2} \A \D^{-1/2}) - \I.
    $$

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - A list of K Chebyshev polynomials of shape
    `[([batch], n_nodes, n_nodes), ..., ([batch], n_nodes, n_nodes)]`; can be computed with
    `spektral.utils.convolution.chebyshev_filter`.

    **Output**

    - Node features with the same shape of the input, but with the last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;
    - `K`: order of the Chebyshev polynomials;
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
        K=1,
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
        self.K = K

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            shape=(self.K, input_dim, self.channels),
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
        x, a = inputs

        T_0 = x
        output = K.dot(T_0, self.kernel[0])

        if self.K > 1:
            T_1 = ops.modal_dot(a, x)
            output += K.dot(T_1, self.kernel[1])

        for k in range(2, self.K):
            T_2 = 2 * ops.modal_dot(a, T_1) - T_0
            output += K.dot(T_2, self.kernel[k])
            T_0, T_1 = T_1, T_2

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        output = self.activation(output)

        return output

    @property
    def config(self):
        return {"channels": self.channels, "K": self.K}

    @staticmethod
    def preprocess(a):
        a = normalized_laplacian(a)
        a = rescale_laplacian(a)
        return a


class DiffusionConv(Conv):
    r"""
      A diffusion convolution operator from the paper

      > [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
    Forecasting](https://arxiv.org/abs/1707.01926)<br>
      > Yaguang Li et al.

      **Mode**: single, disjoint, mixed, batch.

      **This layer expects a dense adjacency matrix.**

      Given a number of diffusion steps \(K\) and a row-normalized adjacency
      matrix \(\hat \A \), this layer calculates the \(q\)-th channel as:
      $$
      \mathbf{X}_{~:,~q}' = \sigma\left( \sum_{f=1}^{F} \left( \sum_{k=0}^{K-1}
      \theta_k {\hat \A}^k \right) \X_{~:,~f} \right)
      $$

      **Input**

      - Node features of shape `([batch], n_nodes, n_node_features)`;
      - Normalized adjacency or attention coef. matrix \(\hat \A \) of shape
      `([batch], n_nodes, n_nodes)`; Use `DiffusionConvolution.preprocess` to normalize.

      **Output**

      - Node features with the same shape as the input, but with the last
      dimension changed to `channels`.

      **Arguments**

      - `channels`: number of output channels;
      - `K`: number of diffusion steps.
      - `activation`: activation function \(\sigma\); (\(\tanh\) by default)
      - `kernel_initializer`: initializer for the weights;
      - `kernel_regularizer`: regularization applied to the weights;
      - `kernel_constraint`: constraint applied to the weights;
    """

    def __init__(
        self,
        channels,
        K=6,
        activation="tanh",
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

        self.channels = channels
        self.K = K + 1

    def build(self, input_shape):
        self.filters = [
            DiffuseFeatures(
                num_diffusion_steps=self.K,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
            )
            for _ in range(self.channels)
        ]

    def apply_filters(self, x, a):
        # This will be a list of channels diffused features.
        # Each diffused feature is a (batch, n_nodes, 1) tensor.
        # Later we will concat all the features to get one
        # (batch, n_nodes, channels) diffused graph signal
        diffused_features = []

        # Iterating over all channels diffusion filters
        for diffusion in self.filters:
            diffused_feature = diffusion((x, a))
            diffused_features.append(diffused_feature)

        return tf.concat(diffused_features, -1)

    def call(self, inputs):
        x, a = inputs
        h = self.apply_filters(x, a)
        h = self.activation(h)

        return h

    @property
    def config(self):
        return {"channels": self.channels, "K": self.K - 1}

    @staticmethod
    def preprocess(a):
        return gcn_filter(a)


class ECCConv(Conv):
    r"""
      An edge-conditioned convolutional layer (ECC) from the paper

      > [Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs](https://arxiv.org/abs/1704.02901)<br>
      > Martin Simonovsky and Nikos Komodakis

    **Mode**: single, disjoint, batch, mixed.

    **In single, disjoint, and mixed mode, this layer expects a sparse adjacency
    matrix. If a dense adjacency is given as input, it will be automatically
    cast to sparse, which might be expensive.**

      This layer computes:
      $$
          \x_i' = \x_{i} \W_{\textrm{root}} + \sum\limits_{j \in \mathcal{N}(i)}
          \x_{j} \textrm{MLP}(\e_{j \rightarrow i}) + \b
      $$
      where \(\textrm{MLP}\) is a multi-layer perceptron that outputs an
      edge-specific weight as a function of edge attributes.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Binary adjacency matrices of shape `([batch], n_nodes, n_nodes)`;
    - Edge features. In single mode, shape `(num_edges, n_edge_features)`; in
    batch mode, shape `(batch, n_nodes, n_nodes, n_edge_features)`.

      **Output**

      - node features with the same shape of the input, but the last dimension
      changed to `channels`.

      **Arguments**

      - `channels`: integer, number of output channels;
      - `kernel_network`: a list of integers representing the hidden neurons of
      the kernel-generating network;
      - 'root': if False, the layer will not consider the root node for computing
      the message passing (first term in equation above), but only the neighbours.
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
        kernel_network=None,
        root=True,
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
        self.kernel_network = kernel_network
        self.root = root

    def build(self, input_shape):
        F = input_shape[0][-1]
        F_ = self.channels
        self.kernel_network_layers = []
        if self.kernel_network is not None:
            for i, l in enumerate(self.kernel_network):
                self.kernel_network_layers.append(
                    Dense(
                        l,
                        name="FGN_{}".format(i),
                        activation="relu",
                        use_bias=self.use_bias,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        bias_constraint=self.bias_constraint,
                    )
                )
        self.kernel_network_layers.append(Dense(F_ * F, name="FGN_out"))

        if self.root:
            self.root_kernel = self.add_weight(
                name="root_kernel",
                shape=(F, F_),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )
        else:
            self.root_kernel = None
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.channels,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        x, a, e = inputs

        # Parameters
        N = K.shape(x)[-2]
        F = K.int_shape(x)[-1]
        F_ = self.channels

        # Filter network
        kernel_network = e
        for layer in self.kernel_network_layers:
            kernel_network = layer(kernel_network)

        # Convolution
        mode = ops.autodetect_mode(x, a)
        if mode == ops.BATCH:
            kernel = K.reshape(kernel_network, (-1, N, N, F_, F))
            output = kernel * a[..., None, None]
            output = tf.einsum("abcde,ace->abd", output, x)
        else:
            # Enforce sparse representation
            if not K.is_sparse(a):
                warnings.warn(
                    "Casting dense adjacency matrix to SparseTensor."
                    "This can be an expensive operation. "
                )
                a = ops.dense_to_sparse(a)

            target_shape = (-1, F, F_)
            if mode == ops.MIXED:
                target_shape = (tf.shape(x)[0],) + target_shape
            kernel = tf.reshape(kernel_network, target_shape)
            index_i = a.indices[:, 1]
            index_j = a.indices[:, 0]
            messages = tf.gather(x, index_j, axis=-2)
            messages = tf.einsum("...ab,...abc->...ac", messages, kernel)
            output = ops.scatter_sum(messages, index_i, N)

        if self.root:
            output += K.dot(x, self.root_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
            "kernel_network": self.kernel_network,
            "root": self.root,
        }


class GATConv(Conv):
    r"""
    A Graph Attention layer (GAT) from the paper

    > [Graph Attention Networks](https://arxiv.org/abs/1710.10903)<br>
    > Petar Veličković et al.

    **Mode**: single, disjoint, mixed, batch.

    **This layer expects dense inputs when working in batch mode.**

    This layer computes a convolution similar to `layers.GraphConv`, but
    uses the attention mechanism to weight the adjacency matrix instead of
    using the normalized Laplacian:
    $$
        \X' = \mathbf{\alpha}\X\W + \b
    $$
    where
    $$
        \mathbf{\alpha}_{ij} =\frac{ \exp\left(\mathrm{LeakyReLU}\left(
        \a^{\top} [(\X\W)_i \, \| \, (\X\W)_j]\right)\right)}{\sum\limits_{k
        \in \mathcal{N}(i) \cup \{ i \}} \exp\left(\mathrm{LeakyReLU}\left(
        \a^{\top} [(\X\W)_i \, \| \, (\X\W)_k]\right)\right)}
    $$
    where \(\a \in \mathbb{R}^{2F'}\) is a trainable attention kernel.
    Dropout is also applied to \(\alpha\) before computing \(\Z\).
    Parallel attention heads are computed in parallel and their results are
    aggregated by concatenation or average.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `([batch], n_nodes, n_nodes)`;

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`;
    - if `return_attn_coef=True`, a list with the attention coefficients for
    each attention head. Each attention coefficient matrix has shape
    `([batch], n_nodes, n_nodes)`.

    **Arguments**

    - `channels`: number of output channels;
    - `attn_heads`: number of attention heads to use;
    - `concat_heads`: bool, whether to concatenate the output of the attention
     heads instead of averaging;
    - `dropout_rate`: internal dropout rate for attention coefficients;
    - `return_attn_coef`: if True, return the attention coefficients for
    the given input (one n_nodes x n_nodes matrix for each head).
    - `activation`: activation function;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `attn_kernel_initializer`: initializer for the attention weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `attn_kernel_regularizer`: regularization applied to the attention kernels;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `attn_kernel_constraint`: constraint applied to the attention kernels;
    - `bias_constraint`: constraint applied to the bias vector.

    """

    def __init__(
        self,
        channels,
        attn_heads=1,
        concat_heads=True,
        dropout_rate=0.5,
        return_attn_coef=False,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        attn_kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        bias_regularizer=None,
        attn_kernel_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        attn_kernel_constraint=None,
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
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate
        self.return_attn_coef = return_attn_coef
        self.attn_kernel_initializer = initializers.get(
            attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(
            attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

        if concat_heads:
            self.output_dim = self.channels * self.attn_heads
        else:
            self.output_dim = self.channels

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]

        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_dim, self.attn_heads, self.channels],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.attn_kernel_self = self.add_weight(
            name="attn_kernel_self",
            shape=[self.channels, self.attn_heads, 1],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )
        self.attn_kernel_neighs = self.add_weight(
            name="attn_kernel_neigh",
            shape=[self.channels, self.attn_heads, 1],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=[self.output_dim],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name="bias",
            )

        self.dropout = Dropout(self.dropout_rate)
        self.built = True

    def call(self, inputs):
        x, a = inputs

        mode = ops.autodetect_mode(x, a)
        if mode == ops.SINGLE and K.is_sparse(a):
            output, attn_coef = self._call_single(x, a)
        else:
            if K.is_sparse(a):
                a = tf.sparse.to_dense(a)
            output, attn_coef = self._call_dense(x, a)

        if self.concat_heads:
            shape = output.shape[:-2] + [self.attn_heads * self.channels]
            shape = [d if d is not None else -1 for d in shape]
            output = tf.reshape(output, shape)
        else:
            output = tf.reduce_mean(output, axis=-2)

        if self.use_bias:
            output += self.bias

        output = self.activation(output)

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def _call_single(self, x, a):
        # Reshape kernels for efficient message-passing
        kernel = tf.reshape(self.kernel, (-1, self.attn_heads * self.channels))
        attn_kernel_self = ops.transpose(self.attn_kernel_self, (2, 1, 0))
        attn_kernel_neighs = ops.transpose(self.attn_kernel_neighs, (2, 1, 0))

        # Prepare message-passing
        indices = a.indices
        N = tf.shape(x, out_type=indices.dtype)[-2]
        indices = ops.add_self_loops_indices(indices, N)
        targets, sources = indices[:, 1], indices[:, 0]

        # Update node features
        x = K.dot(x, kernel)
        x = tf.reshape(x, (-1, self.attn_heads, self.channels))

        # Compute attention
        attn_for_self = tf.reduce_sum(x * attn_kernel_self, -1)
        attn_for_self = tf.gather(attn_for_self, targets)
        attn_for_neighs = tf.reduce_sum(x * attn_kernel_neighs, -1)
        attn_for_neighs = tf.gather(attn_for_neighs, sources)

        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)
        attn_coef = ops.unsorted_segment_softmax(attn_coef, targets, N)
        attn_coef = self.dropout(attn_coef)
        attn_coef = attn_coef[..., None]

        # Update representation
        output = attn_coef * tf.gather(x, sources)
        output = tf.math.unsorted_segment_sum(output, targets, N)

        return output, attn_coef

    def _call_dense(self, x, a):
        shape = tf.shape(a)[:-1]
        a = tf.linalg.set_diag(a, tf.zeros(shape, a.dtype))
        a = tf.linalg.set_diag(a, tf.ones(shape, a.dtype))
        x = tf.einsum("...NI , IHO -> ...NHO", x, self.kernel)
        attn_for_self = tf.einsum(
            "...NHI , IHO -> ...NHO", x, self.attn_kernel_self)
        attn_for_neighs = tf.einsum(
            "...NHI , IHO -> ...NHO", x, self.attn_kernel_neighs
        )
        attn_for_neighs = tf.einsum("...ABC -> ...CBA", attn_for_neighs)

        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)

        mask = -10e9 * (1.0 - a)
        attn_coef += mask[..., None, :]
        attn_coef = tf.nn.softmax(attn_coef, axis=-1)
        attn_coef_drop = self.dropout(attn_coef)

        output = tf.einsum("...NHM , ...MHI -> ...NHI", attn_coef_drop, x)

        return output, attn_coef

    @property
    def config(self):
        return {
            "channels": self.channels,
            "attn_heads": self.attn_heads,
            "concat_heads": self.concat_heads,
            "dropout_rate": self.dropout_rate,
            "return_attn_coef": self.return_attn_coef,
            "attn_kernel_initializer": initializers.serialize(
                self.attn_kernel_initializer
            ),
            "attn_kernel_regularizer": regularizers.serialize(
                self.attn_kernel_regularizer
            ),
            "attn_kernel_constraint": constraints.serialize(
                self.attn_kernel_constraint
            ),
        }


class GCNConv(Conv):
    r"""
    A graph convolutional layer (GCN) from the paper

    > [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)<br>
    > Thomas N. Kipf and Max Welling

    **Mode**: single, disjoint, mixed, batch.

    This layer computes:
    $$
        \X' = \hat \D^{-1/2} \hat \A \hat \D^{-1/2} \X \W + \b
    $$
    where \( \hat \A = \A + \I \) is the adjacency matrix with added self-loops
    and \(\hat\D\) is its degree matrix.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Modified Laplacian of shape `([batch], n_nodes, n_nodes)`; can be computed with
    `spektral.utils.convolution.gcn_filter`.

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;
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

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
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
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        x, a = inputs

        output = K.dot(x, self.kernel)
        output = ops.modal_dot(a, output)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        output = self.activation(output)

        return output

    @property
    def config(self):
        return {"channels": self.channels}

    @staticmethod
    def preprocess(a):
        return gcn_filter(a)


class GCSConv(Conv):
    r"""
    A `GraphConv` layer with a trainable skip connection.

    **Mode**: single, disjoint, mixed, batch.

    This layer computes:
    $$
        \Z' = \D^{-1/2} \A \D^{-1/2} \X \W_1 + \X \W_2 + \b
    $$
    where \( \A \) does not have self-loops.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Normalized adjacency matrix of shape `([batch], n_nodes, n_nodes)`; can be computed
    with `spektral.utils.convolution.normalized_adjacency`.

    **Output**

    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;
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

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]

        self.kernel_1 = self.add_weight(
            shape=(input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel_1",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.kernel_2 = self.add_weight(
            shape=(input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel_2",
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
        x, a = inputs

        output = K.dot(x, self.kernel_1)
        output = ops.modal_dot(a, output)
        skip = K.dot(x, self.kernel_2)
        output += skip

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    @property
    def config(self):
        return {"channels": self.channels}

    @staticmethod
    def preprocess(a):
        return normalized_adjacency(a)
