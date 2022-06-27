import numpy as np
import tensorflow as tf
from scipy import sparse as sp
from tensorflow.keras import backend as K
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse as tfsp

SINGLE = 1  # Single mode    rank(x) = 2, rank(a) = 2
DISJOINT = SINGLE  # Disjoint mode  rank(x) = 2, rank(a) = 2
BATCH = 3  # Batch mode     rank(x) = 3, rank(a) = 3
MIXED = 4  # Mixed mode     rank(x) = 3, rank(a) = 2


def transpose(a, perm=None, name=None):
    """
    Transposes a according to perm, dealing automatically with sparsity.
    :param a: Tensor or SparseTensor with rank k.
    :param perm: permutation indices of size k.
    :param name: name for the operation.
    :return: Tensor or SparseTensor with rank k.
    """
    if K.is_sparse(a):
        transpose_op = tf.sparse.transpose
    else:
        transpose_op = tf.transpose

    if perm is None:
        perm = (1, 0)  # Make explicit so that shape will always be preserved
    return transpose_op(a, perm=perm, name=name)


def reshape(a, shape=None, name=None):
    """
    Reshapes a according to shape, dealing automatically with sparsity.
    :param a: Tensor or SparseTensor.
    :param shape: new shape.
    :param name: name for the operation.
    :return: Tensor or SparseTensor.
    """
    if K.is_sparse(a):
        reshape_op = tf.sparse.reshape
    else:
        reshape_op = tf.reshape

    return reshape_op(a, shape=shape, name=name)


def repeat(x, repeats):
    """
    Repeats elements of a Tensor (equivalent to np.repeat, but only for 1D
    tensors).
    :param x: rank 1 Tensor;
    :param repeats: rank 1 Tensor with same shape as x, the number of
    repetitions for each element;
    :return: rank 1 Tensor, of shape `(sum(repeats), )`.
    """
    x = tf.expand_dims(x, 1)
    max_repeats = tf.reduce_max(repeats)
    tile_repeats = [1, max_repeats]
    arr_tiled = tf.tile(x, tile_repeats)
    mask = tf.less(tf.range(max_repeats), tf.expand_dims(repeats, 1))
    result = tf.reshape(tf.boolean_mask(arr_tiled, mask), [-1])
    return result


def segment_top_k(x, I, ratio):
    """
    Returns indices to get the top K values in x segment-wise, according to
    the segments defined in I. K is not fixed, but it is defined as a ratio of
    the number of elements in each segment.
    :param x: a rank 1 Tensor;
    :param I: a rank 1 Tensor with segment IDs for x;
    :param ratio: float, ratio of elements to keep for each segment;
    :return: a rank 1 Tensor containing the indices to get the top K values of
    each segment in x.
    """
    rt = tf.RaggedTensor.from_value_rowids(x, I)
    row_lengths = rt.row_lengths()
    dense = rt.to_tensor(default_value=-np.inf)
    indices = tf.cast(tf.argsort(dense, direction="DESCENDING"), tf.int64)
    row_starts = tf.cast(rt.row_starts(), tf.int64)
    indices = indices + tf.expand_dims(row_starts, 1)
    row_lengths = tf.cast(
        tf.math.ceil(ratio * tf.cast(row_lengths, tf.float32)), tf.int32
    )
    return tf.RaggedTensor.from_tensor(indices, row_lengths).values


def indices_to_mask(indices, shape, dtype=tf.bool):
    """
    Return mask with true values at indices of the given shape.
    This can be used as an inverse to tf.where.
    :param indices: [nnz, k] or [nnz] Tensor indices of True values.
    :param shape: [k] or [] (scalar) Tensor shape/size of output.
    :return: Tensor of given shape and dtype.
    """
    indices = tf.convert_to_tensor(indices, dtype_hint=tf.int64)
    if indices.shape.ndims == 1:
        assert isinstance(shape, int) or shape.shape.ndims == 0
        indices = tf.expand_dims(indices, axis=1)
        if isinstance(shape, int):
            shape = tf.TensorShape([shape])
        else:
            shape = tf.expand_dims(shape, axis=0)
    else:
        indices.shape.assert_has_rank(2)
    assert indices.dtype.is_integer
    nnz = tf.shape(indices)[0]
    indices = tf.cast(indices, tf.int64)
    shape = tf.cast(shape, tf.int64)
    return tf.scatter_nd(indices, tf.ones((nnz,), dtype=dtype), shape)


def disjoint_signal_to_batch(X, I):
    """
    Converts a disjoint graph signal to batch node by zero-padding.

    :param X: Tensor, node features of shape (nodes, features).
    :param I: Tensor, graph IDs of shape `(n_nodes, )`;
    :return batch: Tensor, batched node features of shape (batch, N_max, n_node_features)
    """
    I = tf.cast(I, tf.int32)
    num_nodes = tf.math.segment_sum(tf.ones_like(I), I)
    start_index = tf.cumsum(num_nodes, exclusive=True)
    n_graphs = tf.shape(num_nodes)[0]
    max_n_nodes = tf.reduce_max(num_nodes)
    batch_n_nodes = tf.shape(I)[0]
    feature_dim = tf.shape(X)[-1]

    index = tf.range(batch_n_nodes)
    index = (index - tf.gather(start_index, I)) + (I * max_n_nodes)
    dense = tf.zeros((n_graphs * max_n_nodes, feature_dim), dtype=X.dtype)
    dense = tf.tensor_scatter_nd_update(dense, index[..., None], X)

    batch = tf.reshape(dense, (n_graphs, max_n_nodes, feature_dim))

    return batch


def _vectorised_get_cum_graph_size(nodes, graph_sizes):
    """
    Takes a list of node ids and graph sizes ordered by segment ID and returns the
    number of nodes contained in graphs with smaller segment ID.

    :param nodes: List of node ids of shape (nodes)
    :param graph_sizes: List of graph sizes (i.e. tf.math.segment_sum(tf.ones_like(I), I) where I are the
    segment IDs).
    :return: A list of shape (nodes) where each entry corresponds to the number of nodes contained in graphs
    with smaller segment ID for each node.
    """

    def get_cum_graph_size(node):
        cum_graph_sizes = tf.cumsum(graph_sizes, exclusive=True)
        indicator_if_smaller = tf.cast(node - cum_graph_sizes >= 0, tf.int32)
        graph_id = tf.reduce_sum(indicator_if_smaller) - 1
        return tf.cumsum(graph_sizes, exclusive=True)[graph_id]

    return tf.map_fn(get_cum_graph_size, nodes)


def disjoint_adjacency_to_batch(A, I):
    """
    Converts a disjoint adjacency matrix to batch node by zero-padding.

    :param A: Tensor, binary adjacency matrix of shape `(n_nodes, n_nodes)`;
    :param I: Tensor, graph IDs of shape `(n_nodes, )`;
    :return: Tensor, batched adjacency matrix of shape `(batch, N_max, N_max)`;
    """
    I = tf.cast(I, tf.int64)
    A = tf.cast(A, tf.float32)
    indices = A.indices
    values = tf.cast(A.values, tf.int64)
    i_nodes, j_nodes = indices[:, 0], indices[:, 1]

    graph_sizes = tf.math.segment_sum(tf.ones_like(I), I)
    max_n_nodes = tf.reduce_max(graph_sizes)
    n_graphs = tf.shape(graph_sizes)[0]
    relative_j_nodes = j_nodes - \
        _vectorised_get_cum_graph_size(j_nodes, graph_sizes)
    relative_i_nodes = i_nodes - \
        _vectorised_get_cum_graph_size(i_nodes, graph_sizes)
    spaced_i_nodes = I * max_n_nodes + relative_i_nodes
    new_indices = tf.transpose(tf.stack([spaced_i_nodes, relative_j_nodes]))

    new_indices = tf.cast(new_indices, tf.int32)
    n_graphs = tf.cast(n_graphs, tf.int32)
    max_n_nodes = tf.cast(max_n_nodes, tf.int32)

    dense_adjacency = tf.scatter_nd(
        new_indices, values, (n_graphs * max_n_nodes, max_n_nodes)
    )
    batch = tf.reshape(dense_adjacency, (n_graphs, max_n_nodes, max_n_nodes))
    batch = tf.cast(batch, tf.float32)
    return batch


def autodetect_mode(x, a):
    """
    Returns a code that identifies the data mode from the given node features
    and adjacency matrix(s).
    The output of this function can be used as follows:

    ```py
    from spektral.layers.ops import modes
    mode = modes.autodetect_mode(x, a)
    if mode == modes.SINGLE:
        print('Single!')
    elif mode == modes.BATCH:
        print('Batch!')
    elif mode == modes.MIXED:
        print('Mixed!')
    ```

    :param x: Tensor or SparseTensor representing the node features
    :param a: Tensor or SparseTensor representing the adjacency matrix(s)
    :return: mode of operation as an integer code.
    """
    x_ndim = K.ndim(x)
    a_ndim = K.ndim(a)
    if x_ndim == 2 and a_ndim == 2:
        return SINGLE
    elif x_ndim == 3 and a_ndim == 3:
        return BATCH
    elif x_ndim == 3 and a_ndim == 2:
        return MIXED
    else:
        raise ValueError(
            "Unknown mode for inputs x, a with ranks {} and {}"
            "respectively.".format(x_ndim, a_ndim)
        )


def dot(a, b):
    """
    Computes a @ b, for a, b of the same rank (both 2 or both 3).

    If the rank is 2, then the innermost dimension of `a` must match the
    outermost dimension of `b`.
    If the rank is 3, the first dimension of `a` and `b` must be equal and the
    function computes a batch matmul.

    Supports both dense and sparse multiplication (including sparse-sparse).

    :param a: Tensor or SparseTensor with rank 2 or 3.
    :param b: Tensor or SparseTensor with same rank as b.
    :return: Tensor or SparseTensor with rank 2 or 3.
    """
    a_ndim = K.ndim(a)
    b_ndim = K.ndim(b)
    assert a_ndim == b_ndim, "Expected equal ranks, got {} and {}" "".format(
        a_ndim, b_ndim
    )
    a_is_sparse = K.is_sparse(a)
    b_is_sparse = K.is_sparse(b)

    # Handle cases: rank 2 sparse-dense, rank 2 dense-sparse
    # In these cases we can use the faster sparse-dense matmul of tf.sparse
    if a_ndim == 2:
        if a_is_sparse and not b_is_sparse:
            return tf.sparse.sparse_dense_matmul(a, b)
        if not a_is_sparse and b_is_sparse:
            return transpose(tf.sparse.sparse_dense_matmul(transpose(b), transpose(a)))

    # Handle cases: rank 2 sparse-sparse, rank 3 sparse-dense,
    # rank 3 dense-sparse, rank 3 sparse-sparse
    # In these cases we can use the tfsp.CSRSparseMatrix implementation (slower,
    # but saves memory)
    if a_is_sparse:
        a = tfsp.CSRSparseMatrix(a)
    if b_is_sparse:
        b = tfsp.CSRSparseMatrix(b)
    if a_is_sparse or b_is_sparse:
        out = tfsp.matmul(a, b)
        if hasattr(out, "to_sparse_tensor"):
            return out.to_sparse_tensor()
        else:
            return out

    # Handle case: rank 2 dense-dense, rank 3 dense-dense
    # Here we use the standard dense operation
    return tf.matmul(a, b)


def mixed_mode_dot(a, b):
    """
    Computes the equivalent of `tf.einsum('ij,bjk->bik', a, b)`, but
    works for both dense and sparse inputs.

    :param a: Tensor or SparseTensor with rank 2.
    :param b: Tensor or SparseTensor with rank 3.
    :return: Tensor or SparseTensor with rank 3.
    """
    shp = K.int_shape(b)
    b_t = transpose(b, (1, 2, 0))
    b_t = reshape(b_t, (shp[1], -1))
    output = dot(a, b_t)
    output = reshape(output, (shp[1], shp[2], -1))
    output = transpose(output, (2, 0, 1))

    return output


def modal_dot(a, b, transpose_a=False, transpose_b=False):
    """
    Computes the matrix multiplication of a and b, handling the data modes
    automatically.

    This is a wrapper to standard matmul operations, for a and b with rank 2
    or 3, that:

    - Supports automatic broadcasting of the "batch" dimension if the two inputs
    have different ranks.
    - Supports any combination of dense and sparse inputs.

    This op is useful for multiplying matrices that represent batches of graphs
    in the different modes, for which the adjacency matrices may or may not be
    sparse and have different ranks from the node attributes.

    Additionally, it can also support the case where we have many adjacency
    matrices and only one graph signal (which is uncommon, but may still happen).

    If you know a-priori the type and shape of the inputs, it may be faster to
    use the built-in functions of TensorFlow directly instead.

    Examples:

        - `a` rank 2, `b` rank 2 -> `a @ b`
        - `a` rank 3, `b` rank 3 -> `[a[i] @ b[i] for i in range(len(a))]`
        - `a` rank 2, `b` rank 3 -> `[a @ b[i] for i in range(len(b))]`
        - `a` rank 3, `b` rank 2 -> `[a[i] @ b for i in range(len(a))]`

    :param a: Tensor or SparseTensor with rank 2 or 3;
    :param b: Tensor or SparseTensor with rank 2 or 3;
    :param transpose_a: transpose the innermost 2 dimensions of `a`;
    :param transpose_b: transpose the innermost 2 dimensions of `b`;
    :return: Tensor or SparseTensor with rank = max(rank(a), rank(b)).
    """
    a_ndim = K.ndim(a)
    b_ndim = K.ndim(b)
    assert a_ndim in (2, 3), "Expected a of rank 2 or 3, got {}".format(a_ndim)
    assert b_ndim in (2, 3), "Expected b of rank 2 or 3, got {}".format(b_ndim)

    if transpose_a:
        perm = None if a_ndim == 2 else (0, 2, 1)
        a = transpose(a, perm)
    if transpose_b:
        perm = None if b_ndim == 2 else (0, 2, 1)
        b = transpose(b, perm)

    if a_ndim == b_ndim:
        # ...ij,...jk->...ik
        return dot(a, b)
    elif a_ndim == 2:
        # ij,bjk->bik
        return mixed_mode_dot(a, b)
    else:  # a_ndim == 3
        # bij,jk->bik
        if not K.is_sparse(a) and not K.is_sparse(b):
            # Immediately fallback to standard dense matmul, no need to reshape
            return tf.matmul(a, b)

        # If either input is sparse, we use dot(a, b)
        # This implementation is faster than using rank 3 sparse matmul with tfsp
        a_shape = tf.shape(a)
        b_shape = tf.shape(b)
        a_flat = reshape(a, (-1, a_shape[2]))
        output = dot(a_flat, b)
        return reshape(output, (-1, a_shape[1], b_shape[1]))


def matmul_at_b_a(a, b):
    """
    Computes a.T @ b @ a, for a, b with rank 2 or 3.

    Supports automatic broadcasting of the "batch" dimension if the two inputs
    have different ranks.
    Supports any combination of dense and sparse inputs.

    :param a: Tensor or SparseTensor with rank 2 or 3.
    :param b: Tensor or SparseTensor with rank 2 or 3.
    :return: Tensor or SparseTensor with rank = max(rank(a), rank(b)).
    """
    at_b = modal_dot(a, b, transpose_a=True)
    at_b_a = modal_dot(at_b, a)

    return at_b_a


def matrix_power(a, k):
    """
    If a is a square matrix, computes a^k. If a is a rank 3 Tensor of square
    matrices, computes the exponent of each inner matrix.

    :param a: Tensor or SparseTensor with rank 2 or 3. The innermost two
    dimensions must be the same.
    :param k: int, the exponent to which to raise the matrices.
    :return: Tensor or SparseTensor with same rank as the input.
    """
    x_k = a
    for _ in range(k - 1):
        x_k = modal_dot(a, x_k)

    return x_k


def mixed_mode_support(scatter_fn):
    def _wrapper_mm_support(updates, indices, N):
        if len(tf.shape(updates)) == 3:
            updates = tf.transpose(updates, perm=(1, 0, 2))
        out = scatter_fn(updates, indices, N)
        if len(tf.shape(out)) == 3:
            out = tf.transpose(out, perm=(1, 0, 2))
        return out

    _wrapper_mm_support.__name__ = scatter_fn.__name__
    return _wrapper_mm_support


@mixed_mode_support
def scatter_sum(messages, indices, n_nodes):
    """
    Sums messages according to the segments defined by `indices`, with
    support for messages in single/disjoint mode (rank 2) and mixed mode (rank 3).
    The output has the same rank as the input, with the "nodes" dimension
    changed to `n_nodes`.

    For single/disjoint mode, messages are expected to have shape
    `[n_messages, n_features]` and the output will have shape
    `[n_nodes, n_features]`.

    For mixed mode, messages are expected to have shape
    `[batch, n_messages, n_features]` and the output will have shape
    `[batch, n_nodes, n_features]`.

    Indices are always expected to be a 1D tensor of integers < `n_nodes`, with
    shape '[n_messages]'.
    For any missing index (i.e., any integer `0 <= i < n_nodes` that does not
    appear in `indices`) the corresponding output will be all zeros.
    If a given index `i` is negative, it is ignored in the aggregation.

    :param messages: a 2D or 3D Tensor.
    :param indices: A 1D Tensor with indices into the "nodes" dimension of the
    messages.
    :param n_nodes: dimension of the output along the "nodes" dimension.
    :return: a Tensor with the same rank as `messages`.
    """
    return tf.math.unsorted_segment_sum(messages, indices, n_nodes)


@mixed_mode_support
def scatter_mean(messages, indices, n_nodes):
    """
    Averages messages according to the segments defined by `indices`, with
    support for messages in single/disjoint mode (rank 2) and mixed mode (rank 3).
    The output has the same rank as the input, with the "nodes" dimension
    changed to `n_nodes`.

    For single/disjoint mode, messages are expected to have shape
    `[n_messages, n_features]` and the output will have shape
    `[n_nodes, n_features]`.

    For mixed mode, messages are expected to have shape
    `[batch, n_messages, n_features]` and the output will have shape
    `[batch, n_nodes, n_features]`.

    Indices are always expected to be a 1D tensor of integers < `n_nodes`, with
    shape '[n_messages]'.
    For any missing index (i.e., any integer `0 <= i < n_nodes` that does not
    appear in `indices`) the corresponding output will be all zeros.
    If a given index `i` is negative, it is ignored in the aggregation.

    :param messages: a 2D or 3D Tensor.
    :param indices: A 1D Tensor with indices into the "nodes" dimension of the
    messages.
    :param n_nodes: dimension of the output along the "nodes" dimension.
    :return: a Tensor with the same rank as `messages`.
    """
    return tf.math.unsorted_segment_mean(messages, indices, n_nodes)


@mixed_mode_support
def scatter_max(messages, indices, n_nodes):
    """
    Max-reduces messages according to the segments defined by `indices`, with
    support for messages in single/disjoint mode (rank 2) and mixed mode (rank 3).
    The output has the same rank as the input, with the "nodes" dimension
    changed to `n_nodes`.

    For single/disjoint mode, messages are expected to have shape
    `[n_messages, n_features]` and the output will have shape
    `[n_nodes, n_features]`.

    For mixed mode, messages are expected to have shape
    `[batch, n_messages, n_features]` and the output will have shape
    `[batch, n_nodes, n_features]`.

    Indices are always expected to be a 1D tensor of integers < `n_nodes`, with
    shape '[n_messages]'.
    For any missing index (i.e., any integer `0 <= i < n_nodes` that does not
    appear in `indices`) the corresponding output will be the minimum possible
    value for the dtype of the messages
    If a given index `i` is negative, it is ignored in the aggregation.

    :param messages: a 2D or 3D Tensor.
    :param indices: A 1D Tensor with indices into the "nodes" dimension of the
    messages.
    :param n_nodes: dimension of the output along the "nodes" dimension.
    :return: a Tensor with the same rank as `messages`.
    """
    return tf.math.unsorted_segment_max(messages, indices, n_nodes)


@mixed_mode_support
def scatter_min(messages, indices, n_nodes):
    """
    Min-reduces messages according to the segments defined by `indices`, with
    support for messages in single/disjoint mode (rank 2) and mixed mode (rank 3).
    The output has the same rank as the input, with the "nodes" dimension
    changed to `n_nodes`.

    For single/disjoint mode, messages are expected to have shape
    `[n_messages, n_features]` and the output will have shape
    `[n_nodes, n_features]`.

    For mixed mode, messages are expected to have shape
    `[batch, n_messages, n_features]` and the output will have shape
    `[batch, n_nodes, n_features]`.

    Indices are always expected to be a 1D tensor of integers < `n_nodes`, with
    shape '[n_messages]'.
    For any missing index (i.e., any integer `0 <= i < n_nodes` that does not
    appear in `indices`) the corresponding output will be the maximum possible
    value for the dtype of the messages.
    If a given index `i` is negative, it is ignored in the aggregation.

    :param messages: a 2D or 3D Tensor.
    :param indices: A 1D Tensor with indices into the "nodes" dimension of the
    messages.
    :param n_nodes: dimension of the output along the "nodes" dimension.
    :return: a Tensor with the same rank as `messages`.
    """
    return tf.math.unsorted_segment_min(messages, indices, n_nodes)


@mixed_mode_support
def scatter_prod(messages, indices, n_nodes):
    """
    Multiplies messages element-wise according to the segments defined by
    `indices`, with support for messages in single/disjoint mode (rank 2)
    and mixed mode (rank 3).
    The output has the same rank as the input, with the "nodes" dimension
    changed to `n_nodes`.

    For single/disjoint mode, messages are expected to have shape
    `[n_messages, n_features]` and the output will have shape
    `[n_nodes, n_features]`.

    For mixed mode, messages are expected to have shape
    `[batch, n_messages, n_features]` and the output will have shape
    `[batch, n_nodes, n_features]`.

    Indices are always expected to be a 1D tensor of integers < `n_nodes`, with
    shape '[n_messages]'.
    For any missing index (i.e., any integer `0 <= i < n_nodes` that does not
    appear in `indices`) the corresponding output will be all ones.
    If a given index `i` is negative, it is ignored in the aggregation.

    :param messages: a 2D or 3D Tensor.
    :param indices: A 1D Tensor with indices into the "nodes" dimension of the
    messages.
    :param n_nodes: dimension of the output along the "nodes" dimension.
    :return: a Tensor with the same rank as `messages`.
    """
    return tf.math.unsorted_segment_prod(messages, indices, n_nodes)


OP_DICT = {
    "sum": scatter_sum,
    "mean": scatter_mean,
    "max": scatter_max,
    "min": scatter_min,
    "prod": scatter_prod,
}


def unsorted_segment_softmax(x, indices, n_nodes=None):
    """
    Applies softmax along the segments of a Tensor. This operator is similar
    to the tf.math.segment_* operators, which apply a certain reduction to the
    segments. In this case, the output tensor is not reduced and maintains the
    same shape as the input.
    :param x: a Tensor. The softmax is applied along the first dimension.
    :param indices: a Tensor, indices to the segments.
    :param n_nodes: the number of unique segments in the indices. If `None`,
    n_nodes is calculated as the maximum entry in the indices plus 1.
    :return: a Tensor with the same shape as the input.
    """
    n_nodes = tf.reduce_max(indices) + 1 if n_nodes is None else n_nodes
    e_x = tf.exp(
        x - tf.gather(tf.math.unsorted_segment_max(x,
                                                   indices, n_nodes), indices)
    )
    e_x /= tf.gather(
        tf.math.unsorted_segment_sum(e_x, indices, n_nodes) + 1e-9, indices
    )
    return e_x


def serialize_scatter(identifier):
    if identifier in OP_DICT:
        return identifier
    elif hasattr(identifier, "__name__"):
        for k, v in OP_DICT.items():
            if v.__name__ == identifier.__name__:
                return k
        return None


def deserialize_scatter(scatter):
    if isinstance(scatter, str):
        if scatter in OP_DICT:
            return OP_DICT[scatter]
        else:
            if callable(scatter):
                return scatter
            else:
                raise ValueError(
                    "scatter must be callable or string in: {}.".format(
                        list(OP_DICT.keys())
                    )
                )


def sp_matrix_to_sp_tensor(x):
    """
    Converts a Scipy sparse matrix to a SparseTensor.
    :param x: a Scipy sparse matrix.
    :return: a SparseTensor.
    """
    if len(x.shape) != 2:
        raise ValueError("x must have rank 2")
    row, col, values = sp.find(x)
    out = tf.SparseTensor(
        indices=np.array([row, col]).T, values=values, dense_shape=x.shape
    )
    return tf.sparse.reorder(out)


def sp_batch_to_sp_tensor(a_list):
    """
    Converts a list of Scipy sparse matrices to a rank 3 SparseTensor.
    :param a_list: list of Scipy sparse matrices with the same shape.
    :return: SparseTensor of rank 3.
    """
    tensor_data = []
    for i, a in enumerate(a_list):
        row, col, values = sp.find(a)
        batch = np.ones_like(col) * i
        tensor_data.append((values, batch, row, col))
    tensor_data = list(map(np.concatenate, zip(*tensor_data)))

    out = tf.SparseTensor(
        indices=np.array(tensor_data[1:]).T,
        values=tensor_data[0],
        dense_shape=(len(a_list),) + a_list[0].shape,
    )

    return out


def dense_to_sparse(x):
    """
    Converts a Tensor to a SparseTensor.
    :param x: a Tensor.
    :return: a SparseTensor.
    """
    indices = tf.where(tf.not_equal(x, 0))
    values = tf.gather_nd(x, indices)
    shape = tf.shape(x, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)


def add_self_loops(a, fill=1.0):
    """
    Adds self-loops to the given adjacency matrix. Self-loops are added only for
    those node that don't have a self-loop already, and are assigned a weight
    of `fill`.
    :param a: a square SparseTensor.
    :param fill: the fill value for the new self-loops. It will be cast to the
    dtype of `a`.
    :return: a SparseTensor with the same shape as the input.
    """
    indices = a.indices
    values = a.values
    N = tf.shape(a, out_type=indices.dtype)[0]

    mask_od = indices[:, 0] != indices[:, 1]
    mask_sl = ~mask_od
    mask_od.set_shape([None])  # For compatibility with TF 2.2
    mask_sl.set_shape([None])

    indices_od = indices[mask_od]
    indices_sl = indices[mask_sl]

    values_sl = tf.fill((N,), tf.cast(fill, values.dtype))
    values_sl = tf.tensor_scatter_nd_update(
        values_sl, indices_sl[:, 0:1], values[mask_sl]
    )

    indices_sl = tf.range(N, dtype=indices.dtype)[:, None]
    indices_sl = tf.repeat(indices_sl, 2, -1)
    indices = tf.concat((indices_od, indices_sl), 0)

    values_od = values[mask_od]
    values = tf.concat((values_od, values_sl), 0)

    out = tf.SparseTensor(indices, values, (N, N))

    return tf.sparse.reorder(out)


def add_self_loops_indices(indices, n_nodes=None):
    """
    Given the indices of a square SparseTensor, adds the diagonal entries (i, i)
    and returns the reordered indices.
    :param indices: Tensor of rank 2, the indices to a SparseTensor.
    :param n_nodes: the size of the n_nodes x n_nodes SparseTensor indexed by
    the indices. If `None`, n_nodes is calculated as the maximum entry in the
    indices plus 1.
    :return: Tensor of rank 2, the indices to a SparseTensor.
    """
    n_nodes = tf.reduce_max(indices) + 1 if n_nodes is None else n_nodes
    row, col = indices[..., 0], indices[..., 1]
    mask = tf.ensure_shape(row != col, row.shape)
    sl_indices = tf.range(n_nodes, dtype=row.dtype)[:, None]
    sl_indices = tf.repeat(sl_indices, 2, -1)
    indices = tf.concat((indices[mask], sl_indices), 0)
    dummy_values = tf.ones_like(indices[:, 0])
    indices, _ = gen_sparse_ops.sparse_reorder(
        indices, dummy_values, (n_nodes, n_nodes)
    )
    return indices


def _square_size(dense_shape):
    dense_shape = tf.unstack(dense_shape)
    size = dense_shape[0]
    for d in dense_shape[1:]:
        tf.debugging.assert_equal(size, d)
    return d


def _indices_to_inverse_map(indices, size):
    """
    Compute inverse indices of a gather.
    :param indices: Tensor, forward indices, rank 1
    :param size: Tensor, size of pre-gathered input, rank 0
    :return: Tensor, inverse indices, shape [size]. Zero values everywhere
    except at indices.
    """
    indices = tf.cast(indices, tf.int64)
    size = tf.cast(size, tf.int64)
    return tf.scatter_nd(
        tf.expand_dims(indices, axis=-1),
        tf.range(tf.shape(indices, out_type=tf.int64)[0]),
        tf.expand_dims(size, axis=-1),
    )


def _boolean_mask_sparse(a, mask, axis, inverse_map, out_size):
    """
    SparseTensor equivalent to tf.boolean_mask.
    :param a: SparseTensor of rank k and nnz non-zeros.
    :param mask: rank-1 bool Tensor.
    :param axis: int, axis on which to mask. Must be in [-k, k).
    :param out_size: number of true entires in mask. Computed if not given.
    :return masked_a: SparseTensor masked along the given axis.
    :return values_mask: bool Tensor indicating surviving edges, shape [nnz].
    """
    mask = tf.convert_to_tensor(mask)
    values_mask = tf.gather(mask, a.indices[:, axis], axis=0)
    dense_shape = tf.tensor_scatter_nd_update(
        a.dense_shape, [[axis]], [out_size])
    indices = tf.boolean_mask(a.indices, values_mask)
    indices = tf.unstack(indices, axis=-1)
    indices[axis] = tf.gather(inverse_map, indices[axis])
    indices = tf.stack(indices, axis=-1)
    a = tf.SparseTensor(
        indices,
        tf.boolean_mask(a.values, values_mask),
        dense_shape,
    )
    return (a, values_mask)


def _boolean_mask_sparse_square(a, mask, inverse_map, out_size):
    """
    Apply boolean_mask to every axis of a SparseTensor.
    :param a: SparseTensor with uniform dimensions and nnz non-zeros.
    :param mask: boolean mask.
    :param inverse_map: Tensor of new indices, shape [nnz]. Computed if None.
    :out_size: number of True values in mask. Computed if None.
    :return a: SparseTensor with uniform dimensions.
    :return values_mask: bool Tensor of shape [nnz] indicating valid edges.
    """
    mask = tf.convert_to_tensor(mask)
    values_mask = tf.reduce_all(tf.gather(mask, a.indices, axis=0), axis=-1)
    dense_shape = [out_size] * a.shape.ndims
    indices = tf.boolean_mask(a.indices, values_mask)
    indices = tf.gather(inverse_map, indices)
    a = tf.SparseTensor(indices, tf.boolean_mask(
        a.values, values_mask), dense_shape)
    return (a, values_mask)


def boolean_mask_sparse(a, mask, axis=0):
    """
    SparseTensor equivalent to tf.boolean_mask.
    :param a: SparseTensor of rank k and nnz non-zeros.
    :param mask: rank-1 bool Tensor.
    :param axis: int, axis on which to mask. Must be in [-k, k).
    :return masked_a: SparseTensor masked along the given axis.
    :return values_mask: bool Tensor indicating surviving values, shape [nnz].
    """
    i = tf.squeeze(tf.where(mask), axis=1)
    out_size = tf.math.count_nonzero(mask)
    in_size = a.dense_shape[axis]
    inverse_map = _indices_to_inverse_map(i, in_size)
    return _boolean_mask_sparse(
        a, mask, axis=axis, inverse_map=inverse_map, out_size=out_size
    )


def boolean_mask_sparse_square(a, mask):
    """
    Apply mask to every axis of SparseTensor a.
    :param a: SparseTensor, square, nnz non-zeros.
    :param mask: boolean mask with size equal to each dimension of a.
    :return masked_a: SparseTensor
    :return values_mask: bool tensor of shape [nnz] indicating valid values.
    """
    i = tf.squeeze(tf.where(mask), axis=-1)
    out_size = tf.size(i)
    in_size = _square_size(a.dense_shape)
    inverse_map = _indices_to_inverse_map(i, in_size)
    return _boolean_mask_sparse_square(
        a, mask, inverse_map=inverse_map, out_size=out_size
    )


def gather_sparse(a, indices, axis=0, mask=None):
    """
    SparseTensor equivalent to tf.gather, assuming indices are sorted.
    :param a: SparseTensor of rank k and nnz non-zeros.
    :param indices: rank-1 int Tensor, rows or columns to keep.
    :param axis: int axis to apply gather to.
    :param mask: boolean mask corresponding to indices. Computed if not provided.
    :return gathered_a: SparseTensor masked along the given axis.
    :return values_mask: bool Tensor indicating surviving values, shape [nnz].
    """
    in_size = _square_size(a.dense_shape)
    out_size = tf.size(indices)
    if mask is None:
        mask = indices_to_mask(indices, in_size)
    inverse_map = _indices_to_inverse_map(indices, in_size)
    return _boolean_mask_sparse(
        a, mask, axis=axis, inverse_map=inverse_map, out_size=out_size
    )


def gather_sparse_square(a, indices, mask=None):
    """
    Gather on every axis of a SparseTensor.
    :param a: SparseTensor of rank k and nnz non-zeros.
    :param indices: rank-1 int Tensor, rows and columns to keep.
    :param mask: boolean mask corresponding to indices. Computed if not provided.
    :return gathered_a: SparseTensor of the gathered input.
    :return values_mask: bool Tensor indicating surviving values, shape [nnz].
    """
    in_size = _square_size(a.dense_shape)
    out_size = tf.size(indices)
    if mask is None:
        mask = indices_to_mask(indices, in_size)
    inverse_map = _indices_to_inverse_map(indices, in_size)
    return _boolean_mask_sparse_square(
        a, mask, inverse_map=inverse_map, out_size=out_size
    )


def normalize_A(A):
    """
    Computes symmetric normalization of A, dealing with sparse A and batch mode
    automatically.
    :param A: Tensor or SparseTensor with rank k = {2, 3}.
    :return: Tensor or SparseTensor of rank k.
    """
    D = degrees(A)
    D = tf.sqrt(D)[:, None] + K.epsilon()
    perm = (0, 2, 1) if K.ndim(A) == 3 else (1, 0)
    output = (A / D) / transpose(D, perm=perm)

    return output


def degrees(A):
    """
    Computes the degrees of each node in A, dealing with sparse A and batch mode
    automatically.
    :param A: Tensor or SparseTensor with rank k = {2, 3}.
    :return: Tensor or SparseTensor of rank k - 1.
    """
    if K.is_sparse(A):
        D = tf.sparse.reduce_sum(A, axis=-1)
    else:
        D = tf.reduce_sum(A, axis=-1)

    return D


def degree_matrix(A, return_sparse_batch=False):
    """
    Computes the degree matrix of A, deals with sparse A and batch mode
    automatically.
    :param A: Tensor or SparseTensor with rank k = {2, 3}.
    :param return_sparse_batch: if operating in batch mode, return a
    SparseTensor. Note that the sparse degree Tensor returned by this function
    cannot be used for sparse matrix multiplication afterwards.
    :return: SparseTensor of rank k.
    """
    D = degrees(A)

    batch_mode = K.ndim(D) == 2
    N = tf.shape(D)[-1]
    batch_size = tf.shape(D)[0] if batch_mode else 1

    inner_index = tf.tile(tf.stack([tf.range(N)] * 2, axis=1), (batch_size, 1))
    if batch_mode:
        if return_sparse_batch:
            outer_index = repeat(
                tf.range(batch_size), tf.ones(
                    batch_size) * tf.cast(N, tf.float32)
            )
            indices = tf.concat([outer_index[:, None], inner_index], 1)
            dense_shape = (batch_size, N, N)
        else:
            return tf.linalg.diag(D)
    else:
        indices = inner_index
        dense_shape = (N, N)

    indices = tf.cast(indices, tf.int64)
    values = tf.reshape(D, (-1,))
    return tf.SparseTensor(indices, values, dense_shape)
