import copy
import warnings

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence
from tensorflow.keras import (activations, constraints, initializers,
                              regularizers)

LAYER_KWARGS = {"activation", "use_bias"}
KERAS_KWARGS = {
    "activity_regularizer",
    "autocast",
    "batch_input_shape",
    "batch_size",
    "input_shape",
    "weights",
}


def is_layer_kwarg(key):
    return key not in KERAS_KWARGS and (
        key.endswith("_initializer")
        or key.endswith("_regularizer")
        or key.endswith("_constraint")
        or key in LAYER_KWARGS
    )


def is_keras_kwarg(key):
    return key in KERAS_KWARGS


def deserialize_kwarg(key, attr):
    if key.endswith("_initializer"):
        return initializers.get(attr)
    if key.endswith("_regularizer"):
        return regularizers.get(attr)
    if key.endswith("_constraint"):
        return constraints.get(attr)
    if key == "activation":
        return activations.get(attr)


def serialize_kwarg(key, attr):
    if key.endswith("_initializer"):
        return initializers.serialize(attr)
    if key.endswith("_regularizer"):
        return regularizers.serialize(attr)
    if key.endswith("_constraint"):
        return constraints.serialize(attr)
    if key == "activation":
        return activations.serialize(attr)
    if key == "use_bias":
        return attr


def degree_matrix(A):
    """
    Computes the degree matrix of the given adjacency matrix.
    :param A: rank 2 array or sparse matrix.
    :return: if A is a dense array, a dense array; if A is sparse, a sparse
    matrix in DIA format.
    """
    degrees = np.array(A.sum(1)).flatten()
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def degree_power(A, k):
    r"""
    Computes \(\D^{k}\) from the given adjacency matrix. Useful for computing
    normalised Laplacian.
    :param A: rank 2 array or sparse matrix.
    :param k: exponent to which elevate the degree matrix.
    :return: if A is a dense array, a dense array; if A is sparse, a sparse
    matrix in DIA format.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        degrees = np.power(np.array(A.sum(1)), k).ravel()
    degrees[np.isinf(degrees)] = 0.0
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def normalized_adjacency(A, symmetric=True):
    r"""
    Normalizes the given adjacency matrix using the degree matrix as either
    \(\D^{-1}\A\) or \(\D^{-1/2}\A\D^{-1/2}\) (symmetric normalization).
    :param A: rank 2 array or sparse matrix;
    :param symmetric: boolean, compute symmetric normalization;
    :return: the normalized adjacency matrix.
    """
    if symmetric:
        normalized_D = degree_power(A, -0.5)
        return normalized_D.dot(A).dot(normalized_D)
    else:
        normalized_D = degree_power(A, -1.0)
        return normalized_D.dot(A)


def laplacian(A):
    r"""
    Computes the Laplacian of the given adjacency matrix as \(\D - \A\).
    :param A: rank 2 array or sparse matrix;
    :return: the Laplacian.
    """
    return degree_matrix(A) - A


def normalized_laplacian(A, symmetric=True):
    r"""
    Computes a  normalized Laplacian of the given adjacency matrix as
    \(\I - \D^{-1}\A\) or \(\I - \D^{-1/2}\A\D^{-1/2}\) (symmetric normalization).
    :param A: rank 2 array or sparse matrix;
    :param symmetric: boolean, compute symmetric normalization;
    :return: the normalized Laplacian.
    """
    if sp.issparse(A):
        I = sp.eye(A.shape[-1], dtype=A.dtype)
    else:
        I = np.eye(A.shape[-1], dtype=A.dtype)
    normalized_adj = normalized_adjacency(A, symmetric=symmetric)
    return I - normalized_adj


def rescale_laplacian(L, lmax=None):
    """
    Rescales the Laplacian eigenvalues in [-1,1], using lmax as largest eigenvalue.
    :param L: rank 2 array or sparse matrix;
    :param lmax: if None, compute largest eigenvalue with scipy.linalg.eisgh.
    If the eigendecomposition fails, lmax is set to 2 automatically.
    If scalar, use this value as largest eigenvalue when rescaling.
    :return:
    """
    if lmax is None:
        try:
            lmax = sp.linalg.eigsh(
                L, 1, which="LM", return_eigenvectors=False)[0]
        except ArpackNoConvergence:
            lmax = 2
    if sp.issparse(L):
        I = sp.eye(L.shape[-1], dtype=L.dtype)
    else:
        I = np.eye(L.shape[-1], dtype=L.dtype)
    L_scaled = (2.0 / lmax) * L - I
    return L_scaled


def gcn_filter(A, symmetric=True):
    r"""
    Computes the graph filter described in
    [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907).
    :param A: array or sparse matrix with rank 2 or 3;
    :param symmetric: boolean, whether to normalize the matrix as
    \(\D^{-\frac{1}{2}}\A\D^{-\frac{1}{2}}\) or as \(\D^{-1}\A\);
    :return: array or sparse matrix with rank 2 or 3, same as A;
    """
    out = copy.deepcopy(A)
    if isinstance(A, list) or (isinstance(A, np.ndarray) and A.ndim == 3):
        for i in range(len(A)):
            out[i] = A[i]
            out[i][np.diag_indices_from(out[i])] += 1
            out[i] = normalized_adjacency(out[i], symmetric=symmetric)
    else:
        if hasattr(out, "tocsr"):
            out = out.tocsr()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[np.diag_indices_from(out)] += 1
        out = normalized_adjacency(out, symmetric=symmetric)

    if sp.issparse(out):
        out.sort_indices()
    return out


def chebyshev_polynomial(X, k):
    """
    Calculates Chebyshev polynomials of X, up to order k.
    :param X: rank 2 array or sparse matrix;
    :param k: the order up to which compute the polynomials,
    :return: a list of k + 1 arrays or sparse matrices with one element for each
    degree of the polynomial.
    """
    T_k = list()
    if sp.issparse(X):
        T_k.append(sp.eye(X.shape[0], dtype=X.dtype).tocsr())
    else:
        T_k.append(np.eye(X.shape[0], dtype=X.dtype))
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        if sp.issparse(X):
            X_ = sp.csr_matrix(X, copy=True)
        else:
            X_ = np.copy(X)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def chebyshev_filter(A, k, symmetric=True):
    r"""
    Computes the Chebyshev filter from the given adjacency matrix, as described
    in [Defferrard et at. (2016)](https://arxiv.org/abs/1606.09375).
    :param A: rank 2 array or sparse matrix;
    :param k: integer, the order of the Chebyshev polynomial;
    :param symmetric: boolean, whether to normalize the matrix as
    \(\D^{-\frac{1}{2}}\A\D^{-\frac{1}{2}}\) or as \(\D^{-1}\A\);
    :return: a list of k + 1 arrays or sparse matrices with one element for each
    degree of the polynomial.
    """
    normalized_adj = normalized_adjacency(A, symmetric)
    if sp.issparse(A):
        I = sp.eye(A.shape[0], dtype=A.dtype)
    else:
        I = np.eye(A.shape[0], dtype=A.dtype)
    L = I - normalized_adj  # Compute Laplacian

    # Rescale Laplacian
    L_scaled = rescale_laplacian(L)

    # Compute Chebyshev polynomial approximation
    T_k = chebyshev_polynomial(L_scaled, k)

    # Sort indices
    if sp.issparse(T_k[0]):
        for i in range(len(T_k)):
            T_k[i].sort_indices()

    return T_k


def add_self_loops(a, value=1):
    """
    Sets the inner diagonals of `a` to `value`.
    :param a: a np.array or scipy.sparse matrix, the innermost two dimensions
    must be equal.
    :param value: value to set the diagonals to.
    :return: a np.array or scipy.sparse matrix with the same shape as `a`.
    """
    a = a.copy()
    if len(a.shape) < 2:
        raise ValueError("a must have at least rank 2")
    n = a.shape[-1]
    if n != a.shape[-2]:
        raise ValueError(
            "Innermost two dimensions must be equal. Got {}".format(a.shape))
    if sp.issparse(a):
        a = a.tolil()
        a.setdiag(value)
        return a.tocsr()
    else:
        idx = np.arange(n)
        a[..., idx, idx] = value
        return a
