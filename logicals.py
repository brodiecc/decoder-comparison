import numpy as np
from scipy.sparse import csc_matrix
__all__ = [
    "create_logical",
    "create_logical_2",
    "create_logical_vertical_horizontal",
]


def create_logical(L, basis):
    """
    Diagonal logicals on an LxL grid with row-major indexing q = r*L + c.

    - X logical: top-left -> bottom-right (main diagonal)
    - Z logical: bottom-left -> top-right (anti-diagonal)

    Returns:
        csc_matrix of shape (1, L^2) with uint8 entries.
    """
    basis = basis.lower()
    n = L * L

    if basis == "x":
        # (r,c) = (i,i)
        cols = np.arange(L, dtype=np.int32) * (L + 1)
    elif basis == "z":
        # (r,c) = (L-1-i, i)
        i = np.arange(L, dtype=np.int32)
        cols = (L - 1 - i) * L + i
    else:
        raise ValueError("Basis must be 'x' or 'z'")

    data = np.ones(L, dtype=np.uint8)
    rows = np.zeros(L, dtype=np.int32)  # single-row matrix
    return csc_matrix((data, (rows, cols)), shape=(1, n), dtype=np.uint8)


def create_logical_2(L, basis):
    """
    Boundary-style logicals on an LxL grid with row-major indexing q = r*L + c.

    - X logical: vertical line along the left boundary (c = 0)
    - Z logical: horizontal line along the top boundary (r = 0)

    Returns:
        csc_matrix of shape (1, L^2) with uint8 entries.
    """
    basis = basis.lower()
    n = L * L

    if basis == "x":
        # (r, c) = (i, 0)
        cols = np.arange(L, dtype=np.int32) * L
    elif basis == "z":
        # (r, c) = (0, i)
        cols = np.arange(L, dtype=np.int32)
    else:
        raise ValueError("Basis must be 'x' or 'z'")

    data = np.ones(L, dtype=np.uint8)
    rows = np.zeros(L, dtype=np.int32)  # single-row matrix
    return csc_matrix((data, (rows, cols)), shape=(1, n), dtype=np.uint8)


def create_logical_vertical_horizontal(L, basis):
    """
    Backward-compatible alias for create_logical_2.
    """
    return create_logical_2(L, basis)
