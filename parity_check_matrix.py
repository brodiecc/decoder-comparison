import numpy as np
from scipy.sparse import csc_matrix


def create_hx_indices(L):
    x = (L - 1) // 2
    
    # 1st Pattern Base
    start = np.array([0, 0], dtype=np.int32)
    end = np.array([2*x - 1, 2*x - 1], dtype=np.int32)
    
    # Middle Pairs
    if x > 1:
        j1 = np.arange(1, x, dtype=np.int32)
        v1 = x + j1 - 1
        v2 = j1
        # column_stack pairs, ravel flattens [v1, v2, v1, v2...]
        middle = np.column_stack((v1, v2, v1, v2)).ravel()
        part1 = np.concatenate((start, middle, end))
    else:
        part1 = np.concatenate((start, end))
        
    # 2nd Pattern
    j2 = np.arange(x, dtype=np.int32)
    v1_2 = 2*x + j2
    v2_2 = x + j2
    part2 = np.column_stack((v1_2, v2_2, v1_2, v2_2)).ravel()
    
    # Combine into Repeating Block
    base_block = np.concatenate((part1, part2))
     
    # Tile and Offset via Broadcasting
    # We have x full blocks. Each block shifts its values up by (L - 1)
    offsets = np.arange(x, dtype=np.int32) * (2 * x)
    
    # base_block[None, :] + offsets[:, None] creates a 2D array of shifted blocks
    repeated_blocks = (base_block[None, :] + offsets[:, None]).ravel()
    
    # Cap it with the Final 1st Pattern
    final_offset = x * (2 * x)
    final_pattern = part1 + final_offset
    
    return np.concatenate((repeated_blocks, final_pattern))

def create_hz_indices(L):
    # Total number of base sequences
    N = L - 1

    # The step size between the start of each sequence
    y = (L + 1) // 2

    # Build all sequences simultaneously
    k_vals = np.arange(N, dtype=np.int32)[:, None]
    base_offsets = k_vals * y

    internal_offsets = np.empty((N, L), dtype=np.int32)
    # Even rows get [0, 0, 1, 1, 2...]
    internal_offsets[0::2, :] = np.arange(L, dtype=np.int32) // 2
    # Odd rows get [0, 1, 1, 2, 2...]
    internal_offsets[1::2, :] = np.arange(1, L + 1, dtype=np.int32) // 2

    # S contains all our base sequences stacked as a 2D matrix
    S = base_offsets + internal_offsets

    # Build the Middle Patterns
    # We pre-allocate a matrix M to hold the zipped pairs
    M = np.empty((N - 1, 2 * L), dtype=np.int32)

    # Instantly zip by dropping S_k into even columns and S_{k+1} into odd columns
    M[:, 0::2] = S[:-1]
    M[:, 1::2] = S[1:]

    # S[0] is the Start pattern, M.ravel() flattens the zipped middles, S[-1] is the End pattern
    return np.concatenate((S[0], M.ravel(), S[-1]))


def create_hx_indptr(L):
    # base difference block: [1, 2, 2, ..., 2, 1]
    diff_block = np.full(L, 2, dtype=np.int32)
    diff_block[0] = 1
    diff_block[-1] = 1

    # Repeat block L times
    diffs = np.tile(diff_block, L)

    # Pre-allocate the indptr array
    indptr = np.empty(L**2 + 1, dtype=np.int32)
    indptr[0] = 0

    # Fill the rest by taking the cumulative sum of the differences
    indptr[1:] = np.cumsum(diffs)

    return indptr

def create_hz_indptr(L):
    # Pre-allocate the differences array
    diffs = np.full(L**2, 2, dtype=np.int32)

    # 2. The top boundary and bottom boundary have weight 1
    diffs[:L] = 1
    diffs[-L:] = 1

    indptr = np.empty(L**2 + 1, dtype=np.int32)
    indptr[0] = 0
    indptr[1:] = np.cumsum(diffs)

    return indptr


def create_data(L):
    return np.ones(2 * L * (L - 1), dtype=np.int32)


def create_csc_H(L, basis):
    basis = basis.lower()

    data = create_data(L)

    if basis == "x":
        indices = create_hx_indices(L)
        indptr = create_hx_indptr(L)
    elif basis == "z":
        indices = create_hz_indices(L)
        indptr = create_hz_indptr(L)
    else:
        raise ValueError("basis must be 'x' or 'z'")

    n_rows = (L**2 - 1) // 2
    n_cols = L**2

    return csc_matrix((data, indices, indptr), shape=(n_rows, n_cols))