import numpy as np
from typing import List, Tuple

Edge = Tuple[int, int, int]  # (u, v, q)

def make_eZ_mask(num_primal_edges: int, eZ_edges: np.ndarray | List[int]) -> np.ndarray:
    """
    eZ_edges: list/array of primal edge indices (qubit cols) in eZ.
    Returns boolean mask length num_primal_edges.
    """
    mask = np.zeros(num_primal_edges, dtype=bool)
    eZ_edges = np.asarray(eZ_edges, dtype=int)
    if eZ_edges.size:
        if eZ_edges.min() < 0 or eZ_edges.max() >= num_primal_edges:
            raise ValueError("eZ edge index out of range.")
        mask[eZ_edges] = True
    return mask

def edge_weights_eZ(edges_dec: List[Edge], eZ_mask: np.ndarray, p: float) -> np.ndarray:
    """
    Step 3 weights:
      w_e = 1                 if q < 0  (gadget-internal)
      w_e = -(1-2p)           if q>=0 and q in eZ
      w_e = +(1-2p)           if q>=0 and q not in eZ
    """
    if not (0.0 <= p <= 0.5):
        raise ValueError("Require 0 <= p <= 0.5.")
    base = 1.0 - 2.0 * float(p)

    w = np.empty(len(edges_dec), dtype=np.float64)
    for i, (u, v, q) in enumerate(edges_dec):
        if q < 0:
            w[i] = 1.0
        else:
            if q >= len(eZ_mask):
                raise ValueError(f"Edge label q={q} exceeds eZ_mask length {len(eZ_mask)}.")
            w[i] = -base if bool(eZ_mask[q]) else base
    return w