"""
pfaffian_matrix_step5_6.py

Implements Algorithm 5.2 Steps 5 and 6:

5. Build the skew-symmetric matrix A(eZ) from:
   - the decorated graph G*_{∅} edges (u,v,q)
   - the Pfaffian (Kasteleyn) orientation \vec{E} (as orient_dir dict)
   - the weights w^(eZ) for each decorated edge

   Using Eq. (1) from your excerpt:
     A_uv = +w_e  if (u,v) is oriented u -> v
     A_uv = -w_e  if (v,u) is oriented v -> u
     A_uv = 0     otherwise
   and A is skew-symmetric (A_vu = -A_uv).

6. Compute det(A(eZ)).

Notes on numerics:
- For L=5 your decorated n is ~74; dense det is feasible and simplest/robust.
- For larger n, computing det directly can under/overflow. Prefer logdet from LU factorisation.
- A is real skew-symmetric, so det(A) should be >= 0 in exact arithmetic (up to roundoff).

This module provides both:
- build_A_csc: sparse CSC matrix builder
- det_dense / logdet_sparse_lu: determinant utilities
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu


Edge = Tuple[int, int, int]  # (u, v, q)
Dir = Tuple[int, int]  # (u, v)


# -------------------------
# Step 5: Build A(eZ)
# -------------------------


def build_A_coo(
    n: int,
    edges: List[Edge],
    weights: np.ndarray,
    orient_dir: Dict[Dir, bool],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return COO triplets (I,J,V) for A(eZ) (including both (u,v) and (v,u) entries).
    weights must align with edges list ordering.

    For each edge i: (u,v, q) with weight w_i:
      if orient u->v:
          A[u,v] = +w_i, A[v,u] = -w_i
      else if orient v->u:
          A[u,v] = -w_i, A[v,u] = +w_i
    """
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(edges):
        raise ValueError("weights must align with edges (same length).")

    I: List[int] = []
    J: List[int] = []
    V: List[float] = []

    for i, (u, v, _q) in enumerate(edges):
        w = float(weights[i])

        if (u, v) in orient_dir:
            I.extend([u, v])
            J.extend([v, u])
            V.extend([+w, -w])
        elif (v, u) in orient_dir:
            I.extend([u, v])
            J.extend([v, u])
            V.extend([-w, +w])
        else:
            raise ValueError(f"Missing orientation for edge ({u},{v}).")

    return (
        np.asarray(I, dtype=int),
        np.asarray(J, dtype=int),
        np.asarray(V, dtype=float),
    )


def build_A_csc(
    n: int,
    edges: List[Edge],
    weights: np.ndarray,
    orient_dir: Dict[Dir, bool],
) -> csc_matrix:
    """
    Build A(eZ) as a sparse CSC matrix.
    """
    I, J, V = build_A_coo(n, edges, weights, orient_dir)
    A = coo_matrix((V, (I, J)), shape=(n, n)).tocsc()

    # Optional sanity checks (cheap):
    # - diagonal should be zero (numerical exact here)
    diag = A.diagonal()
    if np.any(diag != 0):
        # In exact construction should be all zeros
        raise RuntimeError(
            "A has nonzero diagonal entries; expected skew-symmetric with zero diagonal."
        )

    return A


def check_skew_symmetric(A: csc_matrix, atol: float = 1e-12) -> None:
    """
    Debug helper: verify A^T = -A (within tolerance).
    """
    D = (A + A.T).tocoo()
    if D.nnz == 0:
        return
    mx = np.max(np.abs(D.data))
    if mx > atol:
        raise RuntimeError(
            f"A is not skew-symmetric within atol={atol}. max|A+A^T|={mx}."
        )


# -------------------------
# Step 6: Determinant
# -------------------------


def det_dense(A: csc_matrix) -> float:
    """
    Dense determinant (robust for n ~ O(10^2)).
    Returns a float (may be large).
    """
    M = A.toarray()
    return float(np.linalg.det(M))


def logdet_sparse_lu(A: csc_matrix) -> Tuple[float, int]:
    """
    Compute log(|det(A)|) and sign(det(A)) using sparse LU.
    Useful when det is huge/small.

    Returns:
      logabsdet, sign
    """
    # splu requires CSC
    if not isinstance(A, csc_matrix):
        A = A.tocsc()

    # LU factorization with COLAMD ordering (default)
    lu = splu(A)

    # det(A) = det(P_r) * det(L) * det(U) * det(P_c)
    # det(L)=1 because L has unit diagonal in SuperLU.
    # det(P_r) and det(P_c) are ±1 depending on permutation parity.
    # det(U)= product of diagonal of U.

    diagU = lu.U.diagonal()
    # If any zero on diag, det = 0
    if np.any(diagU == 0):
        return float("-inf"), 0

    logabsdet = float(np.sum(np.log(np.abs(diagU))))
    signU = int(np.prod(np.sign(diagU)))

    # Permutation parity: compute sign of permutation vector p (maps i -> p[i])
    def perm_sign(p: np.ndarray) -> int:
        p = np.asarray(p, dtype=int)
        n = p.size
        visited = np.zeros(n, dtype=bool)
        cycles = 0
        for i in range(n):
            if not visited[i]:
                cycles += 1
                j = i
                while not visited[j]:
                    visited[j] = True
                    j = p[j]
        # sign = (-1)^(n - cycles)
        return -1 if ((n - cycles) % 2) else 1

    sPr = perm_sign(lu.perm_r)
    sPc = perm_sign(lu.perm_c)

    sign = int(sPr * signU * sPc)

    return logabsdet, sign


def det_from_logdet(logabsdet: float, sign: int) -> float:
    """
    Convert (logabsdet, sign) to a float determinant when possible.
    WARNING: may overflow/underflow for large |det|.
    """
    if sign == 0:
        return 0.0
    return float(sign) * float(np.exp(logabsdet))


# -------------------------
# End-to-end convenience
# -------------------------


def build_A_and_det(
    XY_dec: np.ndarray,
    edges_dec: List[Edge],
    weights: np.ndarray,
    orient_dir: Dict[Dir, bool],
    method: str = "dense",
) -> Tuple[csc_matrix, float]:
    """
    Build A(eZ) and compute det(A).
    method:
      - "dense": convert to dense and call numpy.linalg.det
      - "lu": compute determinant via sparse LU -> exp(logdet) (may overflow); prefer logdet output instead
    """
    n = int(XY_dec.shape[0])
    A = build_A_csc(n, edges_dec, weights, orient_dir)
    check_skew_symmetric(A)

    if method == "dense":
        detA = det_dense(A)
        return A, detA

    if method == "lu":
        logabsdet, sign = logdet_sparse_lu(A)
        detA = det_from_logdet(logabsdet, sign)
        return A, detA

    raise ValueError("method must be 'dense' or 'lu'")


def _self_test_skew_det(L: int = 5) -> None:
    """
    End-to-end smoke test for skew_det using the project's current G* pipeline.
    Runs when executing this file directly.
    """
    from parity_checks import create_H
    from graphs.construct_gstar import build_G_from_Hz, decorate_gstar
    from pfaffian_orientation import kasteleyn_pfaffian_orientation

    Hz = create_H(L, "z")
    G = build_G_from_Hz(Hz, L)
    XY_dec, edges_dec, vtype = decorate_gstar(G)

    def drop_hidden_boundary_dead(XY, edges, vtype_list):
        keep = [i for i, t in enumerate(vtype_list) if t != "boundary_dead"]
        remap = {old: new for new, old in enumerate(keep)}
        XY2 = np.asarray(XY, float)[keep]
        edges2 = []
        for u, v, q in edges:
            if u in remap and v in remap:
                a, b = remap[u], remap[v]
                if a != b:
                    uu, vv = (a, b) if a < b else (b, a)
                    edges2.append((uu, vv, q))
        edges2 = sorted(set(edges2))
        return XY2, edges2

    XY_dec, edges_dec = drop_hidden_boundary_dead(XY_dec, edges_dec, vtype)

    orient_dir, _edge_id, _faces, _outer_idx = kasteleyn_pfaffian_orientation(
        XY_dec, edges_dec
    )

    rng = np.random.default_rng(12345)
    # Generic nonzero weights reduce the chance of a singular test instance.
    weights = rng.uniform(0.5, 1.5, size=len(edges_dec)).astype(float)

    # Build and verify skew-symmetry.
    A = build_A_csc(len(XY_dec), edges_dec, weights, orient_dir)
    check_skew_symmetric(A)

    # Determinant consistency across methods (within numerical tolerance).
    det_d = det_dense(A)
    lu_ok = True
    try:
        logabsdet, sign = logdet_sparse_lu(A)
        det_lu = det_from_logdet(logabsdet, sign)
    except RuntimeError as e:
        lu_ok = False
        logabsdet, sign, det_lu = float("-inf"), 0, 0.0
        # Singular factorization is acceptable if dense determinant is numerically ~0.
        assert "singular" in str(e).lower(), f"unexpected LU failure: {e}"
        assert abs(det_d) <= 1e-6, f"LU singular but dense determinant not near zero: {det_d}"

    # det(A) for real skew-symmetric matrices should be >= 0 in exact arithmetic.
    assert det_d >= -1e-6, f"dense det unexpectedly negative: {det_d}"
    if sign != 0:
        assert sign >= 0, f"LU sign unexpectedly negative for skew-symmetric A: {sign}"

    # Use relative comparison when finite and nonzero; otherwise absolute.
    if lu_ok and np.isfinite(det_d) and np.isfinite(det_lu):
        scale = max(1.0, abs(det_d), abs(det_lu))
        assert abs(det_d - det_lu) <= 1e-6 * scale, (
            f"dense/LU determinant mismatch: dense={det_d}, lu={det_lu}"
        )

    # Cross-check convenience wrapper.
    A2, det2 = build_A_and_det(XY_dec, edges_dec, weights, orient_dir, method="dense")
    assert A2.shape == A.shape
    assert abs(det2 - det_d) <= 1e-8 * max(1.0, abs(det_d))

    print("skew_det self-test passed")
    print(f"L={L} | decorated: V={len(XY_dec)} E={len(edges_dec)}")
    print(
        f"det(A) dense={det_d:.6g} | logabsdet={logabsdet:.6g} sign={sign} | "
        f"lu_ok={lu_ok} | "
        f"nnz={A.nnz}"
    )


if __name__ == "__main__":
    _self_test_skew_det()
