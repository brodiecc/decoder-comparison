from __future__ import annotations

from collections import deque
from typing import Dict, Tuple, Any

import numpy as np

from graphs.construct_gstar import build_G_from_Hz, decorate_gstar
from parity_checks import create_H
from pfaffian_orientation import kasteleyn_pfaffian_orientation
from skew_det import build_A_and_det
from weights import edge_weights_eZ


_PIPELINE_CACHE: Dict[Tuple[Any, ...], dict] = {}


def _infer_L_from_num_qubits(num_qubits: int) -> int:
    L = int(round(np.sqrt(num_qubits)))
    if L * L != num_qubits:
        raise ValueError(f"Expected number of qubits to be a perfect square, got {num_qubits}.")
    return L


def _sparse_fingerprint(H) -> Tuple[Any, ...]:
    """Deterministic cache key for a graphlike sparse parity-check matrix."""
    Hc = H.tocsc()
    return (
        Hc.shape,
        Hc.indptr.tobytes(),
        Hc.indices.tobytes(),
        np.asarray(Hc.data % 2, dtype=np.uint8).tobytes(),
    )


def _compact_boundary_dead(XY_dec, edges_dec, vtype):
    keep = [i for i, t in enumerate(vtype) if t != "boundary_dead"]
    remap = {old: new for new, old in enumerate(keep)}

    XY2 = np.asarray(XY_dec, dtype=float)[keep]
    edges2 = []
    for u, v, q in edges_dec:
        if u in remap and v in remap:
            a, b = remap[u], remap[v]
            if a != b:
                uu, vv = (a, b) if a < b else (b, a)
                edges2.append((uu, vv, int(q)))
    return XY2, sorted(set(edges2))


def _get_pfaffian_static_context(Hx) -> dict:
    """
    Build/cache decorated graph + Pfaffian orientation for a given graphlike check matrix.

    Important convention:
      - If the input matrix is Hx (used to decode Z errors), build the Pfaffian graph from Hz.
      - If the input matrix is Hz (used to decode X errors), build the Pfaffian graph from Hx.
    """
    key = _sparse_fingerprint(Hx)
    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        return cached

    L = _infer_L_from_num_qubits(Hx.shape[1])

    # Infer whether the supplied matrix matches create_H(L,"x") or create_H(L,"z"),
    # then build the Pfaffian graph from the opposite basis matrix.
    H_in = Hx.tocsc()
    Hx_ref = create_H(L, "x").tocsc()
    Hz_ref = create_H(L, "z").tocsc()
    fp_in = _sparse_fingerprint(H_in)
    fp_x = _sparse_fingerprint(Hx_ref)
    fp_z = _sparse_fingerprint(Hz_ref)

    if fp_in == fp_x:
        H_pf = Hz_ref
        pf_basis = "z"
    elif fp_in == fp_z:
        H_pf = Hx_ref
        pf_basis = "x"
    else:
        # Fallback for custom/reordered matrices: preserve prior behavior.
        H_pf = H_in
        pf_basis = "unknown(same-as-input)"

    G = build_G_from_Hz(H_pf, L)
    XY_dec, edges_dec, vtype = decorate_gstar(G)
    XY_dec, edges_dec = _compact_boundary_dead(XY_dec, edges_dec, vtype)

    orient_dir, edge_id, faces, outer_idx = kasteleyn_pfaffian_orientation(XY_dec, edges_dec)

    ctx = {
        "L": L,
        "pf_basis": pf_basis,
        "XY_dec": XY_dec,
        "edges_dec": edges_dec,
        "orient_dir": orient_dir,
        "edge_id": edge_id,
        "faces": faces,
        "outer_idx": outer_idx,
        "num_qubits": int(Hx.shape[1]),
    }
    _PIPELINE_CACHE[key] = ctx
    return ctx


def _logical_to_dense_vector(logical_z, num_qubits: int) -> np.ndarray:
    """
    Accepts a dense vector or a 1xN sparse/dense matrix and returns uint8 length-N.
    """
    if hasattr(logical_z, "toarray"):
        arr = np.asarray(logical_z.toarray(), dtype=np.uint8)
        if arr.ndim == 2:
            if arr.shape[0] != 1:
                raise ValueError(
                    f"logical_z must be a vector or shape (1, N), got {arr.shape}."
                )
            arr = arr[0]
    else:
        arr = np.asarray(logical_z, dtype=np.uint8)
        if arr.ndim == 2:
            if 1 not in arr.shape:
                raise ValueError(
                    f"logical_z must be a vector or shape (1, N), got {arr.shape}."
                )
            arr = arr.reshape(-1)

    arr = (arr.astype(np.uint8) & 1).reshape(-1)
    if arr.size != num_qubits:
        raise ValueError(f"logical_z length {arr.size} does not match num_qubits {num_qubits}.")
    return arr


def calculate_gadget_det(Hx, eZ: np.ndarray, p: float) -> float:
    """
    Compute det(A(eZ)) using the cached decorated graph + Pfaffian orientation pipeline.
    """
    ctx = _get_pfaffian_static_context(Hx)
    eZ = (np.asarray(eZ, dtype=np.uint8).reshape(-1) & 1)
    if eZ.size != ctx["num_qubits"]:
        raise ValueError(f"eZ length {eZ.size} does not match {ctx['num_qubits']}.")

    weights = edge_weights_eZ(ctx["edges_dec"], eZ.astype(bool), p)
    _A, detA = build_A_and_det(
        ctx["XY_dec"], ctx["edges_dec"], weights, ctx["orient_dir"], method="dense"
    )
    return float(detA)


def find_initial_solution(Hx, syndrome):
    """
    Peeling on a graphlike check matrix (each column has 1 or 2 ones).
    Returns eZ such that (Hx @ eZ) % 2 == syndrome, when solvable.

    Important:
      - boundary node has no syndrome constraint
      - any component containing boundary must be rooted at boundary
      - components without boundary must have even syndrome parity
    """
    num_checks, num_qubits = Hx.shape
    boundary_node = num_checks

    # Build adjacency on detectors + boundary
    adj = [[] for _ in range(num_checks + 1)]
    Hx_csc = Hx.tocsc()

    for q in range(num_qubits):
        checks = Hx_csc.indices[Hx_csc.indptr[q] : Hx_csc.indptr[q + 1]]
        if len(checks) == 2:
            u, v = int(checks[0]), int(checks[1])
        elif len(checks) == 1:
            u, v = int(checks[0]), boundary_node
        else:
            # Not graphlike; peeling doesn't apply
            continue

        adj[u].append((v, q))
        adj[v].append((u, q))

    visited = np.zeros(num_checks + 1, dtype=bool)
    eZ = np.zeros(num_qubits, dtype=np.uint8)

    syn = np.zeros(num_checks + 1, dtype=np.uint8)
    syn[:num_checks] = np.asarray(syndrome, dtype=np.uint8).reshape(-1)[:num_checks]

    def peel_component(root):
        queue = deque([root])
        visited[root] = True

        order = []
        parent_edge = {}
        parent_node = {}

        while queue:
            u = queue.popleft()
            order.append(u)
            for v, edge_idx in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent_edge[v] = edge_idx
                    parent_node[v] = u
                    queue.append(v)

        # peel upward
        for u in reversed(order):
            if syn[u] == 1 and u in parent_node:
                e_idx = parent_edge[u]
                eZ[e_idx] ^= 1
                pnode = parent_node[u]
                syn[pnode] ^= 1
                syn[u] = 0

        return order  # nodes in this component

    # 1) Always peel the boundary component first (if boundary has any edges)
    if not visited[boundary_node] and len(adj[boundary_node]) > 0:
        _ = peel_component(boundary_node)
        # leftover syn at boundary is allowed (don't care)
        syn[boundary_node] = 0

    # 2) Peel remaining components; they must have even parity (no boundary sink)
    for root in range(num_checks):
        if visited[root]:
            continue
        comp_nodes = peel_component(root)
        parity = int(np.bitwise_xor.reduce(syn[comp_nodes])) if comp_nodes else 0
        if parity != 0:
            raise ValueError(
                "Unsatisfiable syndrome in a component not connected to boundary "
                "(odd parity with no boundary sink)."
            )
        for u in comp_nodes:
            syn[u] = 0

    return eZ


def decode(Hx, logical_z, syndrome, p):
    """
    Decode by comparing determinant weights for:
      eZ       = peeling solution
      eZ_prime = eZ XOR logical_z
    and returning the more likely class under the Pfaffian determinant criterion.
    """
    eZ = find_initial_solution(Hx, syndrome)
    logical_vec = _logical_to_dense_vector(logical_z, eZ.size)
    eZ_prime = np.bitwise_xor(eZ.astype(np.uint8), logical_vec).astype(np.uint8)

    det_1 = calculate_gadget_det(Hx, eZ, p)
    det_2 = calculate_gadget_det(Hx, eZ_prime, p)

    return eZ if det_1 > det_2 else eZ_prime
