"""
pfaffian_orientation_step4.py

Step 4: Compute a Pfaffian orientation (Kasteleyn orientation) for the decorated planar graph G*_{∅}.

This implementation is designed for your pipeline:
- You already have a decorated graph with:
    XY_dec   : (n,2) float coordinates (planar embedding coordinates for vertices)
    edges_dec: list[(u,v,q)] where u,v are vertex ids (0..n-1), and q is an edge label:
        q >= 0  -> inherited edge from G* (primal edge index / qubit column)
        q < 0   -> gadget-internal edges (bulk gadgets, boundary gadget, semicircle, etc.)
- You have weights from Step 3 aligned with edges_dec:
    w[i] = 1 for gadget-internal, ±(1-2p) for inherited edges depending on eZ.

Goal:
- Produce a global edge orientation \vec{E} that is Pfaffian for the planar graph.
- Practically: compute a Kasteleyn orientation by enforcing "clockwise-odd" parity on ALL BOUNDED faces.
  (Outer face is excluded from constraints.)

Important: You have a special "semicircle" edge (typically tagged with q == -3) that must be treated
as an outer arc in the embedding (not a straight chord). This file handles that by:
- Using a virtual outside point when ordering that edge in the rotation system
- Using an arc-aware face area computation to robustly identify the outer face

Outputs:
- orient_dir: dict mapping a directed edge (u,v) to True, meaning the edge is oriented u -> v.
  For each undirected edge {a,b}, exactly one of (a,b) or (b,a) is present.
- edge_id: dict mapping undirected edges (min(u,v), max(u,v)) -> index
- faces: list of face cycles (each is a list of vertex ids)
- outer_face_index: index into faces that is the outer face

Then Step 5 can build A(eZ):
A_uv = +w_e if (u,v) in orient_dir else -w_e, and A_vu = -A_uv.

"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional
import numpy as np


Edge = Tuple[int, int, int]  # (u, v, q)
Undir = Tuple[int, int]  # (a, b) with a < b
Dir = Tuple[int, int]  # (u, v)


# -----------------------------
# Rotation system (embedding)
# -----------------------------


def _bbox_outside_point(
    XY: np.ndarray, pad: float = 50.0, corner: str = "NW"
) -> np.ndarray:
    """Pick a far-away point outside the bounding box for arc handling."""
    minx, miny = XY.min(axis=0)
    maxx, maxy = XY.max(axis=0)
    if corner == "NW":
        return np.array([minx - pad, maxy + pad], dtype=float)
    if corner == "SE":
        return np.array([maxx + pad, miny - pad], dtype=float)
    if corner == "NE":
        return np.array([maxx + pad, maxy + pad], dtype=float)
    if corner == "SW":
        return np.array([minx - pad, miny - pad], dtype=float)
    raise ValueError(corner)


def build_rotation_system(
    n: int,
    XY: np.ndarray,
    edges: List[Edge],
    semicircle_q: int = -3,
    outside_pad: float = 50.0,
) -> Dict[int, List[int]]:
    """
    Build cyclic neighbour order around each vertex by polar angle.

    Special handling for the semicircle edge (q == semicircle_q):
    Treat it as pointing to a far-away 'outside' point so it sits on the outer boundary
    in the rotation system, even if the straight chord would appear to cross in a plot.
    """
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, q in edges:
        adj[u].append((v, q))
        adj[v].append((u, q))

    OUT_NW = _bbox_outside_point(XY, pad=outside_pad, corner="NW")
    OUT_SE = _bbox_outside_point(XY, pad=outside_pad, corner="SE")

    cx = float((XY[:, 0].min() + XY[:, 0].max()) / 2.0)
    cy = float((XY[:, 1].min() + XY[:, 1].max()) / 2.0)

    rot: Dict[int, List[int]] = {}
    for u in range(n):
        ux, uy = XY[u]

        def ang(v: int, q: int) -> float:
            if q == semicircle_q:
                use_nw = (ux <= cx) and (uy >= cy)
                vx, vy = OUT_NW if use_nw else OUT_SE
            else:
                vx, vy = XY[v]
            return math.atan2(vy - uy, vx - ux)

        nbrs = adj[u]
        nbrs_sorted = sorted(nbrs, key=lambda t: ang(t[0], t[1]))
        rot[u] = [v for (v, _q) in nbrs_sorted]

    return rot


# -----------------------------
# Face extraction (half-edge walk)
# -----------------------------


def _build_pos_in_rot(rot: Dict[int, List[int]]) -> Dict[Dir, int]:
    pos: Dict[Dir, int] = {}
    for u, nbrs in rot.items():
        for i, v in enumerate(nbrs):
            pos[(u, v)] = i
    return pos


def _next_halfedge(
    rot: Dict[int, List[int]], pos: Dict[Dir, int], u: int, v: int
) -> Dir:
    idx = pos[(v, u)]
    w = rot[v][(idx - 1) % len(rot[v])]
    return (v, w)


def extract_faces(rot: Dict[int, List[int]]) -> List[List[int]]:
    half_edges: List[Dir] = []
    for u, nbrs in rot.items():
        for v in nbrs:
            half_edges.append((u, v))

    pos = _build_pos_in_rot(rot)
    used: set[Dir] = set()
    faces: List[List[int]] = []

    for he in half_edges:
        if he in used:
            continue
        cyc: List[int] = []
        curr = he
        while curr not in used:
            used.add(curr)
            u, v = curr
            cyc.append(u)
            curr = _next_halfedge(rot, pos, u, v)

        if len(cyc) >= 3:
            faces.append(cyc)

    def normalize(c: List[int]) -> Tuple[int, ...]:
        m = min(c)
        k = c.index(m)
        r1 = c[k:] + c[:k]
        cr = list(reversed(c))
        k2 = cr.index(m)
        r2 = cr[k2:] + cr[:k2]
        t1, t2 = tuple(r1), tuple(r2)
        return t1 if t1 < t2 else t2

    uniq: Dict[Tuple[int, ...], List[int]] = {}
    for f in faces:
        uniq[normalize(f)] = f
    return list(uniq.values())


# -----------------------------
# Arc-aware face area (outer-face detection)
# -----------------------------


def find_semicircle_endpoints(
    edges: List[Edge], semicircle_q: int = -3
) -> Optional[Undir]:
    arc = [(min(u, v), max(u, v)) for (u, v, q) in edges if q == semicircle_q]
    if len(arc) == 0:
        return None
    if len(arc) != 1:
        raise ValueError(f"Expected 0 or 1 semicircle edges, found {len(arc)}.")
    return arc[0]


def polygon_signed_area_with_arc(
    XY: np.ndarray,
    cycle: List[int],
    arc_uv: Optional[Undir],
    outside_point: np.ndarray,
) -> float:
    pts: List[np.ndarray] = []
    m = len(cycle)
    for i in range(m):
        u = cycle[i]
        v = cycle[(i + 1) % m]
        pts.append(XY[u])
        if arc_uv is not None:
            a, b = arc_uv
            if (u == a and v == b) or (u == b and v == a):
                pts.append(outside_point)

    pts = np.asarray(pts, dtype=float)
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


# -----------------------------
# GF(2) linear solve (Kasteleyn constraints)
# -----------------------------


def gf2_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = A.copy().astype(np.uint8)
    b = b.copy().astype(np.uint8)
    m, n = A.shape

    row = 0
    pivot_col_for_row = [-1] * m

    for col in range(n):
        piv = None
        for r in range(row, m):
            if A[r, col] & 1:
                piv = r
                break
        if piv is None:
            continue
        if piv != row:
            A[[row, piv]] = A[[piv, row]]
            b[[row, piv]] = b[[piv, row]]
        pivot_col_for_row[row] = col

        for r in range(m):
            if r != row and (A[r, col] & 1):
                A[r, :] ^= A[row, :]
                b[r] ^= b[row]

        row += 1
        if row == m:
            break

    for r in range(m):
        if A[r, :].sum() == 0 and (b[r] & 1):
            raise ValueError("Inconsistent GF(2) system for Kasteleyn constraints.")

    x = np.zeros(n, dtype=np.uint8)
    for r in range(m - 1, -1, -1):
        col = pivot_col_for_row[r]
        if col == -1:
            continue
        s = 0
        for j in range(col + 1, n):
            if A[r, j] & 1:
                s ^= x[j]
        x[col] = b[r] ^ s
    return x


# -----------------------------
# Step 4: Kasteleyn / Pfaffian orientation
# -----------------------------


def kasteleyn_pfaffian_orientation(
    XY: np.ndarray,
    edges: List[Edge],
    semicircle_q: int = -3,
    outside_pad: float = 50.0,
) -> Tuple[Dict[Dir, bool], Dict[Undir, int], List[List[int]], int]:
    """
    Compute a Kasteleyn orientation (clockwise-odd) on ALL BOUNDED faces only.

    Returns:
      orient_dir: dict[(u,v)] = True meaning the edge is oriented u -> v.
      edge_id   : dict[(a,b)] = eid for undirected edge {a,b}
      faces     : list of face cycles (vertex ids)
      outer_idx : index of the outer face in faces
    """
    XY = np.asarray(XY, dtype=float)
    n = XY.shape[0]

    rot = build_rotation_system(
        n, XY, edges, semicircle_q=semicircle_q, outside_pad=outside_pad
    )
    faces = extract_faces(rot)
    if not faces:
        raise ValueError(
            "No faces extracted; embedding/rotation system likely invalid."
        )

    arc_uv = find_semicircle_endpoints(edges, semicircle_q=semicircle_q)
    outside_point = _bbox_outside_point(XY, pad=outside_pad, corner="NW")
    areas = [
        abs(polygon_signed_area_with_arc(XY, f, arc_uv, outside_point)) for f in faces
    ]
    outer_idx = int(np.argmax(areas))

    edge_id: Dict[Undir, int] = {}
    undirs: List[Undir] = []
    for u, v, _q in edges:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in edge_id:
            edge_id[(a, b)] = len(undirs)
            undirs.append((a, b))
    E = len(undirs)

    bounded_faces = [f for i, f in enumerate(faces) if i != outer_idx]
    m = len(bounded_faces)

    A = np.zeros((m, E), dtype=np.uint8)
    b = np.zeros(m, dtype=np.uint8)

    for fi, cyc in enumerate(bounded_faces):
        if polygon_signed_area_with_arc(XY, cyc, arc_uv, outside_point) < 0:
            cyc = list(reversed(cyc))

        parity0 = 0
        Lc = len(cyc)
        for k in range(Lc):
            u = cyc[k]
            v = cyc[(k + 1) % Lc]
            a, bb = (u, v) if u < v else (v, u)
            eid = edge_id[(a, bb)]
            A[fi, eid] ^= 1

            base_matches = a == u and bb == v  # base orientation is min->max
            if not base_matches:
                parity0 ^= 1

        b[fi] = (1 ^ parity0) & 1  # clockwise-odd

    x = gf2_solve(A, b)

    orient_dir: Dict[Dir, bool] = {}
    for (a, bb), eid in edge_id.items():
        flip = bool(x[eid] & 1)
        if not flip:
            orient_dir[(a, bb)] = True
        else:
            orient_dir[(bb, a)] = True

    return orient_dir, edge_id, faces, outer_idx


# -----------------------------
# Utility for Step 5: Build A(eZ) from orientation + weights
# -----------------------------


def build_skew_symmetric_coo(
    n: int,
    edges: List[Edge],
    weights: np.ndarray,
    orient_dir: Dict[Dir, bool],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build COO triplets (I,J,V) for the skew-symmetric matrix A(eZ).

    For each undirected edge e=(u,v) with weight w_e:
      if oriented u->v: A[u,v]=+w_e, A[v,u]=-w_e
      else (v->u):      A[u,v]=-w_e, A[v,u]=+w_e

    weights must align with edges list ordering.
    """
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(edges):
        raise ValueError("weights must align with edges list ordering (same length).")

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


def _self_test_pfaffian_orientation(L: int = 5) -> None:
    """
    End-to-end smoke test for this module using the project's G* builder/decorator.
    Runs when the file is executed directly.
    """
    from parity_checks import create_H
    from graphs.construct_gstar import build_G_from_Hz, decorate_gstar

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

    orient_dir, edge_id, faces, outer_idx = kasteleyn_pfaffian_orientation(XY_dec, edges_dec)

    # 1) Exactly one directed orientation per undirected edge.
    assert len(orient_dir) == len(edge_id), (
        f"orientation count mismatch: {len(orient_dir)} vs {len(edge_id)}"
    )
    for (a, b) in edge_id:
        has_ab = (a, b) in orient_dir
        has_ba = (b, a) in orient_dir
        assert has_ab ^ has_ba, f"edge {(a, b)} is not oriented exactly once"

    # 2) Face extraction sanity.
    assert len(faces) > 0, "no faces extracted"
    assert 0 <= outer_idx < len(faces), f"invalid outer face index {outer_idx}"

    # 3) Bounded-face Kasteleyn parity sanity (clockwise-odd).
    arc_uv = find_semicircle_endpoints(edges_dec, semicircle_q=-3)
    outside_point = _bbox_outside_point(np.asarray(XY_dec, float), pad=50.0, corner="NW")
    for i, cyc in enumerate(faces):
        if i == outer_idx:
            continue
        ccw = list(cyc)
        if polygon_signed_area_with_arc(XY_dec, ccw, arc_uv, outside_point) < 0:
            ccw = list(reversed(ccw))
        # Count edges oriented in the clockwise direction (reverse of CCW traversal).
        cw_count = 0
        m = len(ccw)
        for k in range(m):
            u = ccw[k]
            v = ccw[(k + 1) % m]
            if (v, u) in orient_dir:
                cw_count += 1
        assert (cw_count % 2) == 1, (
            f"bounded face {i} violates clockwise-odd parity (count={cw_count})"
        )

    # 4) Skew-symmetric matrix construction sanity.
    weights = np.ones(len(edges_dec), dtype=float)
    I, J, V = build_skew_symmetric_coo(len(XY_dec), edges_dec, weights, orient_dir)
    A = np.zeros((len(XY_dec), len(XY_dec)), dtype=float)
    for ii, jj, vv in zip(I, J, V):
        A[ii, jj] += vv
    assert np.allclose(A + A.T, 0.0), "constructed matrix is not skew-symmetric"

    print("Pfaffian orientation self-test passed")
    print(f"L={L} | base G*: V={G.num_vertices()} E={G.num_edges()}")
    print(
        f"decorated: V={len(XY_dec)} E={len(edges_dec)} | faces={len(faces)} "
        f"(bounded={len(faces)-1}) | oriented_edges={len(orient_dir)}"
    )


if __name__ == "__main__":
    _self_test_pfaffian_orientation()
