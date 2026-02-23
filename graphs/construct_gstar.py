"""
gstar_builder.py

Builds a primal-graph-like G* from an input Hz (CSC), including extra boundary vertices
that are not checks (not present as rows in Hz), and assigns "expanded" integer coordinates
to all vertices for later decoration (Fisher gadgets, etc.).

Assumptions (matches the user's description for rotated planar / diamond layout, odd L):
- Hz has shape (n_checks, L^2) with n_checks = (L^2 - 1)//2.
- Each qubit column has weight 2 (internal) or 1 (boundary).
- Check-vertices (rows of Hz) are embedded on a diamond with row sizes:
    1,3,5,...,L,L,...,5,3,1  (length L+1)
  giving total vertices = 18 when L=5.
- Additional boundary vertices exist:
    * rightmost vertex of each diamond row (including the bottom row of size 1)
    * plus leftmost vertex of each row in the *decreasing* half (rows after the plateau)
  These boundary vertices are included in G even though they don't appear in Hz.

What you get:
- A graph object with:
    * vertex positions (logical and expanded coords)
    * edges labelled by qubit column index q
    * ability to export adjacency / edge lists for decoration and Pfaffian steps

This file is intentionally "from scratch" and does not depend on your earlier builders.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
from scipy.sparse import csc_matrix


# -------------------------
# Diamond geometry helpers
# -------------------------


def diamond_row_sizes(L: int) -> List[int]:
    """
    For odd L (>=3):
      rowsizes = [1,3,...,L, L, ...,3,1] with length L+1.
    Example L=5: [1,3,5,5,3,1]
    """
    if L % 2 == 0 or L < 3:
        raise ValueError("L must be odd and >= 3")
    d = (L - 1) // 2
    up = [2 * i + 1 for i in range(d + 1)]  # 1..L
    down = [2 * i + 1 for i in range(d - 1, -1, -1)]  # L-2..1
    return up + [L] + down  # duplicate max row -> L+1 rows


def diamond_positions(L: int) -> List[Tuple[int, int]]:
    """
    Returns a list of (x,y) integer logical positions for the full diamond vertex set,
    enumerated bottom->top, left->right within each row.
    """
    rows = diamond_row_sizes(L)
    out: List[Tuple[int, int]] = []
    for y, m in enumerate(rows):
        for x in range(-(m // 2), (m // 2) + 1):
            out.append((x, y))
    return out


def boundary_only_positions(L: int) -> set[tuple[int, int]]:
    """
    One boundary-only vertex per row:
      - rightmost on rows up to the *first* max-width row (inclusive)
      - leftmost starting from the next row (second max row and beyond)

    For L=5 rowsizes [1,3,5,5,3,1]:
      y=0,1,2 -> rightmost
      y=3,4,5 -> leftmost
    """
    rows = diamond_row_sizes(L)

    plateau_ys = [y for y, m in enumerate(rows) if m == L]
    if not plateau_ys:
        raise ValueError("No plateau row found; check diamond_row_sizes.")
    y_first_max = min(plateau_ys)  # <-- key change (was max)

    b: set[tuple[int, int]] = set()
    for y, m in enumerate(rows):
        if y > y_first_max:
            b.add((-(m // 2), y))  # leftmost
        else:
            b.add((+(m // 2), y))  # rightmost
    return b


def expanded_row_offsets(
    rowsizes: List[int], boundary_height: float = 2.4, interior_height: float = 2.4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Variable vertical spacing:
      - bottom row and top row take boundary_height rows
      - interior rows take interior_height rows

    Returns:
      Yoff[y] = integer Y offset for row y
      heights[y] = row band height
    """
    n = len(rowsizes)
    heights = np.full(n, interior_height, dtype=float)
    heights[0] = boundary_height
    heights[-1] = boundary_height
    Yoff = np.zeros(n, dtype=float)
    for y in range(1, n):
        Yoff[y] = Yoff[y - 1] + heights[y - 1]
    return Yoff, heights


# -------------------------
# Hz -> edges extraction
# -------------------------


def build_edges_from_hz_zigzag(
    L: int, pos_to_vid: dict[tuple[int, int], int], rowsizes: list[int]
) -> list[tuple[int, int, int]]:
    """
    Construct edges using the user's Hz-column zigzag rule.
    There are L subcolumns, each of length L, so total L^2 edges/qubits.

    subcol c start:
      y0 = (c + 1)//2
      x0 = leftmost x in row y0  (for y0=0, this is 0)
    move pattern:
      if c even: U, R, U, R, ...
      if c odd : R, U, R, U, ...
    """
    edges: list[tuple[int, int, int]] = []
    q = 0
    for c in range(L):
        y = (c + 1) // 2
        m = rowsizes[y]
        x = -(m // 2)  # leftmost in that row

        for t in range(L):
            # direction
            if c % 2 == 0:
                move_up = t % 2 == 0  # U,R,U,R,...
            else:
                move_up = t % 2 == 1  # R,U,R,U,...

            x2, y2 = (x, y + 1) if move_up else (x + 1, y)

            if (x, y) not in pos_to_vid:
                raise ValueError(
                    f"Zigzag step lands on missing vertex {(x, y)} at (c={c}, t={t})."
                )
            if (x2, y2) not in pos_to_vid:
                raise ValueError(
                    f"Zigzag step lands on missing vertex {(x2, y2)} at (c={c}, t={t})."
                )

            u = pos_to_vid[(x, y)]
            v = pos_to_vid[(x2, y2)]
            uu, vv = (u, v) if u < v else (v, u)
            edges.append((uu, vv, q))

            x, y = x2, y2
            q += 1

    if q != L * L:
        raise RuntimeError("Internal error: did not generate L^2 edges.")
    return edges


def extract_edges_from_Hz(
    Hz: csc_matrix,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]]]:
    """
    From CSC Hz, extract:
      internal_edges: list of (check_a, check_b, qubit_col)
      boundary_cols : list of (check_a, qubit_col) where column weight is 1

    Raises if a column has weight not in {1,2}.
    """
    Hz = Hz.tocsc()
    indptr = Hz.indptr
    indices = Hz.indices

    internal_edges: List[Tuple[int, int, int]] = []
    boundary_cols: List[Tuple[int, int]] = []

    ncols = Hz.shape[1]
    for q in range(ncols):
        rows = indices[indptr[q] : indptr[q + 1]]
        if len(rows) == 2:
            a, b = int(rows[0]), int(rows[1])
            if a == b:
                raise ValueError(f"Hz column {q} repeats the same check index {a}.")
            if a < b:
                internal_edges.append((a, b, q))
            else:
                internal_edges.append((b, a, q))
        elif len(rows) == 1:
            boundary_cols.append((int(rows[0]), q))
        else:
            raise ValueError(f"Hz column {q} has weight {len(rows)}; expected 1 or 2.")
    return internal_edges, boundary_cols


# -------------------------
# Graph data model
# -------------------------


@dataclass
class GStar:
    """
    A lightweight graph container for G*.

    Vertex ids are 0..(n_vertices-1).
    Some vertex ids correspond to checks (appear as rows in Hz); others are boundary-only vertices.
    """

    L: int
    # vertex logical coords (x,y) in the diamond embedding
    logical_xy: np.ndarray  # shape (nV,2) int32
    # expanded coords (X,Y) integer for decoration space
    expanded_XY: np.ndarray  # shape (nV,2) int32
    # map Hz check row index -> vertex id in this graph
    check_to_vid: Dict[int, int]
    # boundary-only vertex ids
    boundary_vids: List[int]
    # edges: list of (u,v, qubit_col)
    edges: List[Tuple[int, int, int]]

    def num_vertices(self) -> int:
        return int(self.logical_xy.shape[0])

    def num_edges(self) -> int:
        return len(self.edges)

    def edge_array(self) -> np.ndarray:
        """Return edges as (m,3) int32 array [u,v,q]."""
        return np.asarray(self.edges, dtype=np.int32)

    def adjacency_lists(self) -> List[List[Tuple[int, int]]]:
        """
        adjacency[u] = list of (v, q) for each incident edge labelled by qubit column q.
        """
        adj: List[List[Tuple[int, int]]] = [[] for _ in range(self.num_vertices())]
        for u, v, q in self.edges:
            adj[u].append((v, q))
            adj[v].append((u, q))
        return adj


# -------------------------
# Main builder
# -------------------------


def build_G_from_Hz(
    Hz: csc_matrix,
    L: int,
    x_spacing: float = 2.4,
    boundary_height: float = 2.4,
    interior_height: float = 2.4,
) -> GStar:
    """
    Build G* from Hz and L, including boundary vertices described by the user,
    and compute expanded coordinates.

    Mapping strategy:
    - Construct full diamond vertex set positions (x,y) with total size = 1+3+...+L+L+...+3+1.
    - Identify boundary-only positions via boundary_only_positions(L).
    - Assign the remaining (non-boundary-only) positions to Hz check indices 0..n_checks-1
      in a stable order: bottom->top, left->right within row (excluding boundary-only positions).
      This matches typical CSC check ordering patterns for these constructions and keeps things deterministic.
    - Add edges:
        * internal qubit columns (weight 2): connect the two check-vertices (via check_to_vid)
        * boundary qubit columns (weight 1): connect the check-vertex to the boundary-only
          vertex in the same row on the correct side:
              - if row is in the increasing/plateau half: connect to rightmost boundary in that row
              - if row is in decreasing half: connect to leftmost boundary in that row
          (This aligns with your statement about where boundary vertices live.)
    """
    if L % 2 == 0 or L < 3:
        raise ValueError("L must be odd and >= 3")

    if not isinstance(Hz, csc_matrix):
        Hz = Hz.tocsc()

    n_checks_expected = (L * L - 1) // 2
    if Hz.shape[0] != n_checks_expected or Hz.shape[1] != L * L:
        raise ValueError(
            f"Hz shape {Hz.shape} does not match expected ({n_checks_expected}, {L * L}) for L={L}."
        )

    rowsizes = diamond_row_sizes(L)
    all_pos = diamond_positions(L)
    boundary_pos = boundary_only_positions(L)

    # Vertex id assignment for all diamond positions (including boundary-only)
    pos_to_vid: Dict[Tuple[int, int], int] = {}
    logical_xy = np.zeros((len(all_pos), 2), dtype=np.int32)
    for vid, (x, y) in enumerate(all_pos):
        pos_to_vid[(x, y)] = vid
        logical_xy[vid, 0] = x
        logical_xy[vid, 1] = y

    # Identify boundary vids
    boundary_vids = [
        pos_to_vid[p] for p in sorted(boundary_pos, key=lambda t: (t[1], t[0]))
    ]

    # Build list of available non-boundary positions in stable order for checks
    check_positions = [p for p in all_pos if p not in boundary_pos]
    if len(check_positions) != n_checks_expected:
        raise ValueError(
            f"Non-boundary positions ({len(check_positions)}) != number of checks ({n_checks_expected}). "
            f"Adjust boundary rule if needed."
        )

    # Map checks -> vids
    check_to_vid: Dict[int, int] = {}
    for chk, p in enumerate(check_positions):
        check_to_vid[chk] = pos_to_vid[p]

    # Expanded coordinates (for decoration)
    Yoff, heights = expanded_row_offsets(
        rowsizes, boundary_height=boundary_height, interior_height=interior_height
    )
    expanded_XY = np.zeros_like(logical_xy, dtype=float)
    expanded_XY[:, 0] = logical_xy[:, 0].astype(float) * float(x_spacing)
    expanded_XY[:, 1] = Yoff[logical_xy[:, 1]]

    # Build edges using the zigzag rule (up/right only, no diagonals)
    edges = build_edges_from_hz_zigzag(L, pos_to_vid, rowsizes)

    # Sort edges by qubit column for reproducibility
    edges.sort(key=lambda e: e[2])

    return GStar(
        L=L,
        logical_xy=logical_xy,
        expanded_XY=expanded_XY,
        check_to_vid=check_to_vid,
        boundary_vids=boundary_vids,
        edges=edges,
    )


import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class DecoratedGraph:
    # coords for plotting / later embedding
    XY: np.ndarray  # (nV',2) int
    # edges: (u,v,q) where q=-1 for gadget-internal edges, else inherited qubit label
    edges: List[Tuple[int, int, int]]
    # map original vertex -> list of new gadget vertices (empty if boundary)
    gadget_of: Dict[int, List[int]]
    # map (orig_vertex, orig_neighbor) -> new endpoint vertex id (port)
    port_map: Dict[Tuple[int, int], int]
    # original boundary vertex ids mapped to same vertex id in decorated graph
    boundary_map: Dict[int, int]


def _adj_edges(nV, edges):
    adj = [[] for _ in range(nV)]
    for u, v, q in edges:
        adj[u].append((v, q))
        adj[v].append((u, q))
    return adj


def split_degree2_boundary_nodes(XY, edges, vtype, split_eps=0.35):
    """
    If a boundary node has exactly 2 incident *inherited* edges (q>=0),
    split it into two boundary nodes, each taking one inherited edge.
    Internal (q<0) edges are not expected at boundary right now.
    """
    XY = XY.copy()
    edges = list(edges)
    vtype = list(vtype)

    nV = XY.shape[0]
    adj = _adj_edges(nV, edges)

    # map from old boundary node -> [new_node_ids]
    split_map = {}

    for b in range(nV):
        if vtype[b] != "boundary":
            continue
        inh = [(nbr, q) for nbr, q in adj[b] if q >= 0]
        if len(inh) != 2:
            continue

        (n1, q1), (n2, q2) = inh
        xb, yb = XY[b]

        # place two new boundary nodes slightly separated (horizontal split)
        b1 = len(vtype)
        XY = np.vstack([XY, [xb - split_eps, yb]])
        vtype.append("boundary")
        b2 = len(vtype)
        XY = np.vstack([XY, [xb + split_eps, yb]])
        vtype.append("boundary")

        split_map[b] = (b1, b2)

        # rebuild edges: remove the two inherited edges (b-n1,q1) and (b-n2,q2)
        new_edges = []
        for u, v, q in edges:
            if q < 0:
                new_edges.append((u, v, q))
                continue
            a, c = (u, v) if u < v else (v, u)
            # remove those two
            if (a, c, q) == tuple(sorted((b, n1)) + [q1]) if False else None:
                pass

        # simpler: filter explicitly
        def is_target(u, v, q, bb, nn, qq):
            return q == qq and ((u == bb and v == nn) or (u == nn and v == bb))

        edges = [
            e
            for e in edges
            if not is_target(*e, b, n1, q1) and not is_target(*e, b, n2, q2)
        ]

        # add reassigned edges
        edges.append((min(b1, n1), max(b1, n1), q1))
        edges.append((min(b2, n2), max(b2, n2), q2))

        # drop old boundary node b by leaving it isolated (or you can remove/reindex later)
        # easiest: mark it as "dead" so plotting can hide it
        vtype[b] = "boundary_dead"

    # clean: remove any isolated boundary_dead nodes (optional). For now keep them.
    edges = sorted(set((min(u, v), max(u, v), int(q)) for u, v, q in edges))
    return XY, edges, vtype


def add_boundary_staircase(XY, edges, vtype, side_nodes, inward_dx):
    """
    side_nodes: list of boundary node ids in order (bottom->top or top->bottom)
    inward_dx: move toward interior (+dx for left boundary, -dx for right boundary)
    Adds blue internal vertices between consecutive boundary vertices.
    """
    XY = XY.copy()
    edges = list(edges)
    vtype = list(vtype)

    def add_vertex(xy, typ):
        nonlocal XY, vtype
        vid = len(vtype)
        XY = np.vstack([XY, [xy[0], xy[1]]])
        vtype.append(typ)
        return vid

    def add_edge(u, v, q=-2):
        edges.append((min(u, v), max(u, v), int(q)))  # q=-2 for "boundary decoration"

    for a, b in zip(side_nodes[:-1], side_nodes[1:]):
        xa, ya = XY[a]
        xb, yb = XY[b]
        # inward point: same y as midpoint, x shifted inward
        xm = (xa + xb) / 2.0 + inward_dx
        ym = (ya + yb) / 2.0
        t = add_vertex((xm, ym), "int")  # blue internal
        add_edge(a, t, -2)
        add_edge(t, b, -2)

    edges = sorted(set((min(u, v), max(u, v), int(q)) for u, v, q in edges))
    return XY, edges, vtype


def pick_boundary_sides(XY, vtype, tol=1e-6):
    boundary = [i for i, t in enumerate(vtype) if t == "boundary"]
    xs = XY[boundary, 0]
    minx = xs.min()
    maxx = xs.max()

    left = [b for b in boundary if abs(XY[b, 0] - minx) < tol]
    right = [b for b in boundary if abs(XY[b, 0] - maxx) < tol]

    # sort by Y
    left.sort(key=lambda i: XY[i, 1])
    right.sort(key=lambda i: XY[i, 1])
    return left, right


def add_semicircle_edge(XY, edges, vtype, left_nodes, right_nodes):
    """
    Add a single edge between endpoints to represent the semicircle connection.
    Use q=-3 to tag it specially.
    """
    edges = list(edges)

    # top of left, bottom of right (matches your picture)
    u = left_nodes[-1]
    v = right_nodes[0]
    edges.append((min(u, v), max(u, v), -3))
    return sorted(set((min(a, b), max(a, b), int(q)) for a, b, q in edges))


def dir_from_delta(dx: int, dy: int) -> str:
    if (dx, dy) == (0, 1):
        return "U"
    if (dx, dy) == (1, 0):
        return "R"
    if (dx, dy) == (0, -1):
        return "D"
    if (dx, dy) == (-1, 0):
        return "L"
    raise ValueError(f"Non-grid neighbour delta {(dx, dy)} (expected 4-neighbour).")


def base_adjacency(nV, edges):
    adj = [[] for _ in range(nV)]
    for u, v, q in edges:
        adj[u].append((v, q))
        adj[v].append((u, q))
    return adj


import numpy as np


def decorate_gstar(
    G,
    port_step: float = 0.95,
    blue_step: float = 0.45,
    boundary_split_eps: float = 0.35,
):
    """
    Decorate G* into G*_{∅}.

    Non-boundary vertices:
      - deg-4 -> 6-vertex gadget (4 external white ports U/R/D/L, 2 internal blue stacked)
      - deg-2 -> 2 external white vertices + 1 internal edge between them

    Boundary vertices (yellow):
      - kept as boundary anchors
      - boundary gadget is built as a linked staircase (shared blue vertices) on NW and SE sides:
          * NW: split anchor into two adjacent yellows; add blue above the inner yellow; then
                each next triangle reuses previous blue as the left corner and continues upward
          * SE: split anchor into two adjacent yellows; add blue below the inner yellow; then
                each next triangle reuses previous blue as the right corner and continues downward
      - semicircle edge connects TOPMOST NW blue to BOTTOMMOST SE blue (tag q=-3)

    Returns:
      XY_dec: (n',2) float coords
      edges_dec: list[(u,v,q)] where:
          q == -1 : gadget-internal edges (deg4 triangles, blue-blue, deg2 internal)
          q == -2 : boundary-staircase edges (triangles)
          q == -3 : semicircle connection edge
          q >= 0  : inherited edges with qubit label
      vtype: list[str] in {"boundary","ext","int"}
    """
    L = int(G.L)
    logical = np.asarray(G.logical_xy, dtype=int)
    XY0 = np.asarray(G.expanded_XY, dtype=float)
    n0 = XY0.shape[0]
    boundary = set(G.boundary_vids)

    adj = base_adjacency(n0, G.edges)

    XY_dec = []
    vtype = []
    bmap = {}  # base boundary vertex -> decorated vertex id (kept)

    def add_vertex(xy, t):
        vid = len(XY_dec)
        XY_dec.append((float(xy[0]), float(xy[1])))
        vtype.append(t)
        return vid

    def add_edge(edges, u, v, q):
        if u == v:
            raise RuntimeError("Self-loop created in decorated graph.")
        a, b = (u, v) if u < v else (v, u)
        edges.append((a, b, int(q)))

    # --- keep boundary anchors (yellow) ---
    for v in range(n0):
        if v in boundary:
            bmap[v] = add_vertex(XY0[v], "boundary")

    port_of = {}  # (base_vertex, dir) -> decorated external port id
    edges_dec = []

    # --- place gadgets on non-boundary vertices ---
    for v in range(n0):
        if v in boundary:
            continue

        nbrs = adj[v]
        deg = len(nbrs)
        if deg not in (2, 4):
            raise ValueError(
                f"Non-boundary vertex {v} has degree {deg}; expected 2 or 4."
            )

        vx, vy = logical[v]
        X, Y = XY0[v]

        # neighbour directions
        dir_to_nbr = {}
        for u, q in nbrs:
            dx = int(logical[u, 0] - vx)
            dy = int(logical[u, 1] - vy)
            d = dir_from_delta(dx, dy)
            dir_to_nbr[d] = (u, q)

        if deg == 4:
            # external ports (white)
            pU = add_vertex((X, Y + port_step), "ext")
            pR = add_vertex((X + port_step, Y), "ext")
            pD = add_vertex((X, Y - port_step), "ext")
            pL = add_vertex((X - port_step, Y), "ext")

            # internal (blue) vertices
            bT = add_vertex((X, Y + blue_step), "int")
            bB = add_vertex((X, Y - blue_step), "int")

            # triangles + blue-blue (all internal q=-1)
            add_edge(edges_dec, pU, pR, -1)
            add_edge(edges_dec, pU, bT, -1)
            add_edge(edges_dec, bT, pR, -1)

            add_edge(edges_dec, pD, pL, -1)
            add_edge(edges_dec, pD, bB, -1)
            add_edge(edges_dec, bB, pL, -1)

            add_edge(edges_dec, bT, bB, -1)

            port_of[(v, "U")] = pU
            port_of[(v, "R")] = pR
            port_of[(v, "D")] = pD
            port_of[(v, "L")] = pL

        else:
            # deg-2: two external white vertices + internal edge
            dirs = sorted(dir_to_nbr.keys())

            def port_pos(d):
                if d == "U":
                    return (X, Y + port_step)
                if d == "R":
                    return (X + port_step, Y)
                if d == "D":
                    return (X, Y - port_step)
                if d == "L":
                    return (X - port_step, Y)
                raise ValueError(d)

            d0, d1 = dirs[0], dirs[1]
            p0 = add_vertex(port_pos(d0), "ext")
            p1 = add_vertex(port_pos(d1), "ext")
            add_edge(edges_dec, p0, p1, -1)

            port_of[(v, d0)] = p0
            port_of[(v, d1)] = p1

    # --- boundary attachment map from actual base-graph boundary incidences ---
    boundary_attach = {}  # (boundary_base_vid, neighbor_base_vid) -> decorated yellow vid

    inc_nw = []
    inc_se = []
    y_mid = (L - 1) / 2.0
    for u, v, q in G.edges:
        if u in boundary and v not in boundary:
            b, nb = u, v
        elif v in boundary and u not in boundary:
            b, nb = v, u
        else:
            continue

        bx, by = int(logical[b, 0]), int(logical[b, 1])
        # Split the x==0 boundary incidences by height:
        # top-center belongs to NW staircase, bottom-center belongs to SE staircase.
        if (bx < 0) or (bx == 0 and by > y_mid):
            inc_nw.append((b, nb, q))
        else:
            inc_se.append((b, nb, q))

    inc_nw.sort(
        key=lambda t: (
            int(logical[t[0], 1]),
            int(logical[t[0], 0]),
            int(logical[t[1], 1]),
            int(logical[t[1], 0]),
        )
    )
    # Reverse SE order so the staircase starts at the upper-right and descends to bottom-center.
    # Tie-break on the interior neighbour so duplicated anchors order "up" before "left",
    # which prevents the bottom-right criss-cross.
    inc_se.sort(
        key=lambda t: (
            int(logical[t[0], 1]),
            int(logical[t[0], 0]),
            int(logical[t[1], 1]),
            int(logical[t[1], 0]),
        ),
        reverse=True,
    )

    def build_boundary_staircase_from_incidences(inc, side, anchor_xy):
        """
        Build a linked triangle staircase with one yellow per boundary incidence.
        Populates boundary_attach[(b, nb)] -> Yi.
        """
        nonlocal XY_dec, vtype, edges_dec, boundary_attach

        Lside = len(inc)
        if Lside == 0:
            return [], []

        X0, Y0 = float(anchor_xy[0]), float(anchor_xy[1])
        dx = port_step
        dy = port_step

        yellows = []
        blues = []

        for i, (b, nb, q) in enumerate(inc):
            if side == "NW":
                # lower-left -> upper-center
                xy = (X0 + i * dx, Y0 + i * dy)
            else:  # SE
                # upper-right -> lower-center
                xy = (X0 - i * dx, Y0 - i * dy)

            Yi = add_vertex(xy, "boundary")
            yellows.append(Yi)
            boundary_attach[(b, nb)] = Yi

        # blue spine vertices positioned between consecutive yellows
        for i in range(Lside - 1):
            x_mid = (XY_dec[yellows[i]][0] + XY_dec[yellows[i + 1]][0]) / 2.0
            y_mid = (XY_dec[yellows[i]][1] + XY_dec[yellows[i + 1]][1]) / 2.0
            inward = -0.6 if side == "NW" else +0.6
            Bi = add_vertex((x_mid + inward, y_mid), "int")
            blues.append(Bi)

        # Blue-blue spine (no yellow-yellow edges except the single cap below)
        for i in range(len(blues) - 1):
            add_edge(edges_dec, blues[i], blues[i + 1], -2)

        # Single yellow-yellow cap at the end opposite the semicircle (index 0/1)
        # plus attachments to the first blue, making the first triangle.
        if Lside >= 2:
            add_edge(edges_dec, yellows[0], yellows[1], -2)
            add_edge(edges_dec, yellows[0], blues[0], -2)
            add_edge(edges_dec, yellows[1], blues[0], -2)

        # Remaining triangles use one yellow + two neighbouring blue vertices.
        # Triangle i (for yellow index i+1) is formed by:
        #   yellows[i+1], blues[i-1], blues[i]  for i = 1..Lside-2
        for i in range(1, Lside - 1):
            add_edge(edges_dec, yellows[i + 1], blues[i - 1], -2)
            add_edge(edges_dec, yellows[i + 1], blues[i], -2)

        return yellows, blues

    if boundary:
        # Hide original copied boundary anchors; actual attachments use staircase yellows.
        for dv in bmap.values():
            vtype[dv] = "boundary_dead"

        # NW anchor: lower-left endpoint of NW side; SE anchor: upper-right endpoint of SE side
        nw_side_base = [b for b in boundary if (int(logical[b, 0]) < 0) or (int(logical[b, 0]) == 0 and int(logical[b, 1]) > y_mid)]
        se_side_base = [b for b in boundary if b not in nw_side_base]
        nw_base = min(nw_side_base, key=lambda b: (int(logical[b, 1]), int(logical[b, 0])))
        se_base = max(se_side_base, key=lambda b: (int(logical[b, 1]), int(logical[b, 0])))
        nw_anchor_xy = XY0[nw_base]
        se_anchor_xy = XY0[se_base]

        nw_yellows, nw_blues = build_boundary_staircase_from_incidences(
            inc_nw, "NW", nw_anchor_xy
        )
        se_yellows, se_blues = build_boundary_staircase_from_incidences(
            inc_se, "SE", se_anchor_xy
        )
    else:
        nw_yellows, nw_blues, se_yellows, se_blues = [], [], [], []

    # --- recreate inherited edges (q>=0) using ports / staircase yellows ---
    seen = set()
    for u, v, q in G.edges:
        key = (min(u, v), max(u, v), int(q))
        if key in seen:
            continue
        seen.add(key)

        def endpoint(a, b):
            if a in boundary:
                return boundary_attach[(a, b)]
            dx = int(logical[b, 0] - logical[a, 0])
            dy = int(logical[b, 1] - logical[a, 1])
            d = dir_from_delta(dx, dy)
            return port_of[(a, d)]

        uu = endpoint(u, v)
        vv = endpoint(v, u)
        add_edge(edges_dec, uu, vv, q)

    # --- boundary semicircle (top NW blue to bottom SE blue) ---
    XY_dec = np.asarray(XY_dec, dtype=float)
    edges_dec = sorted(set(edges_dec))
    if nw_blues and se_blues:
        nw_top_blue = max(nw_blues, key=lambda i: XY_dec[i, 1])
        se_bottom_blue = min(se_blues, key=lambda i: XY_dec[i, 1])
        add_edge(edges_dec, nw_top_blue, se_bottom_blue, -3)

    edges_dec = sorted(set(edges_dec))
    return XY_dec, edges_dec, vtype


def _decorate_boundary_linked(
    XY,
    edges,
    vtype,
    anchor_vid: int,
    mode: str,
    step_dx: float,
    step_dy: float,
    split_eps: float,
):
    """
    Linked boundary staircase builder.

    mode="NW":
      - split anchor into two adjacent yellows (outer left, inner right)
      - create blue above inner-right; triangle (yL,yR,b0)
      - then repeat upward: new yellow to the right of previous blue, new blue above it,
        triangle (prev_blue, new_yellow, new_blue)

    mode="SE":
      - split anchor into two adjacent yellows (inner left, outer right)
      - create blue below inner-left; triangle (yL,yR,b0)
      - then repeat downward: new yellow to the left of previous blue, new blue below it,
        triangle (new_yellow, prev_blue, new_blue)

    Returns:
      XY, edges, vtype, top_blue, bottom_blue
    """
    XY = XY.copy()
    edges = list(edges)
    vtype = list(vtype)

    def add_v(xy, typ):
        nonlocal XY, vtype
        vid = len(vtype)
        XY = np.vstack([XY, [float(xy[0]), float(xy[1])]])
        vtype.append(typ)
        return vid

    def add_e(u, v, q=-2):
        a, b = (u, v) if u < v else (v, u)
        edges.append((a, b, int(q)))

    ax, ay = float(XY[anchor_vid, 0]), float(XY[anchor_vid, 1])

    # determine vertical extent from existing boundary anchors (for stopping)
    boundary_y = [float(XY[i, 1]) for i, t in enumerate(vtype) if t == "boundary"]
    if boundary_y:
        y_min = min(boundary_y)
        y_max = max(boundary_y)
    else:
        y_min = ay - 10 * step_dy
        y_max = ay + 10 * step_dy

    if mode == "NW":
        # split into two adjacent yellows (outer left, inner right)
        yL = add_v((ax - split_eps, ay), "boundary")
        yR = add_v((ax + split_eps, ay), "boundary")

        # first blue above inner-right
        b0 = add_v((float(XY[yR, 0]), float(XY[yR, 1]) + step_dy), "int")

        # triangle (yL,yR,b0)
        add_e(yL, yR, -2)
        add_e(yR, b0, -2)
        add_e(yL, b0, -2)

        bottom_blue = b0
        curr_left = b0

        # repeat upward
        while float(XY[curr_left, 1]) + step_dy <= y_max + 1e-6:
            # new yellow to the right of current blue (same height)
            yRk = add_v(
                (float(XY[curr_left, 0]) + step_dx, float(XY[curr_left, 1])), "boundary"
            )
            # new blue above it
            bk = add_v(
                (float(XY[yRk, 0]), float(XY[yRk, 1]) + step_dy), "int"
            )

            # triangle (curr_left, yRk, bk)
            add_e(curr_left, yRk, -2)
            add_e(yRk, bk, -2)
            add_e(bk, curr_left, -2)

            curr_left = bk

        top_blue = curr_left
        return XY, sorted(set(edges)), vtype, top_blue, bottom_blue

    if mode == "SE":
        # split into two adjacent yellows (inner left, outer right)
        yL = add_v((ax - split_eps, ay), "boundary")
        yR = add_v((ax + split_eps, ay), "boundary")

        # first blue below inner-left
        b0 = add_v((float(XY[yL, 0]), float(XY[yL, 1]) - step_dy), "int")

        # triangle (yL,yR,b0)
        add_e(yL, yR, -2)
        add_e(yL, b0, -2)
        add_e(yR, b0, -2)

        top_blue = b0
        curr_right = b0

        # repeat downward
        while float(XY[curr_right, 1]) - step_dy >= y_min - 1e-6:
            # new yellow to the left of current blue (same height)
            yLk = add_v(
                (float(XY[curr_right, 0]) - step_dx, float(XY[curr_right, 1])),
                "boundary",
            )
            # new blue below it
            bk = add_v(
                (float(XY[yLk, 0]), float(XY[yLk, 1]) - step_dy), "int"
            )

            # triangle (yLk, curr_right, bk)
            add_e(yLk, curr_right, -2)
            add_e(curr_right, bk, -2)
            add_e(bk, yLk, -2)

            curr_right = bk

        bottom_blue = curr_right
        return XY, sorted(set(edges)), vtype, top_blue, bottom_blue

    raise ValueError(f"Unknown mode {mode}")

# -------------------------
# helpers for boundary decoration
# -------------------------


def _adj_edges(nV, edges):
    adj = [[] for _ in range(nV)]
    for u, v, q in edges:
        adj[u].append((v, q))
        adj[v].append((u, q))
    return adj


def _split_boundary_degree2(XY, edges, vtype, split_eps=0.35):
    """
    If a boundary node has exactly 2 incident inherited edges (q>=0),
    split into two boundary nodes, each taking one inherited edge.
    The original node is marked 'boundary_dead'.
    """
    XY = XY.copy()
    edges = list(edges)
    vtype = list(vtype)

    nV = XY.shape[0]
    adj = _adj_edges(nV, edges)

    def add_vertex(xy, typ):
        nonlocal XY, vtype
        vid = len(vtype)
        XY = np.vstack([XY, [float(xy[0]), float(xy[1])]])
        vtype.append(typ)
        return vid

    def is_target(u, v, q, bb, nn, qq):
        return q == qq and ((u == bb and v == nn) or (u == nn and v == bb))

    for b in range(nV):
        if vtype[b] != "boundary":
            continue
        inh = [(nbr, q) for nbr, q in adj[b] if q >= 0]
        if len(inh) != 2:
            continue

        (n1, q1), (n2, q2) = inh
        xb, yb = XY[b]

        b1 = add_vertex((xb - split_eps, yb), "boundary")
        b2 = add_vertex((xb + split_eps, yb), "boundary")

        edges = [
            e
            for e in edges
            if not is_target(*e, b, n1, q1) and not is_target(*e, b, n2, q2)
        ]

        edges.append((min(b1, n1), max(b1, n1), int(q1)))
        edges.append((min(b2, n2), max(b2, n2), int(q2)))

        vtype[b] = "boundary_dead"

    edges = sorted(set((min(u, v), max(u, v), int(q)) for u, v, q in edges))
    return XY, edges, vtype


def _pick_boundary_sides(XY, vtype):
    """
    Partition boundary nodes into two sides using x-position relative to the median.
    This is more robust than "extreme x" after splitting, and matches the two
    boundary chains in the reference figure.

    Returns:
      left_nodes  sorted by y (bottom->top)
      right_nodes sorted by y (bottom->top)
    """
    boundary = [i for i, t in enumerate(vtype) if t == "boundary"]
    if not boundary:
        return [], []

    xs = np.asarray(XY, dtype=float)[boundary, 0]
    x_med = float(np.median(xs))

    left = [b for b in boundary if float(XY[b, 0]) <= x_med]
    right = [b for b in boundary if float(XY[b, 0]) > x_med]

    left.sort(key=lambda i: float(XY[i, 1]))
    right.sort(key=lambda i: float(XY[i, 1]))
    return left, right

def _add_vertex(XY, vtype, xy, typ):
    vid = len(vtype)
    XY = np.vstack([XY, [float(xy[0]), float(xy[1])]])
    vtype.append(typ)
    return XY, vtype, vid

def _add_uedge(edges, u, v, q):
    a, b = (u, v) if u < v else (v, u)
    edges.append((a, b, int(q)))

def build_boundary_chain_nodes(XY, vtype, which="NW", tol=1e-6):
    """
    Pick the *square* boundary anchors from your existing boundary vertices.
    Practically: use extreme x plus y filtering.
    For L=5 your anchors are sparse, so this works well.

    Returns a single anchor vertex id (the square corner) for that side.
    """
    b = [i for i, t in enumerate(vtype) if t == "boundary"]
    if not b:
        raise ValueError("No boundary vertices found")

    xs = XY[b, 0]
    ys = XY[b, 1]

    minx = float(xs.min())
    maxx = float(xs.max())

    # NW anchor ~ leftmost with high y (the “leftmost square on NW side”)
    if which == "NW":
        cand = [i for i in b if abs(float(XY[i, 0]) - minx) < tol]
        return max(cand, key=lambda i: float(XY[i, 1]))
    # SE anchor ~ rightmost with low y
    if which == "SE":
        cand = [i for i in b if abs(float(XY[i, 0]) - maxx) < tol]
        return min(cand, key=lambda i: float(XY[i, 1]))

    raise ValueError(which)

def decorate_boundary_linked_staircase(
    XY,
    edges,
    vtype,
    anchor_vid: int,
    direction: str,
    step_dx: float,
    step_dy: float,
    split_eps: float = 0.35,
    q_tri: int = -2,
):
    """
    Builds the linked staircase starting from a SINGLE boundary anchor.

    direction = "NW" (go upward):
      anchor -> split into yL0 (outer, left) and yR0 (inner, right)
      b0 above yR0, triangle (yL0,yR0,b0)
      next triangle uses b0 as left corner: (b0,yR1,b1), etc.

    direction = "SE" (go downward):
      anchor -> split into yL0 (inner, left) and yR0 (outer, right)  (mirrored)
      b0 below yL0, triangle (yL0,yR0,b0)
      next triangle uses b0 as right corner: (yL1,b0,b1), etc.

    Returns:
      XY, edges, vtype, top_blue_vid, bottom_blue_vid
    """
    XY = XY.copy()
    edges = list(edges)
    vtype = list(vtype)

    ax, ay = XY[anchor_vid]

    # split anchor into two yellows side-by-side
    if direction == "NW":
        # outer left, inner right
        XY, vtype, yL = _add_vertex(XY, vtype, (ax - split_eps, ay), "boundary")
        XY, vtype, yR = _add_vertex(XY, vtype, (ax + split_eps, ay), "boundary")
        # first blue above inner-right
        XY, vtype, b = _add_vertex(XY, vtype, (XY[yR, 0], XY[yR, 1] + step_dy), "int")
        # triangle edges
        _add_uedge(edges, yL, yR, q_tri)
        _add_uedge(edges, yR, b, q_tri)
        _add_uedge(edges, yL, b, q_tri)

        bottom_blue = b  # for NW this is lower end of spine (near anchor)
        # build upward steps; each step creates new inner yellow to the right of current blue
        curr_left = b
        curr_y = XY[b, 1]
        # stop when next step would go beyond current boundary extent
        # heuristic: build until we exceed max boundary y by a margin
        max_by = max(
            float(Y)
            for i, Y in enumerate(XY[:, 1])
            if vtype[i] in ("boundary", "boundary_dead")
        )
        while curr_y + step_dy <= max_by + 1e-6:
            # new inner yellow to the right (same height as curr_left)
            XY, vtype, yRk = _add_vertex(
                XY, vtype, (XY[curr_left, 0] + step_dx, XY[curr_left, 1]), "boundary"
            )
            # new blue above that
            XY, vtype, bk = _add_vertex(
                XY, vtype, (XY[yRk, 0], XY[yRk, 1] + step_dy), "int"
            )
            # triangle: curr_left -- yRk -- bk -- curr_left
            _add_uedge(edges, curr_left, yRk, q_tri)
            _add_uedge(edges, yRk, bk, q_tri)
            _add_uedge(edges, bk, curr_left, q_tri)

            curr_left = bk
            curr_y = XY[bk, 1]

        top_blue = curr_left
        return np.asarray(XY, float), sorted(set(edges)), vtype, top_blue, bottom_blue

    if direction == "SE":
        # inner left, outer right (mirrored)
        XY, vtype, yL = _add_vertex(XY, vtype, (ax - split_eps, ay), "boundary")
        XY, vtype, yR = _add_vertex(XY, vtype, (ax + split_eps, ay), "boundary")
        # first blue below inner-left
        XY, vtype, b = _add_vertex(XY, vtype, (XY[yL, 0], XY[yL, 1] - step_dy), "int")
        # triangle edges
        _add_uedge(edges, yL, yR, q_tri)
        _add_uedge(edges, yL, b, q_tri)
        _add_uedge(edges, yR, b, q_tri)

        top_blue = b  # for SE this is upper spine end (near anchor)
        curr_right = b
        curr_y = XY[b, 1]
        min_by = min(
            float(Y)
            for i, Y in enumerate(XY[:, 1])
            if vtype[i] in ("boundary", "boundary_dead")
        )
        while curr_y - step_dy >= min_by - 1e-6:
            # new inner yellow to the left of current blue (same height)
            XY, vtype, yLk = _add_vertex(
                XY, vtype, (XY[curr_right, 0] - step_dx, XY[curr_right, 1]), "boundary"
            )
            # new blue below that
            XY, vtype, bk = _add_vertex(
                XY, vtype, (XY[yLk, 0], XY[yLk, 1] - step_dy), "int"
            )
            # triangle: yLk -- curr_right -- bk -- yLk  (shares curr_right blue)
            _add_uedge(edges, yLk, curr_right, q_tri)
            _add_uedge(edges, curr_right, bk, q_tri)
            _add_uedge(edges, bk, yLk, q_tri)

            curr_right = bk
            curr_y = XY[bk, 1]

        bottom_blue = curr_right
        return np.asarray(XY, float), sorted(set(edges)), vtype, top_blue, bottom_blue

    raise ValueError(direction)


def add_semicircle_edge(edges, top_blue, bottom_blue, q=-3):
    edges = list(edges)
    _add_uedge(edges, top_blue, bottom_blue, q)
    return sorted(set(edges))



def _add_boundary_staircase(XY, edges, vtype, side_nodes, inward_dx):
    """
    Build the boundary "staircase" gadget along one boundary chain.

    For each consecutive pair of boundary (yellow) nodes (a,b), add ONE blue internal node t
    and connect a-b, a-t, b-t (a triangle). Edges are tagged q=-2.

    The blue node is placed at the "corner" of the step (not the midpoint) to make it visible.
    """
    XY = XY.copy()
    edges = list(edges)
    vtype = list(vtype)

    created_blue = []

    def add_vertex(xy, typ):
        nonlocal XY, vtype
        vid = len(vtype)
        XY = np.vstack([XY, [float(xy[0]), float(xy[1])]])
        vtype.append(typ)
        return vid

    def add_edge(u, v, q=-2):
        edges.append((min(u, v), max(u, v), int(q)))

    if len(side_nodes) < 2:
        edges = sorted(set((min(u, v), max(u, v), int(q)) for u, v, q in edges))
        return XY, edges, vtype, created_blue

    for a, b in zip(side_nodes[:-1], side_nodes[1:]):
        xa, ya = float(XY[a, 0]), float(XY[a, 1])
        xb, yb = float(XY[b, 0]), float(XY[b, 1])

        # place blue at the higher node's y to create a visible "step"
        y_hi = max(ya, yb)
        x_hi = xa if ya >= yb else xb

        # move inward to avoid overlapping the yellow nodes
        t = add_vertex((x_hi + inward_dx, y_hi), "int")
        created_blue.append(t)

        # triangle edges (stair step)
        add_edge(a, b, -2)
        add_edge(a, t, -2)
        add_edge(b, t, -2)

    edges = sorted(set((min(u, v), max(u, v), int(q)) for u, v, q in edges))
    return XY, edges, vtype, created_blue

def plot_gstar(g: GStar, use_expanded: bool = True, annotate: bool = True):
    """
    Quick matplotlib plot.
    Boundary-only vertices are colored yellow.
    """
    import matplotlib.pyplot as plt

    coords = g.expanded_XY if use_expanded else g.logical_xy
    coords = coords.astype(float)

    fig, ax = plt.subplots()

    # edges
    for u, v, q in g.edges:
        ax.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]])

    # vertex colors
    n = g.num_vertices()
    colors = np.full(n, "C0", dtype=object)  # default matplotlib blue
    colors[np.array(g.boundary_vids, dtype=int)] = "yellow"

    ax.scatter(
        coords[:, 0], coords[:, 1], s=60, c=colors, edgecolors="k", linewidths=0.5
    )

    if annotate:
        for vid, (x, y) in enumerate(coords):
            ax.text(x + 0.15, y + 0.15, str(vid), fontsize=8)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"G* (L={g.L}) {'expanded' if use_expanded else 'logical'} coords")
    plt.show()


def plot_decorated_graph(dec, annotate: bool = True):
    """
    Quick matplotlib plot for the locally decorated graph from decorate_graph(...).
    Gadget-internal edges (q=-1) are gray; inherited edges are blue.
    Supports either:
      - DecoratedGraph dataclass with .XY/.edges/.boundary_map
      - tuple (XY, edges, vtype), where vtype marks boundary-like vertices
    """
    import matplotlib.pyplot as plt

    if hasattr(dec, "XY") and hasattr(dec, "edges"):
        coords = dec.XY.astype(float)
        edges = dec.edges
        boundary_new_ids = set(getattr(dec, "boundary_map", {}).values())
        vtype = None
    else:
        # legacy tuple return from decorate_graph: (XY_dec, edges_dec, vtype)
        XY_dec, edges_dec, vtype = dec
        coords = np.asarray(XY_dec, dtype=float)
        edges = edges_dec
        boundary_new_ids = {
            i for i, t in enumerate(vtype) if str(t).lower().startswith("boundary")
        }

    n = coords.shape[0]

    fig, ax = plt.subplots()

    for u, v, q in edges:
        if q == -1:
            color, lw = ("0.65", 1.0)  # gadget-internal
        elif q == -2:
            color, lw = ("tab:orange", 1.4)  # boundary staircase
        elif q == -3:
            color, lw = ("0.25", 1.8)  # semicircle
        else:
            color, lw = ("C0", 1.6)  # inherited/base edge
        ax.plot(
            [coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]], c=color, lw=lw
        )

    colors = np.full(n, "C1", dtype=object)  # fallback
    sizes = np.full(n, 45.0, dtype=float)
    visible = np.ones(n, dtype=bool)

    if vtype is not None:
        for i, t in enumerate(vtype):
            t = str(t)
            if t == "int":
                colors[i] = "tab:blue"
                sizes[i] = 55.0
            elif t == "ext":
                colors[i] = "white"
                sizes[i] = 42.0
            elif t == "boundary":
                colors[i] = "yellow"
                sizes[i] = 55.0
            elif t == "boundary_dead":
                visible[i] = False
    elif boundary_new_ids:
        colors[np.array(sorted(boundary_new_ids), dtype=int)] = "yellow"

    ax.scatter(
        coords[visible, 0],
        coords[visible, 1],
        s=sizes[visible],
        c=colors[visible],
        edgecolors="k",
        linewidths=0.6,
    )

    if annotate:
        for vid, (x, y) in enumerate(coords):
            if visible[vid]:
                ax.text(x + 0.12, y + 0.12, str(vid), fontsize=7)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title("Decorated G* (local gadget replacement)")
    plt.show()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ensure repo root is importable when run as:
    #   python graphs/construct_Gstar.py
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from parity_checks import create_H

    L = 5
    Hz = create_H(L, "z")
    g = build_G_from_Hz(Hz, L)
    print("V:", g.num_vertices(), "E:", g.num_edges())
    dec = decorate_gstar(g)

    if hasattr(dec, "XY") and hasattr(dec, "edges"):
        dec_nV = dec.XY.shape[0]
        dec_nE = len(dec.edges)
    else:
        XY_dec, edges_dec, vtype_dec = dec
        visible = [i for i, t in enumerate(vtype_dec) if str(t) != "boundary_dead"]
        visible_set = set(visible)
        dec_nV = len(visible)
        dec_nE = sum(1 for u, v, _ in edges_dec if u in visible_set and v in visible_set)

    print("Decorated V:", dec_nV, "Decorated E:", dec_nE)
    plot_decorated_graph(dec, annotate=False)
