import numpy as np
import matplotlib.pyplot as plt
from parity_checks import create_H
from graph import build_Gstar_from_Hz_rotated


def plot_Gstar(edges, xy, m_internal, Bstar, title="", show_edge_labels=False):
    """
    Visualise G* using the xy coords you already have.

    - internal nodes: 0..m_internal-1
    - boundary nodes: Bstar (subset of m_internal..)
    """
    num_nodes = xy.shape[0]
    Bstar = set(Bstar)

    # Build degree for sizing/debug
    deg = np.zeros(num_nodes, dtype=int)
    for u, v, _ in edges:
        deg[u] += 1
        deg[v] += 1

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw edges
    for u, v, q in edges:
        x0, y0 = xy[u]
        x1, y1 = xy[v]
        ax.plot([x0, x1], [y0, y1], linewidth=1)
        if show_edge_labels:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, str(q), fontsize=7)

    # Draw nodes: internal vs boundary
    internal_ids = np.arange(m_internal, dtype=int)
    boundary_ids = np.array(sorted(Bstar), dtype=int)

    ax.scatter(
        xy[internal_ids, 0],
        xy[internal_ids, 1],
        s=80,
        marker="o",
        zorder=3,
        label="Z-check vertices",
    )

    if boundary_ids.size:
        ax.scatter(
            xy[boundary_ids, 0],
            xy[boundary_ids, 1],
            s=110,
            marker="s",
            zorder=4,
            label="Boundary vertices",
        )

    # Label nodes (optional but useful)
    for v in range(num_nodes):
        ax.text(xy[v, 0] + 0.05, xy[v, 1] + 0.05, str(v), fontsize=9)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title(title or "G* (undecorated)")
    ax.legend(loc="best")
    plt.show()


def print_local_neighbourhood(edges, v, max_items=20):
    """Text check: list neighbours of v with qubit indices."""
    nbrs = []
    for u, w, q in edges:
        if u == v:
            nbrs.append((w, q))
        elif w == v:
            nbrs.append((u, q))
    nbrs.sort(key=lambda t: t[0])
    print(f"v={v} neighbours (nbr, q):", nbrs[:max_items])


def sanity_check_planarity_style(edges, m_internal, Bstar):
    """
    Quick sanity checks that are easy to interpret without pictures.
    """
    num_nodes = m_internal + len(
        set(Bstar)
    )  # not exact if Bstar not contiguous; just for display

    # degree summary
    # (Use actual max node id from edges)
    max_node = 0
    for u, v, _ in edges:
        max_node = max(max_node, u, v)
    N = max_node + 1

    deg = np.zeros(N, dtype=int)
    for u, v, _ in edges:
        deg[u] += 1
        deg[v] += 1

    print("degree stats:", "min", deg.min(), "max", deg.max(), "mean", deg.mean())
    print(
        "internal degree multiset:",
        dict(zip(*np.unique(deg[:m_internal], return_counts=True))),
    )
    b = sorted(Bstar)
    if b:
        print(
            "boundary degree multiset:",
            dict(zip(*np.unique(deg[b], return_counts=True))),
        )


def main():
    L = 5
    Hz = create_H(L, "z")
    edges, num_nodes, Bstar, xy, _ = build_Gstar_from_Hz_rotated(L, Hz)

    # Visual check
    plot_Gstar(
        edges,
        xy,
        m_internal=Hz.shape[0],
        Bstar=Bstar,
        title=f"G* for L={L}",
        show_edge_labels=False,  # set True for L=3/5 only
    )

    # Optional: print a few neighbourhoods
    print_local_neighbourhood(edges, v=0)
    print_local_neighbourhood(edges, v=1)


if __name__ == "__main__":
    main()
