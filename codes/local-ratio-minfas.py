#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import heapq
import time
import pandas as pd
from collections import defaultdict

# ============================================================
# 1) DIMACS reader (aggregates parallel arcs) -> paper-friendly
# ============================================================

def read_graph_dimacs_agg(file_path):
    """
    Reads DIMACS-like lines:
      a <source> <target> <weight> <transit_time...>

    - Aggregates parallel arcs: multiple (u,v) become one (u,v) with summed weight
    - Deterministic node mapping (sorted node ids)
    - Deterministic edge order (sorted by (u_idx, v_idx))
    Returns:
      edges_indexed: list[(u_idx, v_idx, w_sum)]
      node_to_index: dict[node_id_str -> int]
      index_to_node: dict[int -> node_id_str]
    """
    agg = defaultdict(float)
    node_ids = set()

    with open(file_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("c", "p")):
                continue
            if not line.startswith("a"):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            u = parts[1]
            v = parts[2]
            try:
                w = float(parts[3])
            except ValueError:
                continue

            node_ids.add(u)
            node_ids.add(v)
            agg[(u, v)] += w

    node_list = sorted(node_ids)
    node_to_index = {node: i for i, node in enumerate(node_list)}
    index_to_node = {i: node for node, i in node_to_index.items()}

    edges_indexed = [(node_to_index[u], node_to_index[v], float(w_sum))
                     for (u, v), w_sum in agg.items()]
    edges_indexed.sort(key=lambda e: (e[0], e[1]))
    return edges_indexed, node_to_index, index_to_node


# ============================================================
# 2) Fast graph representation: edge IDs + active flags
#    (NO adjacency rebuild per iteration)
# ============================================================

def build_eid_graph(edges_indexed, n_nodes, tol=1e-12):
    """
    Converts edges_indexed into edge-id arrays for speed.

    Returns:
      U, V: list[int] endpoints per edge id
      W0:  list[float] original weights
      W:   list[float] mutable reduced weights
      active: bytearray (1 if W>tol else 0)
      adj: list[list[int]] adjacency lists of edge IDs (outgoing)
    """
    m = len(edges_indexed)
    U = [0] * m
    V = [0] * m
    W0 = [0.0] * m
    W = [0.0] * m
    active = bytearray(m)
    adj = [[] for _ in range(n_nodes)]

    for eid, (u, v, w) in enumerate(edges_indexed):
        U[eid] = u
        V[eid] = v
        W0[eid] = float(w)
        W[eid] = float(w)
        if w > tol:
            active[eid] = 1
        adj[u].append(eid)

    # edges_indexed was sorted by (u,v), so for each u, adj[u] is already deterministic.
    return U, V, W0, W, active, adj


# ============================================================
# 3) Cycle finding in the ACTIVE graph (edge-id adjacency)
#    (NO O(n) array reinitialization; reset only touched nodes)
# ============================================================

def find_any_cycle_eids(n_nodes, adj, U, V, active):
    """
    Finds any directed cycle in the active graph.
    Returns: list of edge IDs forming a directed cycle, or None if acyclic.

    Implementation details:
      - iterative DFS
      - skips inactive edges
      - resets only visited nodes (not full O(n) reset)
    """
    state = bytearray(n_nodes)  # 0=unvisited, 1=visiting, 2=done (for THIS call)
    parent = [-1] * n_nodes
    parent_eid = [-1] * n_nodes
    next_ptr = [0] * n_nodes
    visited_nodes = []

    def reset():
        for x in visited_nodes:
            state[x] = 0
            parent[x] = -1
            parent_eid[x] = -1
            next_ptr[x] = 0
        visited_nodes.clear()

    for s in range(n_nodes):
        if state[s] != 0:
            continue

        stack = [s]
        state[s] = 1
        visited_nodes.append(s)

        while stack:
            u = stack[-1]
            i = next_ptr[u]

            # advance to next ACTIVE outgoing edge
            out = adj[u]
            while i < len(out) and active[out[i]] == 0:
                i += 1
            next_ptr[u] = i

            if i >= len(out):
                state[u] = 2
                stack.pop()
                continue

            eid = out[i]
            v = V[eid]
            next_ptr[u] = i + 1  # move forward

            if state[v] == 0:
                parent[v] = u
                parent_eid[v] = eid
                state[v] = 1
                visited_nodes.append(v)
                stack.append(v)
            elif state[v] == 1:
                # back-edge u->v forms a cycle
                cycle_eids = [eid]
                cur = u
                # follow parents from u back to v
                while cur != v:
                    pe = parent_eid[cur]
                    if pe == -1:
                        break
                    cycle_eids.append(pe)
                    cur = parent[cur]
                    if cur == -1:
                        break

                if cur == v:
                    cycle_eids.reverse()
                    reset()
                    return cycle_eids
                # else: something inconsistent (shouldn't happen), continue

        # finished component; continue to next s

    reset()
    return None


# ============================================================
# 4) Topological order on ACTIVE edges (deterministic)
# ============================================================

def topo_order_active(n_nodes, adj, V, active):
    """
    Kahn topo sort on active graph.
    Returns:
      order: list of node indices
      rank: list[int] rank[node] in 0..n-1
    Raises if graph is not acyclic.
    """
    indeg = [0] * n_nodes
    for u in range(n_nodes):
        for eid in adj[u]:
            if active[eid]:
                indeg[V[eid]] += 1

    heap = []
    for i in range(n_nodes):
        if indeg[i] == 0:
            heapq.heappush(heap, i)

    order = []
    while heap:
        u = heapq.heappop(heap)
        order.append(u)
        for eid in adj[u]:
            if not active[eid]:
                continue
            v = V[eid]
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, v)

    if len(order) != n_nodes:
        raise RuntimeError("Topological sort failed: active graph is not acyclic (unexpected).")

    rank = [0] * n_nodes
    for r, node in enumerate(order):
        rank[node] = r
    return order, rank


# ============================================================
# 5) Reachability test for add-back: cycle iff v reaches u
#    (prune by topo rank interval)
# ============================================================

def make_reachability_checker(n_nodes, adj, V, active):
    """
    Returns a closure reachable(src, target, rank, rank_limit).
    Uses stamp-based visited to avoid clearing per call.
    """
    visited = [0] * n_nodes
    stamp = 0

    def reachable(src, target, rank, rank_limit):
        nonlocal stamp
        stamp += 1
        st = stamp

        # quick cases
        if src == target:
            return True
        if rank[src] > rank_limit:
            return False  # pruned by interval

        stack = [src]
        visited[src] = st

        while stack:
            x = stack.pop()
            for eid in adj[x]:
                if not active[eid]:
                    continue
                y = V[eid]
                if rank[y] > rank_limit:
                    continue  # prune by topo interval
                if y == target:
                    return True
                if visited[y] != st:
                    visited[y] = st
                    stack.append(y)
        return False

    return reachable


# ============================================================
# 6) Paper algorithm: Local-ratio cycle reductions + add-back
#    Improvements:
#      - no adjacency rebuild
#      - cycle finder resets only touched nodes
#      - add-back: heavy edges first
#      - add-back: O(1) accept if rank[u] < rank[v]
#      - else: reachability v->u with topo interval pruning
#      - recompute topo ranks only when needed
# ============================================================

def local_ratio_fas_fast(edges_indexed, n_nodes, tol=1e-12):
    """
    Returns:
      removed_eids: set[int] edge IDs in the final minimal FAS (under heuristic order)
      U,V,W0: edge arrays (for reporting)
      active, adj: final active DAG
    """
    U, V, W0, W, active, adj = build_eid_graph(edges_indexed, n_nodes, tol=tol)

    removed_eids = set()

    # ---- Phase 1: cycle reductions ----
    while True:
        cyc = find_any_cycle_eids(n_nodes, adj, U, V, active)
        if cyc is None:
            break

        # eps = min reduced weight on this cycle
        eps = None
        for eid in cyc:
            w = W[eid]
            if eps is None or w < eps:
                eps = w

        if eps is None or eps <= tol:
            # numerical safety: deactivate the first edge on the cycle
            eid0 = cyc[0]
            if active[eid0]:
                active[eid0] = 0
                W[eid0] = 0.0
                removed_eids.add(eid0)
            continue

        # subtract eps from all cycle edges
        for eid in cyc:
            new_w = W[eid] - eps
            W[eid] = new_w
            if new_w <= tol and active[eid]:
                active[eid] = 0
                W[eid] = 0.0
                removed_eids.add(eid)

    # At this point, active graph should be a DAG.
    # ---- Phase 2: add-back (heavy first) ----
    removed_list = sorted(
        list(removed_eids),
        key=lambda eid: (-W0[eid], U[eid], V[eid])
    )

    # initial topo ranks
    _, rank = topo_order_active(n_nodes, adj, V, active)
    reachable = make_reachability_checker(n_nodes, adj, V, active)

    for eid in removed_list:
        u = U[eid]
        v = V[eid]

        # Fast accept: respects current topo order => cannot form cycle
        if rank[u] < rank[v]:
            active[eid] = 1
            removed_eids.discard(eid)
            continue

        # Otherwise, adding u->v creates a cycle iff v reaches u
        # Prune search to nodes with rank <= rank[u]
        if not reachable(v, u, rank=rank, rank_limit=rank[u]):
            active[eid] = 1
            removed_eids.discard(eid)

            # rank may no longer be a valid topo order after adding a "backward" edge.
            # For correctness of future O(1) checks and pruning, recompute ranks now.
            _, rank = topo_order_active(n_nodes, adj, V, active)
        # else: keep it removed

    return removed_eids, U, V, W0, active, adj


# ============================================================
# 7) End-to-end: DIMACS -> paper FAS -> topo ranking CSV
# ============================================================

def paper_fas_ranking_from_dimacs_fast(dimacs_path, output_ranking_csv_path, tol=1e-12):
    """
    Produces a ranking CSV (Node ID, Order) using the paper algorithm with speedups.
    Compatible with your forward/backward sum checker (rank_u < rank_v).
    """
    edges_indexed, node_to_index, index_to_node = read_graph_dimacs_agg(dimacs_path)
    n = len(node_to_index)

    removed_eids, U, V, W0, active, adj = local_ratio_fas_fast(edges_indexed, n, tol=tol)

    # Final topo order gives ranking
    order, rank = topo_order_active(n, adj, V, active)

    # Write ranking: Node ID, Order
    rows = [{"Node ID": str(index_to_node[i]).strip(), "Order": int(rank[i])} for i in range(n)]
    rows.sort(key=lambda r: r["Order"])
    pd.DataFrame(rows).to_csv(output_ranking_csv_path, index=False)

    # Convert removed edges to (u,v) pairs for reporting
    F_removed_pairs = {(U[eid], V[eid]) for eid in removed_eids}

    scores = {i: int(rank[i]) for i in range(n)}
    return edges_indexed, node_to_index, index_to_node, scores, F_removed_pairs


# ============================================================
# 8) Forward/backward evaluation helper (your semantics)
# ============================================================

def compute_forward_backward(edges_indexed, scores):
    total_w = 0.0
    fw = 0.0
    for u, v, w in edges_indexed:
        total_w += w
        if scores[u] < scores[v]:
            fw += w
    bw = total_w - fw
    return total_w, fw, bw


# ============================================================
# 9) Main
# ============================================================

if __name__ == "__main__":
    edge_file = "/mmfs1/home/sv96/Feedback-arc-set-paper/datasets/connectome.d"
    out_csv = edge_file.replace(".d", "") + "_paper_fas_ranking.csv"

    t0 = time.perf_counter()

    edges_indexed, node_to_index, index_to_node, scores, F_removed = paper_fas_ranking_from_dimacs_fast(
        dimacs_path=edge_file,
        output_ranking_csv_path=out_csv,
        tol=1e-12
    )

    total_w, fw, bw = compute_forward_backward(edges_indexed, scores)

    elapsed_sec = time.perf_counter() - t0

    print(f"✅ Wrote ranking: {out_csv}")
    print(f"Graph: n={len(node_to_index)} nodes, m={len(edges_indexed)} edges (after aggregation)")
    print(f"Total Weight: {total_w:.6f}")
    print(f"Forward Weight: {fw:.6f}")
    print(f"Backward Weight: {bw:.6f}")
    print(f"Forward Ratio: {fw/total_w:.6f}")
    print(f"Removed edges in minimal FAS (count): {len(F_removed)}")
    print(f"⏱️ Running time: {elapsed_sec:.3f} seconds ({elapsed_sec/60.0:.3f} minutes)")


# In[ ]:




