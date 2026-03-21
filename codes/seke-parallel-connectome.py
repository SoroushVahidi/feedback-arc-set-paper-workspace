#!/usr/bin/env python3
# coding: utf-8
"""
WMSF implementation (paper049) for FAIR comparison (UPDATED + PARALLEL SCC)

Parallelization strategy (safe + deterministic):
  - Parallelize across SCCs (each SCC is independent for the pipeline).
  - For "whole graph is a single SCC": parallelize the two orderings (L1 vs L2) using 2 processes.

Output naming requirement (NO timestamp):
  - Ranking CSV filename includes: "Seke" + dataset name
  - Log filename includes: "Seke" + dataset name
"""

import os
import sys
import heapq
import math
import time
import pandas as pd
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# Small logger (stdout + file)
# ============================================================

class TeeLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._f = open(log_path, "w", buffering=1)

    def log(self, msg: str):
        line = msg.rstrip("\n")
        print(line, flush=True)
        self._f.write(line + "\n")

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def auto_workers():
    for k in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        v = os.environ.get(k, "").strip()
        if v.isdigit() and int(v) > 0:
            return int(v)
    c = os.cpu_count() or 1
    return int(c)


# ============================================================
# Graph IO (same as you had)
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


def build_eid_graph_inout(edges_indexed, n_nodes, tol=1e-12):
    """
    Edge-ID arrays with BOTH out- and in-adjacency (for stabilization step).
    """
    m = len(edges_indexed)
    U = [0] * m
    V = [0] * m
    W0 = [0.0] * m
    active = bytearray(m)
    out_adj = [[] for _ in range(n_nodes)]
    in_adj  = [[] for _ in range(n_nodes)]

    for eid, (u, v, w) in enumerate(edges_indexed):
        U[eid] = u
        V[eid] = v
        W0[eid] = float(w)
        if w > tol:
            active[eid] = 1
        out_adj[u].append(eid)
        in_adj[v].append(eid)

    return U, V, W0, active, out_adj, in_adj


# ============================================================
# SCC decomposition (Kosaraju), deterministic adjacency order
# ============================================================

def kosaraju_scc(n_nodes, edges_indexed):
    """
    Returns list of SCCs, each as list of vertices.
    Deterministic if edges_indexed is deterministic.
    """
    outN = [[] for _ in range(n_nodes)]
    inN  = [[] for _ in range(n_nodes)]
    for (u, v, _w) in edges_indexed:
        outN[u].append(v)
        inN[v].append(u)

    seen = bytearray(n_nodes)
    order = []

    for s in range(n_nodes):
        if seen[s]:
            continue
        stack = [(s, 0)]
        seen[s] = 1
        while stack:
            u, it = stack[-1]
            if it >= len(outN[u]):
                order.append(u)
                stack.pop()
                continue
            v = outN[u][it]
            stack[-1] = (u, it + 1)
            if not seen[v]:
                seen[v] = 1
                stack.append((v, 0))

    comp_id = [-1] * n_nodes
    comps = []
    cid = 0

    for s in reversed(order):
        if comp_id[s] != -1:
            continue
        comp = []
        stack = [s]
        comp_id[s] = cid
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in inN[u]:
                if comp_id[v] == -1:
                    comp_id[v] = cid
                    stack.append(v)
        comps.append(comp)
        cid += 1

    return comps, comp_id


def edges_by_scc(edges_indexed, comp_id):
    """
    Returns: dict[scc_index] -> list of edge tuples (u,v,w,eid_global)
    Only edges INSIDE the SCC.
    """
    by = defaultdict(list)
    for eid, (u, v, w) in enumerate(edges_indexed):
        cu = comp_id[u]
        cv = comp_id[v]
        if cu == cv:
            by[cu].append((u, v, w, eid))
    return by


# ============================================================
# Topological order on active edges (deterministic via heap)
# ============================================================

def topo_order_active(n_nodes, adj, V, active):
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
        raise RuntimeError("Topological sort failed: active graph is not acyclic.")

    rank = [0] * n_nodes
    for r, node in enumerate(order):
        rank[node] = r
    return order, rank


def make_reachability_checker(n_nodes, adj, V, active):
    visited = [0] * n_nodes
    stamp = 0

    def reachable(src, target, rank, rank_limit):
        nonlocal stamp
        stamp += 1
        st = stamp

        if src == target:
            return True
        if rank[src] > rank_limit:
            return False

        stack = [src]
        visited[src] = st

        while stack:
            x = stack.pop()
            for eid in adj[x]:
                if not active[eid]:
                    continue
                y = V[eid]
                if rank[y] > rank_limit:
                    continue
                if y == target:
                    return True
                if visited[y] != st:
                    visited[y] = st
                    stack.append(y)
        return False

    return reachable


def _is_acyclic_active(n_nodes, out_adj, V, active):
    try:
        _order, rank = topo_order_active(n_nodes, out_adj, V, active)
        return True, rank
    except RuntimeError:
        return False, None


# ============================================================
# WMSF core steps on one SCC (paper049)
#   removeArcs -> MinimizeFas -> StabilizeFas -> MinimizeFas
# ============================================================

def wmsf_removeArcs_scc(n_nodes, U, V, W0, active, out_adj, in_adj, ordering="L2", tol=1e-12):
    # Win/Wout for L2
    Win = [0.0] * n_nodes
    Wout = [0.0] * n_nodes
    for eid in range(len(U)):
        if not active[eid]:
            continue
        u = U[eid]; v = V[eid]
        w = W0[eid]
        Wout[u] += w
        Win[v]  += w

    eids = [eid for eid in range(len(U)) if active[eid]]

    if ordering.upper() == "L1":
        eids.sort(key=lambda eid: (W0[eid], U[eid], V[eid], eid))
    else:
        def keyL2(eid):
            denom = Win[U[eid]] + Wout[V[eid]]
            if denom <= 0.0:
                denom = 1.0
            return (W0[eid] / denom, W0[eid], U[eid], V[eid], eid)
        eids.sort(key=keyL2)

    F = set()

    # 2-cycles preprocessing: remove earlier edge in ordering
    pos = {eid: i for i, eid in enumerate(eids)}
    pair_to_eid = {}
    for eid in eids:
        pair_to_eid[(U[eid], V[eid])] = eid

    for eid in eids:
        if not active[eid]:
            continue
        u, v = U[eid], V[eid]
        rev = pair_to_eid.get((v, u), None)
        if rev is None or not active[rev]:
            continue
        if pos[eid] < pos[rev]:
            active[eid] = 0
            F.add(eid)

    # safe arcs trimming
    indeg = [0] * n_nodes
    outdeg = [0] * n_nodes
    for u in range(n_nodes):
        for eid in out_adj[u]:
            if active[eid]:
                outdeg[u] += 1
                indeg[V[eid]] += 1

    q = deque()
    for eid in range(len(U)):
        if not active[eid]:
            continue
        u = U[eid]; v = V[eid]
        if indeg[u] == 0 or outdeg[v] == 0:
            q.append(eid)

    safe_tmp = []
    while q:
        eid = q.popleft()
        if not active[eid]:
            continue
        u = U[eid]; v = V[eid]
        if not (indeg[u] == 0 or outdeg[v] == 0):
            continue

        active[eid] = 0
        safe_tmp.append(eid)

        outdeg[u] -= 1
        indeg[v] -= 1

        for ee in out_adj[u]:
            if active[ee]:
                q.append(ee)
        for ee in in_adj[v]:
            if active[ee]:
                q.append(ee)

    # delete by ordering, check every alpha deletions
    m_act = 0
    for eid in range(len(U)):
        if active[eid]:
            m_act += 1
    alpha = max(1, m_act // max(1, n_nodes))
    since_check = 0

    for eid in eids:
        if not active[eid]:
            continue
        active[eid] = 0
        F.add(eid)
        since_check += 1
        if since_check >= alpha:
            since_check = 0
            ok, _rank = _is_acyclic_active(n_nodes, out_adj, V, active)
            if ok:
                break

    ok, _rank = _is_acyclic_active(n_nodes, out_adj, V, active)
    if not ok:
        for eid in eids:
            if not active[eid]:
                continue
            active[eid] = 0
            F.add(eid)
            ok, _rank = _is_acyclic_active(n_nodes, out_adj, V, active)
            if ok:
                break

    # restore safe arcs
    for eid in safe_tmp:
        active[eid] = 1

    return F, safe_tmp


def wmsf_minimizeFas_scc(n_nodes, U, V, W0, active, out_adj, F, tol=1e-12):
    for eid in F:
        active[eid] = 0

    _, rank = topo_order_active(n_nodes, out_adj, V, active)
    reachable = make_reachability_checker(n_nodes, out_adj, V, active)

    cand = sorted(list(F), key=lambda eid: (-W0[eid], U[eid], V[eid], eid))

    for eid in cand:
        u = U[eid]; v = V[eid]
        if rank[u] < rank[v]:
            active[eid] = 1
            F.discard(eid)
            continue
        if not reachable(v, u, rank=rank, rank_limit=rank[u]):
            active[eid] = 1
            F.discard(eid)
            _, rank = topo_order_active(n_nodes, out_adj, V, active)

    return F


def wmsf_stabilizeFas_scc(n_nodes, U, V, W0, active, out_adj, in_adj, F, tol=1e-12):
    WinG = [0.0] * n_nodes
    WoutG = [0.0] * n_nodes
    for eid in range(len(U)):
        u = U[eid]; v = V[eid]
        w = W0[eid]
        WoutG[u] += w
        WinG[v]  += w

    max_passes = max(1, int(math.log2(max(2, n_nodes))))
    for _ in range(max_passes):
        order, _rank = topo_order_active(n_nodes, out_adj, V, active)
        changed = False

        for v in order:
            WinStar = 0.0
            for eid in in_adj[v]:
                if active[eid]:
                    WinStar += W0[eid]
            WoutStar = 0.0
            for eid in out_adj[v]:
                if active[eid]:
                    WoutStar += W0[eid]

            removed_in  = WinG[v]  - WinStar
            removed_out = WoutG[v] - WoutStar

            if removed_in > WoutStar + tol:
                for eid in out_adj[v]:
                    if active[eid]:
                        active[eid] = 0
                        F.add(eid)
                        changed = True
                for eid in in_adj[v]:
                    if (not active[eid]) and (eid in F):
                        active[eid] = 1
                        F.discard(eid)
                        changed = True

            elif removed_out > WinStar + tol:
                for eid in in_adj[v]:
                    if active[eid]:
                        active[eid] = 0
                        F.add(eid)
                        changed = True
                for eid in out_adj[v]:
                    if (not active[eid]) and (eid in F):
                        active[eid] = 1
                        F.discard(eid)
                        changed = True

        if not changed:
            break

    return F


def _sync_active_from_F(active, m, F):
    for eid in range(m):
        active[eid] = 0 if (eid in F) else 1


def _wmsf_pipeline_scc(k, U2, V2, W02, active2, out2, in2, ordering, tol=1e-12):
    F2, _safe = wmsf_removeArcs_scc(k, U2, V2, W02, active2, out2, in2, ordering=ordering, tol=tol)
    for e in F2:
        active2[e] = 0

    F2 = wmsf_minimizeFas_scc(k, U2, V2, W02, active2, out2, F2, tol=tol)
    _sync_active_from_F(active2, len(U2), F2)

    F2 = wmsf_stabilizeFas_scc(k, U2, V2, W02, active2, out2, in2, F2, tol=tol)
    _sync_active_from_F(active2, len(U2), F2)

    F2 = wmsf_minimizeFas_scc(k, U2, V2, W02, active2, out2, F2, tol=tol)
    _sync_active_from_F(active2, len(U2), F2)

    return F2, active2


def _build_local_scc_graph(verts, e_list, tol=1e-12):
    verts_sorted = sorted(verts)
    loc = {v: i for i, v in enumerate(verts_sorted)}
    k = len(verts_sorted)

    edges_local = [(loc[u], loc[v], w, eid_global) for (u, v, w, eid_global) in e_list]
    m_scc = len(edges_local)

    U2 = [0] * m_scc
    V2 = [0] * m_scc
    W02 = [0.0] * m_scc
    eidG = [0] * m_scc
    active2 = bytearray(m_scc)
    out2 = [[] for _ in range(k)]
    in2  = [[] for _ in range(k)]

    for eid2, (uu, vv, ww, eg) in enumerate(edges_local):
        U2[eid2] = uu
        V2[eid2] = vv
        W02[eid2] = float(ww)
        eidG[eid2] = eg
        if ww > tol:
            active2[eid2] = 1
        out2[uu].append(eid2)
        in2[vv].append(eid2)

    return k, U2, V2, W02, eidG, active2, out2, in2


# ============================================================
# Worker (must be top-level for ProcessPoolExecutor)
# ============================================================

def _worker_run_scc(verts, e_list, ordering_choice, tol):
    k, U2, V2, W02, eidG, active2, out2, in2 = _build_local_scc_graph(verts, e_list, tol=tol)
    F2, _ = _wmsf_pipeline_scc(
        k, U2, V2, W02, active2, out2, in2,
        ordering=ordering_choice, tol=tol
    )
    removed_global = {eidG[eid2] for eid2 in F2}
    return removed_global


# ============================================================
# Main WMSF entry (PARALLEL SCC)
# ============================================================

def wmsf_ranking_from_dimacs_parallel(
    dimacs_path,
    output_ranking_csv_path,
    ordering="L2",
    tol=1e-12,
    max_workers=None,
    parallel_min_nodes=200,
    parallel_min_edges=2000,
    log=None
):
    edges_indexed, node_to_index, index_to_node = read_graph_dimacs_agg(dimacs_path)
    n = len(node_to_index)

    comps, comp_id = kosaraju_scc(n, edges_indexed)
    by_scc = edges_by_scc(edges_indexed, comp_id)

    U, V, W0, active_glob0, out_adj_glob, in_adj_glob = build_eid_graph_inout(edges_indexed, n, tol=tol)

    nontrivial = [c for c in comps if len(c) > 1]
    whole_single_scc = (len(nontrivial) == 1 and len(nontrivial[0]) == n)

    if max_workers is None:
        max_workers = auto_workers()

    def run_one(ordering_choice, use_scc_parallel=True):
        active_glob = bytearray(active_glob0)
        F_global = set()

        tasks = []
        small_work = []

        # NOTE: SCC index equals the order comps were appended in kosaraju_scc
        for scc_idx, verts in enumerate(comps):
            e_list = by_scc.get(scc_idx, [])

            if len(verts) <= 1:
                # remove self-loops (if any)
                for (u, v, w, eid) in e_list:
                    if u == v and w > tol:
                        F_global.add(eid)
                        active_glob[eid] = 0
                continue

            if not e_list:
                continue

            if use_scc_parallel and (len(verts) >= parallel_min_nodes or len(e_list) >= parallel_min_edges):
                tasks.append((verts, e_list))
            else:
                small_work.append((verts, e_list))

        # sequential small SCCs
        for verts, e_list in small_work:
            removed = _worker_run_scc(verts, e_list, ordering_choice, tol)
            for eg in removed:
                if active_glob[eg]:
                    active_glob[eg] = 0
                    F_global.add(eg)

        # parallel big SCCs
        if tasks:
            w = min(max_workers, len(tasks))
            with ProcessPoolExecutor(max_workers=w) as ex:
                futs = [ex.submit(_worker_run_scc, verts, e_list, ordering_choice, tol) for verts, e_list in tasks]
                for fut in as_completed(futs):
                    removed = fut.result()
                    for eg in removed:
                        if active_glob[eg]:
                            active_glob[eg] = 0
                            F_global.add(eg)

        # final topo on global graph
        order_nodes, rank = topo_order_active(n, out_adj_glob, V, active_glob)
        scores = {i: int(rank[i]) for i in range(n)}
        _tot, _fw, bw = compute_forward_backward(edges_indexed, scores)
        return bw, scores, F_global, active_glob

    # If whole graph is a single SCC:
    # - SCC-parallelism does not help (there is only one SCC)
    # - Run L1 and L2 in parallel (2 processes) and choose better BW
    if whole_single_scc:
        if log:
            log.log("Detected: whole graph is ONE SCC -> running L1 and L2 in parallel and choosing min BW.")
        with ProcessPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(run_one, "L1", False)
            f2 = ex.submit(run_one, "L2", False)
            bw1, scores1, F1, active1 = f1.result()
            bw2, scores2, F2, active2 = f2.result()

        if bw1 <= bw2:
            best_scores, best_F, best_active, best_ordering, best_bw = scores1, F1, active1, "L1", bw1
        else:
            best_scores, best_F, best_active, best_ordering, best_bw = scores2, F2, active2, "L2", bw2

        if log:
            log.log(f"Chosen ordering for single-SCC case: {best_ordering} (BW={best_bw:.6f})")

    else:
        if log:
            log.log(f"Multi-SCC case -> SCC-parallel enabled. Ordering={ordering}, max_workers={max_workers}")
        _bw, best_scores, best_F, best_active = run_one(ordering, True)

    # write ranking CSV
    rows = [{"Node ID": str(index_to_node[i]).strip(), "Order": int(best_scores[i])} for i in range(n)]
    rows.sort(key=lambda r: r["Order"])
    pd.DataFrame(rows).to_csv(output_ranking_csv_path, index=False)

    F_removed_pairs = {(U[eid], V[eid]) for eid in best_F}
    return edges_indexed, node_to_index, index_to_node, best_scores, F_removed_pairs


# ============================================================
# Example usage (NO timestamp; filenames include "Seke" + dataset name)
# ============================================================

if __name__ == "__main__":
    # ---- INPUT (set inside main, no CLI args) ----
    edge_file = "/mmfs1/home/sv96/Feedback-arc-set-paper/datasets/connectome.d"
    tol = 1e-12
    ordering = "L2"  # used only if graph is NOT a single SCC

    # ---- names: must include "Seke" + dataset name; no timestamp ----
    dataset_name = os.path.splitext(os.path.basename(edge_file))[0]  # e.g., arwiki202601
    out_csv = os.path.join(os.path.dirname(edge_file), f"{dataset_name}_Seke_wmsf_ranking.csv")
    log_path = os.path.join(os.path.dirname(edge_file), f"{dataset_name}_Seke_wmsf.log")

    logger = TeeLogger(log_path)

    try:
        t0 = time.perf_counter()
        workers = auto_workers()
        logger.log(f"=== WMSF (paper049) PARALLEL SCC ===")
        logger.log(f"Dataset: {dataset_name}")
        logger.log(f"Edge file: {edge_file}")
        logger.log(f"Output ranking CSV: {out_csv}")
        logger.log(f"Log file: {log_path}")
        logger.log(f"Workers (env auto): {workers}")
        logger.log(f"Ordering (multi-SCC): {ordering}")
        logger.log(f"tol: {tol}")

        edges_indexed, node_to_index, index_to_node, scores, F_removed = wmsf_ranking_from_dimacs_parallel(
            dimacs_path=edge_file,
            output_ranking_csv_path=out_csv,
            ordering=ordering,
            tol=tol,
            max_workers=workers,
            parallel_min_nodes=200,
            parallel_min_edges=2000,
            log=logger
        )

        total_w, fw, bw = compute_forward_backward(edges_indexed, scores)
        elapsed_sec = time.perf_counter() - t0

        logger.log("")
        logger.log(f"✅ Wrote ranking: {out_csv}")
        logger.log(f"Graph: n={len(node_to_index)} nodes, m={len(edges_indexed)} edges (after aggregation)")
        logger.log(f"Total Weight: {total_w:.6f}")
        logger.log(f"Forward Weight: {fw:.6f}")
        logger.log(f"Backward Weight: {bw:.6f}")
        logger.log(f"Forward Ratio: {fw/total_w:.6f}")
        logger.log(f"Removed edges (count): {len(F_removed)}")
        logger.log(f"⏱️ Running time: {elapsed_sec:.3f} seconds ({elapsed_sec/60.0:.3f} minutes)")

    finally:
        logger.close()
