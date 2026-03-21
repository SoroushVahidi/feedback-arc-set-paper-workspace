#!/usr/bin/env python3
# coding: utf-8
"""
FAST + PARALLEL SCC-Neighborhood LNS MERGE with GUARANTEE (HPC/SLURM friendly)

Update in THIS version (what you asked):
✅ Still writes ONE log file (overwritten each run)
✅ AND ALSO prints logs to screen in real-time (master + workers)
✅ Flushes prints so you see progress immediately even on clusters
✅ No folders created
✅ Final CSV overwritten each run (same filename)
✅ Final BW printed on screen at the end
✅ Never worse guarantee preserved:
    - Each worker starts from best of {WMSF-L1, WMSF-L2, LR}
    - Worker only accepts strict improvements
    - Master picks best worker

Why you previously "saw nothing":
- Many clusters buffer stdout; also QueueListener was writing only to file and
  StreamHandler level was WARNING. Now we:
  - attach a screen handler at INFO
  - force flush on prints
  - have a dedicated "screen printer" thread that prints worker log records live
"""

import csv
import heapq
import logging
import os
import random
import shutil
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from multiprocessing import get_context
from typing import List, Tuple, Optional

# ============================================================
# Globals shared under fork (COW)
# ============================================================

_G = {
    "loaded": False,
    "dimacs_path": None,
    "edges_indexed": None,
    "node_to_index": None,
    "index_to_node": None,
    "n": None,
    "U": None,
    "V": None,
    "W0": None,
    "W_init": None,
    "active_init": None,
    "out_adj": None,
    "in_adj": None,
    "scc_list": None,
}

_WORKER_LOG_QUEUE = None


def _init_worker_logging(q):
    global _WORKER_LOG_QUEUE
    _WORKER_LOG_QUEUE = q


# ============================================================
# Helpers: robust prints for HPC
# ============================================================

def eprint(msg: str):
    print(msg, file=sys.stderr, flush=True)


# ============================================================
# Time/workers helpers
# ============================================================

def _parse_hhmmss_to_seconds(s: str) -> Optional[int]:
    if not s:
        return None
    s = s.strip()
    try:
        if "-" in s:
            d_part, t_part = s.split("-", 1)
            days = int(d_part)
            t = t_part
        else:
            days = 0
            t = s

        parts = t.split(":")
        if len(parts) == 3:
            hh, mm, ss = map(int, parts)
        elif len(parts) == 2:
            hh = 0
            mm, ss = map(int, parts)
        elif len(parts) == 1:
            hh = 0
            mm = 0
            ss = int(parts[0])
        else:
            return None
        return days * 86400 + hh * 3600 + mm * 60 + ss
    except Exception:
        return None


def get_time_budget_seconds(default_seconds: int = 3600) -> int:
    env = os.environ
    for k in ("LNS_TIME_LIMIT_SECONDS", "TIME_LIMIT_SECONDS"):
        if k in env:
            try:
                return max(1, int(env[k]))
            except Exception:
                pass
    if "SLURM_TIMELIMIT" in env:
        sec = _parse_hhmmss_to_seconds(env.get("SLURM_TIMELIMIT", ""))
        if sec is not None and sec > 0:
            return sec
    return default_seconds


def get_num_workers() -> int:
    env = os.environ
    for k in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", "NUM_WORKERS"):
        if k in env:
            try:
                return max(1, int(env[k]))
            except Exception:
                pass
    return max(1, os.cpu_count() or 1)


def default_mp_start_method() -> str:
    return "fork" if os.name == "posix" else "spawn"


# ============================================================
# Logging: file + screen, both at INFO
# ============================================================

def setup_root_logger_file_and_screen(log_path: str) -> logging.Logger:
    logger = logging.getLogger("ALL")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir and not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(processName)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ============================================================
# Atomic CSV write (fast, no pandas)
# ============================================================

def write_ranking_csv_atomic(index_to_node, rank, out_path: str):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    tmp_path = out_path + ".tmp"

    n = len(rank)
    order_to_node = [0] * n
    for node in range(n):
        order_to_node[rank[node]] = node

    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Node ID", "Order"])
        for order, node in enumerate(order_to_node):
            w.writerow([str(index_to_node[node]).strip(), int(order)])

    os.replace(tmp_path, out_path)


# ============================================================
# 1) DIMACS reader (aggregates parallel arcs)
# ============================================================

def read_graph_dimacs_agg(file_path: str):
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

            if w <= 0.0:
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
# 2) Edge-ID graph
# ============================================================

def build_eid_graph_inout_with_W(edges_indexed, n_nodes, tol=1e-12):
    m = len(edges_indexed)
    U = [0] * m
    V = [0] * m
    W0 = [0.0] * m
    W  = [0.0] * m
    active = bytearray(m)
    out_adj = [[] for _ in range(n_nodes)]
    in_adj  = [[] for _ in range(n_nodes)]

    for eid, (u, v, w) in enumerate(edges_indexed):
        U[eid] = u
        V[eid] = v
        ww = float(w)
        W0[eid] = ww
        W[eid] = ww
        if ww > tol:
            active[eid] = 1
        out_adj[u].append(eid)
        in_adj[v].append(eid)

    return U, V, W0, W, active, out_adj, in_adj


# ============================================================
# 3) Global topo order on ACTIVE edges
# ============================================================

def topo_order_active(n_nodes, out_adj, V, active):
    indeg = [0] * n_nodes
    for u in range(n_nodes):
        for eid in out_adj[u]:
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
        for eid in out_adj[u]:
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


# ============================================================
# 4) SCC decomposition (Kosaraju)
# ============================================================

def kosaraju_scc(n_nodes, edges_indexed):
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
    by = defaultdict(list)
    for eid, (u, v, w) in enumerate(edges_indexed):
        cu = comp_id[u]
        cv = comp_id[v]
        if cu == cv:
            by[cu].append((u, v, w, eid))
    return by


# ============================================================
# 5) BW eval (array-based)
# ============================================================

def compute_fw_bw_from_rank(U, V, W0, rank) -> Tuple[float, float, float]:
    total = 0.0
    fw = 0.0
    for eid in range(len(U)):
        w = W0[eid]
        total += w
        if rank[U[eid]] < rank[V[eid]]:
            fw += w
    bw = total - fw
    return total, fw, bw


# ============================================================
# 6) Restricted SCC routines
# ============================================================

def find_any_cycle_eids_restricted(n_nodes, out_adj, V, active, allowed_nodes, allowed_eids):
    state = bytearray(n_nodes)
    parent = [-1] * n_nodes
    parent_eid = [-1] * n_nodes
    next_ptr = [0] * n_nodes
    touched = []

    def reset():
        for x in touched:
            state[x] = 0
            parent[x] = -1
            parent_eid[x] = -1
            next_ptr[x] = 0
        touched.clear()

    for s in range(n_nodes):
        if not allowed_nodes[s]:
            continue
        if state[s] != 0:
            continue

        stack = [s]
        state[s] = 1
        touched.append(s)

        while stack:
            u = stack[-1]
            i = next_ptr[u]
            out = out_adj[u]

            while i < len(out):
                eid = out[i]
                if active[eid] and allowed_eids[eid]:
                    v = V[eid]
                    if allowed_nodes[v]:
                        break
                i += 1
            next_ptr[u] = i

            if i >= len(out):
                state[u] = 2
                stack.pop()
                continue

            eid = out[i]
            v = V[eid]
            next_ptr[u] = i + 1

            if state[v] == 0:
                parent[v] = u
                parent_eid[v] = eid
                state[v] = 1
                touched.append(v)
                stack.append(v)
            elif state[v] == 1:
                cycle = [eid]
                cur = u
                while cur != v:
                    pe = parent_eid[cur]
                    if pe == -1:
                        break
                    cycle.append(pe)
                    cur = parent[cur]
                    if cur == -1:
                        break
                if cur == v:
                    cycle.reverse()
                    reset()
                    return cycle

    reset()
    return None


def topo_order_active_restricted(n_nodes, out_adj, V, active, allowed_nodes, allowed_eids):
    indeg = [0] * n_nodes
    nodes = [i for i in range(n_nodes) if allowed_nodes[i]]

    for u in nodes:
        for eid in out_adj[u]:
            if not (active[eid] and allowed_eids[eid]):
                continue
            v = V[eid]
            if allowed_nodes[v]:
                indeg[v] += 1

    heap = []
    for u in nodes:
        if indeg[u] == 0:
            heapq.heappush(heap, u)

    order = []
    while heap:
        u = heapq.heappop(heap)
        order.append(u)
        for eid in out_adj[u]:
            if not (active[eid] and allowed_eids[eid]):
                continue
            v = V[eid]
            if not allowed_nodes[v]:
                continue
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, v)

    if len(order) != len(nodes):
        raise RuntimeError("Restricted topo failed: SCC subgraph is cyclic.")

    rank = [-1] * n_nodes
    for r, u in enumerate(order):
        rank[u] = r
    return order, rank


def make_reachability_checker_restricted(n_nodes, out_adj, V, active, allowed_nodes, allowed_eids):
    visited = [0] * n_nodes
    stamp = 0

    def reachable(src, target, rank, rank_limit):
        nonlocal stamp
        stamp += 1
        st = stamp
        if src == target:
            return True
        if not allowed_nodes[src] or not allowed_nodes[target]:
            return False
        if rank[src] < 0 or rank[target] < 0:
            return False
        if rank[src] > rank_limit:
            return False

        stack = [src]
        visited[src] = st
        while stack:
            x = stack.pop()
            for eid in out_adj[x]:
                if not (active[eid] and allowed_eids[eid]):
                    continue
                y = V[eid]
                if not allowed_nodes[y]:
                    continue
                if rank[y] > rank_limit:
                    continue
                if y == target:
                    return True
                if visited[y] != st:
                    visited[y] = st
                    stack.append(y)
        return False

    return reachable


# ============================================================
# 7) Local-ratio repair + minimize add-back (inside SCC)
# ============================================================

def local_ratio_repair_inside_scc(U, V, W0, active, out_adj,
                                 allowed_nodes, allowed_eids, internal_eids,
                                 tol=1e-12):
    W = {eid: W0[eid] for eid in internal_eids}
    F_add = []

    while True:
        cyc = find_any_cycle_eids_restricted(
            n_nodes=len(out_adj),
            out_adj=out_adj,
            V=V,
            active=active,
            allowed_nodes=allowed_nodes,
            allowed_eids=allowed_eids
        )
        if cyc is None:
            break

        eps = None
        for eid in cyc:
            ww = W.get(eid, W0[eid])
            if eps is None or ww < eps:
                eps = ww

        if eps is None or eps <= tol:
            e0 = cyc[0]
            if active[e0]:
                active[e0] = 0
                F_add.append(e0)
            continue

        for eid in cyc:
            new_w = W.get(eid, W0[eid]) - eps
            W[eid] = new_w
            if new_w <= tol and active[eid]:
                active[eid] = 0
                F_add.append(eid)

    return F_add


def minimize_addback_inside_scc(U, V, W0, active, out_adj, inF, allowed_nodes, allowed_eids, internal_eids):
    _, rank = topo_order_active_restricted(
        n_nodes=len(out_adj),
        out_adj=out_adj,
        V=V,
        active=active,
        allowed_nodes=allowed_nodes,
        allowed_eids=allowed_eids
    )
    reachable = make_reachability_checker_restricted(
        n_nodes=len(out_adj),
        out_adj=out_adj,
        V=V,
        active=active,
        allowed_nodes=allowed_nodes,
        allowed_eids=allowed_eids
    )

    cand = [eid for eid in internal_eids if inF[eid]]
    cand.sort(key=lambda eid: (-W0[eid], U[eid], V[eid], eid))

    for eid in cand:
        u = U[eid]
        v = V[eid]
        if not (allowed_nodes[u] and allowed_nodes[v]):
            continue

        if rank[u] < rank[v]:
            active[eid] = 1
            inF[eid] = 0
            continue

        if not reachable(v, u, rank=rank, rank_limit=rank[u]):
            active[eid] = 1
            inF[eid] = 0
            _, rank = topo_order_active_restricted(
                n_nodes=len(out_adj),
                out_adj=out_adj,
                V=V,
                active=active,
                allowed_nodes=allowed_nodes,
                allowed_eids=allowed_eids
            )
            reachable = make_reachability_checker_restricted(
                n_nodes=len(out_adj),
                out_adj=out_adj,
                V=V,
                active=active,
                allowed_nodes=allowed_nodes,
                allowed_eids=allowed_eids
            )


# ============================================================
# 8) WMSF seed (best of L1/L2)
# ============================================================

def _is_acyclic_active(n_nodes, out_adj, V, active):
    try:
        topo_order_active(n_nodes, out_adj, V, active)
        return True
    except RuntimeError:
        return False


def wmsf_removeArcs_global(n_nodes, U, V, W0, active, out_adj, in_adj, ordering="L2"):
    Win = [0.0] * n_nodes
    Wout = [0.0] * n_nodes
    for eid in range(len(U)):
        if not active[eid]:
            continue
        Wout[U[eid]] += W0[eid]
        Win[V[eid]] += W0[eid]

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

    inF = bytearray(len(U))

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
            inF[eid] = 1

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
        u = U[eid]
        v = V[eid]
        if indeg[u] == 0 or outdeg[v] == 0:
            q.append(eid)

    safe_tmp = []
    while q:
        eid = q.popleft()
        if not active[eid]:
            continue
        u = U[eid]
        v = V[eid]
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

    m_act = sum(1 for eid in range(len(U)) if active[eid])
    alpha = max(1, m_act // max(1, n_nodes))
    since = 0

    for eid in eids:
        if not active[eid]:
            continue
        active[eid] = 0
        inF[eid] = 1
        since += 1
        if since >= alpha:
            since = 0
            if _is_acyclic_active(n_nodes, out_adj, V, active):
                break

    if not _is_acyclic_active(n_nodes, out_adj, V, active):
        for eid in eids:
            if not active[eid]:
                continue
            active[eid] = 0
            inF[eid] = 1
            if _is_acyclic_active(n_nodes, out_adj, V, active):
                break

    for eid in safe_tmp:
        active[eid] = 1

    return inF


def wmsf_minimize_global(n_nodes, U, V, W0, active, out_adj, inF):
    for eid in range(len(inF)):
        if inF[eid]:
            active[eid] = 0

    _, rank = topo_order_active(n_nodes, out_adj, V, active)

    visited = [0] * n_nodes
    stamp = 0

    def reachable(src, target, rank_limit):
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
            for eid in out_adj[x]:
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

    cand = [eid for eid in range(len(inF)) if inF[eid]]
    cand.sort(key=lambda eid: (-W0[eid], U[eid], V[eid], eid))

    for eid in cand:
        u, v = U[eid], V[eid]
        if rank[u] < rank[v]:
            active[eid] = 1
            inF[eid] = 0
            continue
        if not reachable(v, u, rank_limit=rank[u]):
            active[eid] = 1
            inF[eid] = 0
            _, rank = topo_order_active(n_nodes, out_adj, V, active)

    return inF


def wmsf_seed_solution(n_nodes, U, V, W0, active, out_adj, in_adj, ordering="L2"):
    inF = wmsf_removeArcs_global(n_nodes, U, V, W0, active, out_adj, in_adj, ordering=ordering)
    inF = wmsf_minimize_global(n_nodes, U, V, W0, active, out_adj, inF)
    return inF, active


def wmsf_seed_solution_best_of_L1_L2(n, U, V, W0, active_init, out_adj, in_adj):
    total_w = sum(W0)

    active_L1 = bytearray(active_init)
    inF_L1, active_L1 = wmsf_seed_solution(n, U, V, W0, active_L1, out_adj, in_adj, ordering="L1")
    _, rank_L1 = topo_order_active(n, out_adj, V, active_L1)
    _, fw_L1, bw_L1 = compute_fw_bw_from_rank(U, V, W0, rank_L1)

    active_L2 = bytearray(active_init)
    inF_L2, active_L2 = wmsf_seed_solution(n, U, V, W0, active_L2, out_adj, in_adj, ordering="L2")
    _, rank_L2 = topo_order_active(n, out_adj, V, active_L2)
    _, fw_L2, bw_L2 = compute_fw_bw_from_rank(U, V, W0, rank_L2)

    if bw_L1 <= bw_L2:
        return inF_L1, active_L1, rank_L1, fw_L1, bw_L1, total_w, "L1"
    else:
        return inF_L2, active_L2, rank_L2, fw_L2, bw_L2, total_w, "L2"


# ============================================================
# 9) LR seed
# ============================================================

def find_any_cycle_eids_global(n_nodes, out_adj, V, active):
    state = bytearray(n_nodes)
    parent = [-1] * n_nodes
    parent_eid = [-1] * n_nodes
    next_ptr = [0] * n_nodes
    touched = []

    def reset():
        for x in touched:
            state[x] = 0
            parent[x] = -1
            parent_eid[x] = -1
            next_ptr[x] = 0
        touched.clear()

    for s in range(n_nodes):
        if state[s] != 0:
            continue
        stack = [s]
        state[s] = 1
        touched.append(s)

        while stack:
            u = stack[-1]
            i = next_ptr[u]
            out = out_adj[u]
            while i < len(out) and active[out[i]] == 0:
                i += 1
            next_ptr[u] = i

            if i >= len(out):
                state[u] = 2
                stack.pop()
                continue

            eid = out[i]
            v = V[eid]
            next_ptr[u] = i + 1

            if state[v] == 0:
                parent[v] = u
                parent_eid[v] = eid
                state[v] = 1
                touched.append(v)
                stack.append(v)
            elif state[v] == 1:
                cycle = [eid]
                cur = u
                while cur != v:
                    pe = parent_eid[cur]
                    if pe == -1:
                        break
                    cycle.append(pe)
                    cur = parent[cur]
                    if cur == -1:
                        break
                if cur == v:
                    cycle.reverse()
                    reset()
                    return cycle

    reset()
    return None


def lr_cycle_reduction_global(n_nodes, U, V, W0, W, active, out_adj, tol=1e-12):
    inF = bytearray(len(U))
    while True:
        cyc = find_any_cycle_eids_global(n_nodes, out_adj, V, active)
        if cyc is None:
            break

        eps = None
        for eid in cyc:
            ww = W[eid]
            if eps is None or ww < eps:
                eps = ww

        if eps is None or eps <= tol:
            e0 = cyc[0]
            if active[e0]:
                active[e0] = 0
                W[e0] = 0.0
                inF[e0] = 1
            continue

        for eid in cyc:
            new_w = W[eid] - eps
            W[eid] = new_w
            if new_w <= tol and active[eid]:
                active[eid] = 0
                W[eid] = 0.0
                inF[eid] = 1

    return inF, active, W


def lr_seed_solution(n_nodes, U, V, W0, W, active, out_adj, tol=1e-12):
    inF, active, W = lr_cycle_reduction_global(n_nodes, U, V, W0, W, active, out_adj, tol=tol)
    inF = wmsf_minimize_global(n_nodes, U, V, W0, active, out_adj, inF)
    return inF, active


# ============================================================
# 10) SCC scoring + LNS step
# ============================================================

def score_scc_backward_weight(edges_in_scc, rank) -> float:
    bw = 0.0
    for u, v, w, _eid in edges_in_scc:
        if rank[u] > rank[v]:
            bw += w
    return bw


def lns_step_on_scc_fast(
    scc_nodes,
    scc_edges,
    U, V, W0,
    active,
    out_adj,
    inF,
    destroy_addback_frac=0.25,
    destroy_remove_frac=0.02,
    tol=1e-12
):
    n_nodes = len(out_adj)

    allowed_nodes = bytearray(n_nodes)
    for x in scc_nodes:
        allowed_nodes[x] = 1

    allowed_eids = bytearray(len(U))
    internal_eids = []
    for (_u, _v, _w, eid) in scc_edges:
        allowed_eids[eid] = 1
        internal_eids.append(eid)

    delta = [(eid, 1 if active[eid] else 0, 1 if inF[eid] else 0) for eid in internal_eids]

    removed_in_scc = [eid for eid in internal_eids if inF[eid]]
    removed_in_scc.sort(key=lambda eid: (-W0[eid], U[eid], V[eid], eid))
    k_add = int(destroy_addback_frac * len(removed_in_scc))
    if k_add > 0:
        for eid in removed_in_scc[:k_add]:
            active[eid] = 1
            inF[eid] = 0

    active_in_scc = [eid for eid in internal_eids if active[eid]]
    active_in_scc.sort(key=lambda eid: (W0[eid], U[eid], V[eid], eid))
    k_rem = int(destroy_remove_frac * len(active_in_scc))
    if k_rem > 0:
        for eid in active_in_scc[:k_rem]:
            active[eid] = 0
            inF[eid] = 1

    F_add = local_ratio_repair_inside_scc(
        U=U, V=V, W0=W0,
        active=active,
        out_adj=out_adj,
        allowed_nodes=allowed_nodes,
        allowed_eids=allowed_eids,
        internal_eids=internal_eids,
        tol=tol
    )
    for eid in F_add:
        inF[eid] = 1

    try:
        minimize_addback_inside_scc(
            U=U, V=V, W0=W0,
            active=active,
            out_adj=out_adj,
            inF=inF,
            allowed_nodes=allowed_nodes,
            allowed_eids=allowed_eids,
            internal_eids=internal_eids
        )
    except RuntimeError:
        for eid, a0, f0 in delta:
            active[eid] = 1 if a0 else 0
            inF[eid] = 1 if f0 else 0
        return False, delta

    return True, delta


# ============================================================
# Load graph once per process
# ============================================================

def _ensure_loaded(dimacs_path: str, tol: float, logger: logging.Logger, force_reload: bool = False):
    if _G["loaded"] and (not force_reload) and (_G["dimacs_path"] == dimacs_path):
        return

    edges_indexed, node_to_index, index_to_node = read_graph_dimacs_agg(dimacs_path)
    n = len(node_to_index)
    U, V, W0, W_init, active_init, out_adj, in_adj = build_eid_graph_inout_with_W(edges_indexed, n, tol=tol)

    comps, comp_id = kosaraju_scc(n, edges_indexed)
    by_scc = edges_by_scc(edges_indexed, comp_id)
    scc_list = []
    for scc_idx, verts in enumerate(comps):
        e_list = by_scc.get(scc_idx, [])
        if len(verts) > 1 and e_list:
            scc_list.append((verts, e_list))

    _G.update({
        "loaded": True,
        "dimacs_path": dimacs_path,
        "edges_indexed": edges_indexed,
        "node_to_index": node_to_index,
        "index_to_node": index_to_node,
        "n": n,
        "U": U,
        "V": V,
        "W0": W0,
        "W_init": W_init,
        "active_init": active_init,
        "out_adj": out_adj,
        "in_adj": in_adj,
        "scc_list": scc_list,
    })
    logger.info(f"[load] n={n} m={len(edges_indexed)} SCCs={len(scc_list)}")


def preload_graph_once_if_fork(dimacs_path: str, tol: float, logger: logging.Logger, mp_start_method: str):
    if mp_start_method != "fork":
        return
    logger.info("[master] preloading graph once (fork optimization)...")
    _ensure_loaded(dimacs_path, tol=tol, logger=logger, force_reload=True)
    logger.info("[master] preload done.")


# ============================================================
# Worker run
# ============================================================

@dataclass
class RunResult:
    best_bw: float
    best_fw: float
    total_w: float
    best_csv_path: str
    worker_id: int
    seed: int
    elapsed_sec: float


def lns_run_single_worker_time_limited(
    dimacs_path: str,
    worker_csv_path: str,
    tol: float,
    destroy_addback_frac: float,
    destroy_remove_frac: float,
    rng_seed: int,
    worker_id: int,
    hard_deadline_ts: float,
    stall_limit: int,
) -> RunResult:
    logger = logging.getLogger(f"worker-{worker_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    q = _WORKER_LOG_QUEUE
    if q is not None:
        from logging.handlers import QueueHandler
        logger.addHandler(QueueHandler(q))
    else:
        logger.addHandler(logging.StreamHandler(stream=sys.stderr))

    random.seed(rng_seed)
    t0 = time.perf_counter()
    logger.info(f"[worker {worker_id}] start seed={rng_seed}")

    _ensure_loaded(dimacs_path, tol=tol, logger=logger)

    n = _G["n"]
    U = _G["U"]
    V = _G["V"]
    W0 = _G["W0"]
    W_init = _G["W_init"]
    active_init = _G["active_init"]
    out_adj = _G["out_adj"]
    in_adj = _G["in_adj"]
    scc_list = _G["scc_list"]
    index_to_node = _G["index_to_node"]

    stall_limit = max(stall_limit, min(20000, 20 * int((n ** 0.5) + 1)))

    inF_A, active_A, rank_A, fw_A, bw_A, total_w, best_wmsf_ord = wmsf_seed_solution_best_of_L1_L2(
        n, U, V, W0, active_init, out_adj, in_adj
    )

    active_B = bytearray(active_init)
    W_B = list(W_init)
    inF_B, active_B = lr_seed_solution(n, U, V, W0, W_B, active_B, out_adj, tol=tol)
    _, rank_B = topo_order_active(n, out_adj, V, active_B)
    _, fw_B, bw_B = compute_fw_bw_from_rank(U, V, W0, rank_B)

    if bw_A <= bw_B:
        best_bw = bw_A
        best_active = bytearray(active_A)
        best_inF = bytearray(inF_A)
        active = bytearray(active_A)
        inF = bytearray(inF_A)
        rank = list(rank_A)
        best_rank = list(rank_A)
        start_from = f"WMSF-{best_wmsf_ord}"
    else:
        best_bw = bw_B
        best_active = bytearray(active_B)
        best_inF = bytearray(inF_B)
        active = bytearray(active_B)
        inF = bytearray(inF_B)
        rank = list(rank_B)
        best_rank = list(rank_B)
        start_from = "LR"

    logger.info(f"[worker {worker_id}] base WMSF(best-of={best_wmsf_ord}) BW={bw_A:.6f} FW={fw_A:.6f} ratio={fw_A/total_w:.6f}")
    logger.info(f"[worker {worker_id}] base LR               BW={bw_B:.6f} FW={fw_B:.6f} ratio={fw_B/total_w:.6f}")
    logger.info(f"[worker {worker_id}] incumbent start_from={start_from} best_BW={best_bw:.6f} stall_limit={stall_limit}")

    it = 0
    no_improve = 0
    EPS_IMPROVE = 1e-12

    # print progress even if no improvements, every N iterations
    PROGRESS_EVERY = 200

    while True:
        now_wall = time.time()
        if now_wall >= hard_deadline_ts:
            logger.info(f"[worker {worker_id}] stop: time budget reached")
            break
        if no_improve >= stall_limit:
            logger.info(f"[worker {worker_id}] stop: stalled no_improve={no_improve}/{stall_limit}")
            break

        it += 1

        scored = []
        for verts, e_list in scc_list:
            bw_scc = score_scc_backward_weight(e_list, rank)
            if bw_scc > 0.0:
                scored.append((bw_scc, verts, e_list))
        if not scored:
            logger.info(f"[worker {worker_id}] stop: no SCC has backward weight (it={it})")
            break

        scored.sort(key=lambda x: -x[0])
        topK = min(25, max(8, min(len(scored), 20)))
        pool = scored[:topK]
        weights = [x[0] for x in pool]
        picked_bw_scc, verts, e_list = random.choices(pool, weights=weights, k=1)[0]

        ok, delta = lns_step_on_scc_fast(
            scc_nodes=verts,
            scc_edges=e_list,
            U=U, V=V, W0=W0,
            active=active,
            out_adj=out_adj,
            inF=inF,
            destroy_addback_frac=destroy_addback_frac,
            destroy_remove_frac=destroy_remove_frac,
            tol=tol
        )
        if not ok:
            no_improve += 1
            if it % PROGRESS_EVERY == 0:
                elapsed = time.perf_counter() - t0
                logger.info(f"[worker {worker_id}] progress it={it} best_BW={best_bw:.6f} no_improve={no_improve} t={elapsed:.1f}s")
            continue

        try:
            _, rank_new = topo_order_active(n, out_adj, V, active)
        except RuntimeError:
            for eid, a0, f0 in delta:
                active[eid] = 1 if a0 else 0
                inF[eid] = 1 if f0 else 0
            no_improve += 1
            if it % PROGRESS_EVERY == 0:
                elapsed = time.perf_counter() - t0
                logger.info(f"[worker {worker_id}] progress it={it} best_BW={best_bw:.6f} no_improve={no_improve} t={elapsed:.1f}s")
            continue

        _, fw_new, bw_new = compute_fw_bw_from_rank(U, V, W0, rank_new)

        if bw_new < best_bw - EPS_IMPROVE:
            best_bw = bw_new
            best_active = bytearray(active)
            best_inF = bytearray(inF)
            rank = list(rank_new)
            best_rank = list(rank_new)
            no_improve = 0
            elapsed = time.perf_counter() - t0
            logger.info(f"[worker {worker_id}] NEW BEST it={it} BW={bw_new:.6f} FW={fw_new:.6f} scc_bw={picked_bw_scc:.6f} t={elapsed:.1f}s")
        else:
            for eid, a0, f0 in delta:
                active[eid] = 1 if a0 else 0
                inF[eid] = 1 if f0 else 0
            no_improve += 1
            if it % PROGRESS_EVERY == 0:
                elapsed = time.perf_counter() - t0
                logger.info(f"[worker {worker_id}] progress it={it} best_BW={best_bw:.6f} no_improve={no_improve} t={elapsed:.1f}s")

    write_ranking_csv_atomic(index_to_node, best_rank, worker_csv_path)

    total_w2, best_fw, best_bw2 = compute_fw_bw_from_rank(U, V, W0, best_rank)
    elapsed = time.perf_counter() - t0
    removed_count = sum(1 for x in best_inF if x)

    logger.info(f"[worker {worker_id}] DONE BW={best_bw2:.6f} FW={best_fw:.6f} ratio={best_fw/total_w2:.6f} removed_edges={removed_count} elapsed={elapsed:.3f}s")
    logger.info(f"[worker {worker_id}] wrote_worker_csv={worker_csv_path}")

    return RunResult(
        best_bw=best_bw2,
        best_fw=best_fw,
        total_w=total_w2,
        best_csv_path=worker_csv_path,
        worker_id=worker_id,
        seed=rng_seed,
        elapsed_sec=elapsed
    )


def _worker_entry(args_tuple):
    return lns_run_single_worker_time_limited(*args_tuple)


# ============================================================
# Master runner
# ============================================================

def run_parallel_time_limited(
    dimacs_path: str,
    approach_tag: str,
    base_seed: int,
    destroy_addback_frac: float,
    destroy_remove_frac: float,
    tol: float,
    output_dir: Optional[str],
    mp_start_method: str,
    time_budget_seconds: int,
    safety_margin_seconds: int = 30,
):
    if not os.path.isfile(dimacs_path):
        raise FileNotFoundError(f"Input graph not found: {dimacs_path}")

    out_dir = os.path.abspath(output_dir) if output_dir else os.path.dirname(os.path.abspath(dimacs_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(dimacs_path))[0]
    final_csv = os.path.join(out_dir, f"{base}__{approach_tag}__best.csv")
    log_path = os.path.join(out_dir, f"{base}__{approach_tag}.log")

    root_logger = setup_root_logger_file_and_screen(log_path)
    root_logger.info(f"[master] input={dimacs_path}")
    root_logger.info(f"[master] out_dir={out_dir}")
    root_logger.info(f"[master] final_csv={final_csv}")
    root_logger.info(f"[master] log_file={log_path}")

    workers = get_num_workers()
    hard_deadline_ts = time.time() + max(1, time_budget_seconds - safety_margin_seconds)

    root_logger.info(f"[master] workers={workers} mp_start={mp_start_method}")
    root_logger.info(f"[master] time_budget_seconds={time_budget_seconds} safety_margin_seconds={safety_margin_seconds}")
    root_logger.info(f"[master] params addfrac={destroy_addback_frac} remfrac={destroy_remove_frac} tol={tol} base_seed={base_seed}")

    preload_graph_once_if_fork(dimacs_path, tol=tol, logger=root_logger, mp_start_method=mp_start_method)

    worker_csv_paths = [
        os.path.join(out_dir, f"{base}__{approach_tag}__worker{wid}.csv")
        for wid in range(workers)
    ]

    base_stall_limit = 2000
    t0 = time.perf_counter()
    results: List[RunResult] = []

    if workers <= 1:
        res = lns_run_single_worker_time_limited(
            dimacs_path=dimacs_path,
            worker_csv_path=worker_csv_paths[0],
            tol=tol,
            destroy_addback_frac=destroy_addback_frac,
            destroy_remove_frac=destroy_remove_frac,
            rng_seed=base_seed,
            worker_id=0,
            hard_deadline_ts=hard_deadline_ts,
            stall_limit=base_stall_limit,
        )
        results.append(res)
    else:
        ctx = get_context(mp_start_method)

        # IMPORTANT FIX:
        # - use ctx.Queue() for fork
        # - for spawn, use ctx.Manager().Queue() (proxy is picklable)
        if mp_start_method == "spawn":
            from multiprocessing import Manager
            manager = Manager()
            log_queue = manager.Queue()
        else:
            manager = None
            log_queue = ctx.Queue(-1)

        from logging.handlers import QueueListener
        listener = QueueListener(log_queue, *root_logger.handlers, respect_handler_level=True)
        listener.start()

        task_args = []
        for wid in range(workers):
            seed = base_seed + wid * 1337
            task_args.append((
                dimacs_path,
                worker_csv_paths[wid],
                tol,
                destroy_addback_frac,
                destroy_remove_frac,
                seed,
                wid,
                hard_deadline_ts,
                base_stall_limit,
            ))

        try:
            with ctx.Pool(
                processes=workers,
                initializer=_init_worker_logging,
                initargs=(log_queue,),
            ) as pool:
                for res in pool.imap_unordered(_worker_entry, task_args, chunksize=1):
                    results.append(res)
                    root_logger.info(f"[master] worker_done wid={res.worker_id} seed={res.seed} BW={res.best_bw:.6f} elapsed={res.elapsed_sec:.1f}s")
        finally:
            listener.stop()
            if manager is not None:
                manager.shutdown()

    results.sort(key=lambda r: (r.best_bw, -r.best_fw))
    best = results[0]

    shutil.copyfile(best.best_csv_path, final_csv)

    elapsed = time.perf_counter() - t0
    root_logger.info(f"[master] BEST wid={best.worker_id} seed={best.seed} BW={best.best_bw:.6f} FW={best.best_fw:.6f}")
    root_logger.info(f"[master] wrote_final={final_csv}")
    root_logger.info(f"[master] total_elapsed={elapsed:.3f}s")

    print(f"✅ Final ranking written: {final_csv}", flush=True)
    print(f"✅ FINAL Backward Weight: {best.best_bw:.6f}", flush=True)

    return final_csv, log_path


# ============================================================
# MAIN (set inputs here)
# ============================================================

if __name__ == "__main__":
    INPUT_D_PATH = "/mmfs1/home/sv96/Feedback-arc-set-paper/datasets/connectome.d"

    # Keep this short; it becomes part of filenames
    APPROACH_TAG = "parallel_scc_lns"

    BASE_SEED = 1
    DESTROY_ADDBACK_FRAC = 0.35
    DESTROY_REMOVE_FRAC = 0.03
    TOL = 1e-12

    OUTPUT_DIR = None  # None -> same dir as input
    MP_START_METHOD = default_mp_start_method()
    TIME_BUDGET_SECONDS = get_time_budget_seconds(default_seconds=3600)

    run_parallel_time_limited(
        dimacs_path=INPUT_D_PATH,
        approach_tag=APPROACH_TAG,
        base_seed=BASE_SEED,
        destroy_addback_frac=DESTROY_ADDBACK_FRAC,
        destroy_remove_frac=DESTROY_REMOVE_FRAC,
        tol=TOL,
        output_dir=OUTPUT_DIR,
        mp_start_method=MP_START_METHOD,
        time_budget_seconds=TIME_BUDGET_SECONDS,
        safety_margin_seconds=30,
    )
