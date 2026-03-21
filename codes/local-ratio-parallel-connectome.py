#!/usr/bin/env python3
# coding: utf-8

"""
Local-ratio cycle reduction for weighted FAS (paper-style) with:
  - DIMACS aggregation
  - parallel cycle finding (shared-memory)
  - edge-disjoint batching + parallel apply
  - directed cycle-core trimming
  - periodic SCC focusing (largest cyclic SCC)
  - strong logging for bottlenecks, overlap, and yield
  - OPTIONAL add-back (heavy edges first) if acyclic before time runs out
  - FINAL ranking CSV is ALWAYS written (topo if acyclic; SCC-condensation otherwise)

Output:
  1) Ranking CSV: columns ["Node ID", "Order"]
  2) Log file with detailed progress

Both output filenames include the program start timestamp.
"""

import os
import sys
import time
import math
import heapq
import random
import datetime
from collections import defaultdict, deque
import multiprocessing as mp

import numpy as np
import pandas as pd
from multiprocessing import shared_memory


# ============================================================
#                 USER SETTINGS (EDIT HERE)
# ============================================================

EDGE_FILE = "/mmfs1/home/sv96/Feedback-arc-set-paper/datasets/connectome.d"  # <-- hard-coded
BASE_DIR = os.path.dirname(EDGE_FILE)
BASE_NAME = os.path.splitext(os.path.basename(EDGE_FILE))[0]

TOL = 1e-12

# Run time
MAX_SECONDS = 72 * 3600  # 72 hours
HEARTBEAT_SEC = 60.0

# Parallelism
DEFAULT_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 1)))
WORKERS = max(1, DEFAULT_CPUS)

# Cycle search / batching knobs
SAMPLE_STARTS_TOTAL = 256              # total starts per "find" iteration (split across workers)
CYCLES_PER_WORKER_TARGET = 8           # ask each worker to return up to this many cycles
BATCH_DISJOINT_TARGET = 32             # how many edge-disjoint cycles to apply per iteration
MAX_DFS_EDGE_STEPS = 250_000           # per DFS attempt budget
MAX_CYCLE_LEN_STORE = 10_000           # cap for storing/reporting cycle; longer cycles are truncated

# Learning / hot-start
HOT_NODE_BUFFER = 200_000              # store recent cycle nodes
HOT_START_FRACTION = 0.70              # portion of starts sampled from hot nodes (if available)

# Directed cycle-core trimming
DO_CORE_TRIM = True
TRIM_PROCESS_LIMIT_PER_ITER = 2_000_000  # safety cap on trimming work per iteration

# SCC focusing (periodic)
DO_SCC_FOCUS = True
SCC_RECOMPUTE_EVERY_SEC = 15 * 60.0    # recompute SCC focus every 15 minutes
SCC_RECOMPUTE_EVERY_DEACT = 200_000    # or after this many deactivations since last SCC

# Optional: final add-back (heavy edges first) if acyclic before time runs out
DO_ADD_BACK = True
ADD_BACK_MAX_FRACTION_REMAINING = 0.15  # spend up to 15% of remaining time on add-back
ADD_BACK_RECOMPUTE_TOPO_EVERY = 5000    # recompute topo ranks periodically during add-back


# ============================================================
#                        LOGGING
# ============================================================

def now_ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_line(msg: str, log_path: str):
    line = f"[{now_ts()}] {msg}"
    with open(log_path, "a", buffering=1) as f:
        f.write(line + "\n")

def fmt_int(x):
    return f"{int(x):,}"

def fmt_float(x, d=3):
    return f"{float(x):.{d}f}"

def human_time(sec):
    sec = float(sec)
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec/60:.2f}m"
    return f"{sec/3600:.2f}h"


# ============================================================
# 1) DIMACS reader (aggregates parallel arcs) -> deterministic
# ============================================================

def read_graph_dimacs_agg(file_path):
    """
    Reads DIMACS-like lines:
      a <source> <target> <weight> <extra...>

    Aggregates parallel arcs (u,v): summed weights.
    Deterministic node mapping: sorted node IDs.
    Deterministic edges: sorted by (u_idx, v_idx).
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
# 2) CSR graph build (out + in), plus shared memory setup
# ============================================================

def build_csr_from_edges(edges_indexed, n_nodes):
    """
    edges_indexed is already sorted by (u,v), so CSR is deterministic.
    Returns:
      U,V,W0 arrays (m)
      out_offsets (n+1), out_eids (m)
      in_offsets (n+1),  in_eids  (m)
    """
    m = len(edges_indexed)

    U = np.empty(m, dtype=np.int32)
    V = np.empty(m, dtype=np.int32)
    W0 = np.empty(m, dtype=np.float64)

    for eid, (u, v, w) in enumerate(edges_indexed):
        U[eid] = u
        V[eid] = v
        W0[eid] = w

    # Out CSR
    out_deg = np.zeros(n_nodes, dtype=np.int32)
    for eid in range(m):
        out_deg[U[eid]] += 1
    out_offsets = np.empty(n_nodes + 1, dtype=np.int64)
    out_offsets[0] = 0
    np.cumsum(out_deg, out=out_offsets[1:])
    out_eids = np.empty(m, dtype=np.int32)
    out_cursor = out_offsets[:-1].copy()
    for eid in range(m):
        u = U[eid]
        pos = out_cursor[u]
        out_eids[pos] = eid
        out_cursor[u] += 1

    # In CSR
    in_deg = np.zeros(n_nodes, dtype=np.int32)
    for eid in range(m):
        in_deg[V[eid]] += 1
    in_offsets = np.empty(n_nodes + 1, dtype=np.int64)
    in_offsets[0] = 0
    np.cumsum(in_deg, out=in_offsets[1:])
    in_eids = np.empty(m, dtype=np.int32)
    in_cursor = in_offsets[:-1].copy()
    for eid in range(m):
        v = V[eid]
        pos = in_cursor[v]
        in_eids[pos] = eid
        in_cursor[v] += 1

    return U, V, W0, out_offsets, out_eids, in_offsets, in_eids


def shm_create_from_np(arr: np.ndarray, name_prefix: str):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=None)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr
    info = {
        "name": shm.name,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "nbytes": arr.nbytes,
        "prefix": name_prefix,
    }
    return shm, info


def shm_attach_np(info: dict):
    shm = shared_memory.SharedMemory(name=info["name"])
    arr = np.ndarray(info["shape"], dtype=np.dtype(info["dtype"]), buffer=shm.buf)
    return shm, arr


# ============================================================
# 3) Directed cycle-core trimming
# ============================================================

def init_core_trim(n, U, V, active, out_offsets, out_eids, in_offsets, in_eids):
    core = np.ones(n, dtype=np.uint8)
    indeg = np.zeros(n, dtype=np.int32)
    outdeg = np.zeros(n, dtype=np.int32)

    m = active.shape[0]
    for eid in range(m):
        if active[eid]:
            outdeg[U[eid]] += 1
            indeg[V[eid]] += 1

    q = deque()
    for u in range(n):
        if indeg[u] == 0 or outdeg[u] == 0:
            q.append(u)

    trimmed = 0
    work_steps = 0

    while q:
        u = q.popleft()
        if core[u] == 0:
            continue
        if indeg[u] > 0 and outdeg[u] > 0:
            continue
        core[u] = 0
        trimmed += 1

        # outgoing u->v: indeg[v]--
        a, b = out_offsets[u], out_offsets[u+1]
        for pos in range(a, b):
            eid = out_eids[pos]
            if not active[eid]:
                continue
            v = V[eid]
            if core[v]:
                indeg[v] -= 1
                work_steps += 1
                if indeg[v] == 0:
                    q.append(v)

        # incoming x->u: outdeg[x]--
        a, b = in_offsets[u], in_offsets[u+1]
        for pos in range(a, b):
            eid = in_eids[pos]
            if not active[eid]:
                continue
            x = U[eid]
            if core[x]:
                outdeg[x] -= 1
                work_steps += 1
                if outdeg[x] == 0:
                    q.append(x)

    return core, indeg, outdeg, trimmed, work_steps


def core_trim_incremental(queue, core, indeg, outdeg, U, V, active,
                          out_offsets, out_eids, in_offsets, in_eids, work_cap: int):
    trimmed = 0
    work_steps = 0

    while queue and work_steps < work_cap:
        u = queue.popleft()
        if core[u] == 0:
            continue
        if indeg[u] > 0 and outdeg[u] > 0:
            continue

        core[u] = 0
        trimmed += 1

        # outgoing u->v: indeg[v]--
        a, b = out_offsets[u], out_offsets[u+1]
        for pos in range(a, b):
            eid = out_eids[pos]
            if not active[eid]:
                continue
            v = V[eid]
            if core[v]:
                indeg[v] -= 1
                work_steps += 1
                if indeg[v] == 0:
                    queue.append(v)
                if work_steps >= work_cap:
                    break
        if work_steps >= work_cap:
            break

        # incoming x->u: outdeg[x]--
        a, b = in_offsets[u], in_offsets[u+1]
        for pos in range(a, b):
            eid = in_eids[pos]
            if not active[eid]:
                continue
            x = U[eid]
            if core[x]:
                outdeg[x] -= 1
                work_steps += 1
                if outdeg[x] == 0:
                    queue.append(x)
                if work_steps >= work_cap:
                    break

    return trimmed, work_steps


# ============================================================
# 4) SCC focusing (Kosaraju) on current ACTIVE graph + core mask
# ============================================================

def kosaraju_scc_active_core(n, U, V, active, core, out_offsets, out_eids, in_offsets, in_eids):
    comp_id = np.full(n, -1, dtype=np.int32)

    # First pass: finish order on forward graph
    order = []
    state = np.zeros(n, dtype=np.uint8)  # 0 unvisited, 1 in stack, 2 done

    for s in range(n):
        if not core[s]:
            continue
        if state[s] != 0:
            continue

        stack = [(s, out_offsets[s])]
        state[s] = 1
        while stack:
            u, it = stack[-1]
            end = out_offsets[u+1]

            while it < end:
                eid = out_eids[it]
                it += 1
                if not active[eid]:
                    continue
                v = V[eid]
                if not core[v]:
                    continue
                if state[v] == 0:
                    stack[-1] = (u, it)
                    stack.append((v, out_offsets[v]))
                    state[v] = 1
                    break
            else:
                stack.pop()
                state[u] = 2
                order.append(u)

    # Second pass on reversed graph
    comp_sizes = []
    cur_comp = 0

    for s in reversed(order):
        if state[s] != 2:
            continue
        if comp_id[s] != -1:
            continue

        stack = [s]
        comp_id[s] = cur_comp
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            a, b = in_offsets[u], in_offsets[u+1]
            for pos in range(a, b):
                eid = in_eids[pos]
                if not active[eid]:
                    continue
                x = U[eid]
                if not core[x]:
                    continue
                if comp_id[x] == -1:
                    comp_id[x] = cur_comp
                    stack.append(x)

        comp_sizes.append(size)
        cur_comp += 1

    num_comps = cur_comp

    cyclic = set()
    for cid, sz in enumerate(comp_sizes):
        if sz > 1:
            cyclic.add(cid)

    # detect self-loops in singleton SCCs
    for u in range(n):
        if not core[u]:
            continue
        cid = comp_id[u]
        if cid < 0:
            continue
        if comp_sizes[cid] != 1:
            continue
        a, b = out_offsets[u], out_offsets[u+1]
        for pos in range(a, b):
            eid = out_eids[pos]
            if not active[eid]:
                continue
            v = V[eid]
            if v == u and core[v]:
                cyclic.add(cid)
                break

    focus_comp = -1
    best_sz = -1
    if cyclic:
        for cid in cyclic:
            sz = comp_sizes[cid]
            if sz > best_sz:
                best_sz = sz
                focus_comp = cid

    if focus_comp >= 0:
        focus_nodes = np.where(comp_id == focus_comp)[0].astype(np.int32)
    else:
        focus_nodes = np.array([], dtype=np.int32)

    return comp_id, comp_sizes, num_comps, cyclic, focus_comp, focus_nodes


# ============================================================
# 5) Worker code: cycle finding (DFS) and apply reductions
# ============================================================

G = {}

def worker_init(shm_infos):
    G["shms"] = []
    for key, info in shm_infos.items():
        shm, arr = shm_attach_np(info)
        G["shms"].append(shm)
        G[key] = arr

    n = int(G["n_nodes"][0])
    G["state"] = np.zeros(n, dtype=np.uint8)
    G["parent"] = np.full(n, -1, dtype=np.int32)
    G["parent_eid"] = np.full(n, -1, dtype=np.int32)
    G["next_ptr"] = np.zeros(n, dtype=np.int64)
    G["stack_nodes"] = np.empty(n, dtype=np.int32)
    G["visited_list"] = np.empty(n, dtype=np.int32)

def worker_shutdown():
    for shm in G.get("shms", []):
        try:
            shm.close()
        except Exception:
            pass

def _dfs_find_cycle_from_start(start_node, max_edge_steps):
    U = G["U"]; V = G["V"]
    active = G["active"]; core = G["core"]
    out_offsets = G["out_offsets"]; out_eids = G["out_eids"]

    state = G["state"]
    parent = G["parent"]
    parent_eid = G["parent_eid"]
    next_ptr = G["next_ptr"]
    stack_buf = G["stack_nodes"]
    visited_buf = G["visited_list"]
    visited_len = 0

    def mark_visited(x):
        nonlocal visited_len
        visited_buf[visited_len] = x
        visited_len += 1

    def reset_touched():
        nonlocal visited_len
        for i in range(visited_len):
            x = visited_buf[i]
            state[x] = 0
            parent[x] = -1
            parent_eid[x] = -1
            next_ptr[x] = 0
        visited_len = 0

    if not core[start_node]:
        return None, 0

    top = 0
    stack_buf[top] = start_node
    top += 1
    state[start_node] = 1
    mark_visited(start_node)
    steps = 0

    while top > 0 and steps < max_edge_steps:
        u = int(stack_buf[top - 1])

        it = int(next_ptr[u])
        a = int(out_offsets[u])
        b = int(out_offsets[u + 1])

        while (a + it) < b and steps < max_edge_steps:
            eid = int(out_eids[a + it])
            it += 1
            steps += 1

            if not active[eid]:
                continue
            v = int(V[eid])
            if not core[v]:
                continue

            sv = int(state[v])
            if sv == 0:
                parent[v] = u
                parent_eid[v] = eid
                state[v] = 1
                mark_visited(v)
                next_ptr[u] = it
                stack_buf[top] = v
                top += 1
                break
            elif sv == 1:
                cycle = [eid]
                cur = u
                while cur != v and cur != -1:
                    pe = int(parent_eid[cur])
                    if pe == -1:
                        break
                    cycle.append(pe)
                    cur = int(parent[cur])

                if cur == v:
                    cycle.reverse()
                    reset_touched()
                    return cycle, steps
                continue

        else:
            next_ptr[u] = it
            state[u] = 2
            top -= 1

    reset_touched()
    return None, steps


def worker_find_cycles(starts_and_limits):
    starts_list, max_edge_steps, cycles_target, max_cycle_len_store = starts_and_limits

    found_cycles = []
    attempts = 0
    found = 0
    steps_tot = 0
    long_trunc = 0

    for s in starts_list:
        attempts += 1
        cyc, steps = _dfs_find_cycle_from_start(int(s), int(max_edge_steps))
        steps_tot += steps
        if cyc is not None:
            found += 1
            if len(cyc) > max_cycle_len_store:
                cyc = cyc[:max_cycle_len_store]
                long_trunc += 1
            found_cycles.append(cyc)
            if len(found_cycles) >= cycles_target:
                break

    return {
        "cycles": found_cycles,
        "attempts": attempts,
        "found": found,
        "steps_tot": steps_tot,
        "trunc": long_trunc
    }


def worker_apply_disjoint_cycles(payload):
    cycles = payload
    W = G["W"]
    active = G["active"]

    deactivated = []
    cycles_applied = 0
    edges_touched = 0

    eps_min = None
    eps_sum = 0.0
    eps_max = None

    for cyc in cycles:
        if not cyc:
            continue

        eps = None
        for eid in cyc:
            if not active[eid]:
                continue
            w = float(W[eid])
            if eps is None or w < eps:
                eps = w

        if eps is None or eps <= TOL:
            eid_pick = None
            for eid in cyc:
                if active[eid]:
                    if eid_pick is None or eid < eid_pick:
                        eid_pick = eid
            if eid_pick is not None:
                active[eid_pick] = 0
                W[eid_pick] = 0.0
                deactivated.append(int(eid_pick))
            cycles_applied += 1
            edges_touched += len(cyc)
            continue

        for eid in cyc:
            if not active[eid]:
                continue
            new_w = float(W[eid]) - eps
            if new_w <= TOL:
                active[eid] = 0
                W[eid] = 0.0
                deactivated.append(int(eid))
            else:
                W[eid] = new_w

        cycles_applied += 1
        edges_touched += len(cyc)

        eps_sum += eps
        eps_min = eps if eps_min is None else min(eps_min, eps)
        eps_max = eps if eps_max is None else max(eps_max, eps)

    eps_avg = (eps_sum / cycles_applied) if cycles_applied > 0 else 0.0
    return {
        "deactivated": deactivated,
        "cycles_applied": cycles_applied,
        "edges_touched": edges_touched,
        "eps_min": 0.0 if eps_min is None else float(eps_min),
        "eps_avg": float(eps_avg),
        "eps_max": 0.0 if eps_max is None else float(eps_max),
    }


# ============================================================
# 6) Coordinator helpers: start selection + disjoint selection
# ============================================================

def sample_starts(rng, core_nodes, focus_nodes, hot_nodes_deque, total_starts):
    starts = []

    base_pool = focus_nodes if (focus_nodes is not None and len(focus_nodes) > 0) else core_nodes
    if base_pool is None or len(base_pool) == 0:
        return starts

    hot_available = len(hot_nodes_deque) > 0
    hot_take = int(total_starts * HOT_START_FRACTION) if hot_available else 0
    hot_take = min(hot_take, total_starts)

    if hot_take > 0:
        hot_list = list(hot_nodes_deque)
        for _ in range(hot_take):
            starts.append(int(hot_list[rng.randrange(0, len(hot_list))]))

    remain = total_starts - len(starts)
    for _ in range(remain):
        starts.append(int(base_pool[rng.randrange(0, len(base_pool))]))

    rng.shuffle(starts)
    return starts


def greedy_select_edge_disjoint(cycles, target_k, scan_all=False):
    """
    Greedy pick up to target_k edge-disjoint cycles.

    If scan_all=True, we scan all cycles to compute a more meaningful overlap count
    against the final used-set (helpful for logging "overlap pressure").
    """
    used = set()
    selected = []
    rejected_overlap_seen = 0

    cycles_sorted = sorted(cycles, key=len)

    for cyc in cycles_sorted:
        ok = True
        for eid in cyc:
            if eid in used:
                ok = False
                break
        if not ok:
            rejected_overlap_seen += 1
            continue
        selected.append(cyc)
        for eid in cyc:
            used.add(eid)
        if len(selected) >= target_k and not scan_all:
            break

    # If scan_all, compute how many candidate cycles overlap the FINAL used-set
    overlap_against_final = None
    if scan_all:
        overlap = 0
        for cyc in cycles_sorted:
            for eid in cyc:
                if eid in used:
                    overlap += 1
                    break
        overlap_against_final = overlap

    stats = {
        "candidates": len(cycles),
        "selected": len(selected),
        "rejected_overlap_seen": rejected_overlap_seen,
        "overlap_against_final": overlap_against_final,
    }
    return selected, stats


# ============================================================
# 7) Ranking helpers
# ============================================================

def topo_order_active(n, out_offsets, out_eids, V, active):
    indeg = np.zeros(n, dtype=np.int32)
    for u in range(n):
        a, b = out_offsets[u], out_offsets[u+1]
        for pos in range(a, b):
            eid = out_eids[pos]
            if active[eid]:
                indeg[V[eid]] += 1

    heap = []
    for i in range(n):
        if indeg[i] == 0:
            heapq.heappush(heap, i)

    order = []
    while heap:
        u = heapq.heappop(heap)
        order.append(u)
        a, b = out_offsets[u], out_offsets[u+1]
        for pos in range(a, b):
            eid = out_eids[pos]
            if not active[eid]:
                continue
            v = V[eid]
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, int(v))

    if len(order) != n:
        raise RuntimeError("Topological sort failed: still cyclic.")

    rank = np.empty(n, dtype=np.int32)
    for r, node in enumerate(order):
        rank[node] = r
    return rank


def scc_condensation_ranking(n, U, V, active, out_offsets, out_eids, in_offsets, in_eids):
    core_all = np.ones(n, dtype=np.uint8)
    comp_id, comp_sizes, num_comps, cyclic, focus_comp, focus_nodes = kosaraju_scc_active_core(
        n, U, V, active, core_all, out_offsets, out_eids, in_offsets, in_eids
    )
    C = num_comps
    indeg_c = np.zeros(C, dtype=np.int32)
    adj_c = [set() for _ in range(C)]

    m = active.shape[0]
    for eid in range(m):
        if not active[eid]:
            continue
        cu = int(comp_id[U[eid]])
        cv = int(comp_id[V[eid]])
        if cu != cv:
            if cv not in adj_c[cu]:
                adj_c[cu].add(cv)
                indeg_c[cv] += 1

    heap = []
    for c in range(C):
        if indeg_c[c] == 0:
            heapq.heappush(heap, c)

    comp_order = []
    while heap:
        c = heapq.heappop(heap)
        comp_order.append(c)
        for d in adj_c[c]:
            indeg_c[d] -= 1
            if indeg_c[d] == 0:
                heapq.heappush(heap, d)

    comp_pos = {c: i for i, c in enumerate(comp_order)}
    nodes = list(range(n))
    nodes.sort(key=lambda u: (comp_pos[int(comp_id[u])], u))

    rank = np.empty(n, dtype=np.int32)
    for r, u in enumerate(nodes):
        rank[u] = r
    return rank, num_comps, len(cyclic)


# ============================================================
# 8) Optional add-back (heavy edges first) if acyclic
# ============================================================

def reachability_pruned(src, target, rank, rank_limit, out_offsets, out_eids, V, active):
    if src == target:
        return True
    if rank[src] > rank_limit:
        return False

    n = rank.shape[0]
    seen = np.zeros(n, dtype=np.uint8)
    stack = [int(src)]
    seen[src] = 1

    while stack:
        x = stack.pop()
        a, b = out_offsets[x], out_offsets[x+1]
        for pos in range(a, b):
            eid = out_eids[pos]
            if not active[eid]:
                continue
            y = int(V[eid])
            if rank[y] > rank_limit:
                continue
            if y == target:
                return True
            if not seen[y]:
                seen[y] = 1
                stack.append(y)
    return False


def add_back_heavy_first(removed_mask, U, V, W0, active,
                         out_offsets, out_eids, n, time_deadline, log_path):
    """
    Add-back: reactivate removed edges in descending original weight, keeping acyclic.
    removed_mask: np.uint8 length m, 1 if edge was removed during phase1 at any time.
    """
    cand = np.where((removed_mask == 1) & (active == 0))[0]
    log_line(f"[ADD_BACK] start candidates={fmt_int(cand.size)}", log_path)
    if cand.size == 0:
        log_line("[ADD_BACK] nothing to do", log_path)
        return 0

    # sort by -W0, then (u,v) for determinism
    # We do a stable lex sort: primary -W0, secondary U, tertiary V.
    # lexsort uses last key as primary, so reverse order carefully.
    # We'll approximate determinism by sorting indices with mergesort on weight then tie-break in Python for top chunk if needed.
    # For full determinism, use np.lexsort with float key via argsort then tie keys; good enough here:
    order = np.argsort(-W0[cand], kind="mergesort")
    cand = cand[order]

    rank = topo_order_active(n, out_offsets, out_eids, V, active)

    accepted = 0
    tried = 0
    since_retopo = 0

    for eid in cand:
        if time.perf_counter() >= time_deadline:
            break
        tried += 1
        u = int(U[eid]); v = int(V[eid])

        if rank[u] < rank[v]:
            active[eid] = 1
            accepted += 1
            since_retopo += 1
            if since_retopo >= ADD_BACK_RECOMPUTE_TOPO_EVERY:
                rank = topo_order_active(n, out_offsets, out_eids, V, active)
                since_retopo = 0
            continue

        if not reachability_pruned(v, u, rank, rank_limit=rank[u],
                                  out_offsets=out_offsets, out_eids=out_eids, V=V, active=active):
            active[eid] = 1
            accepted += 1
            rank = topo_order_active(n, out_offsets, out_eids, V, active)
            since_retopo = 0

        if tried % 50_000 == 0:
            log_line(f"[ADD_BACK] tried={fmt_int(tried)} accepted={fmt_int(accepted)}", log_path)

    log_line(f"[ADD_BACK] done tried={fmt_int(tried)} accepted={fmt_int(accepted)}", log_path)
    return accepted


# ============================================================
# 9) Main coordinator
# ============================================================

def main():
    start_wall = time.perf_counter()
    start_dt = datetime.datetime.now()
    start_tag = start_dt.strftime("%Y%m%d_%H%M%S")

    out_csv = os.path.join(BASE_DIR, f"{BASE_NAME}_paper_fas_ranking_{start_tag}.csv")
    log_path = os.path.join(BASE_DIR, f"{BASE_NAME}_paper_fas_log_{start_tag}.txt")

    open(log_path, "w").close()

    log_line(f"[START] edge_file={EDGE_FILE}", log_path)
    log_line(f"[START] out_csv={out_csv}", log_path)
    log_line(f"[START] log_path={log_path}", log_path)
    log_line(f"[KNOBS] tol={TOL} max_seconds={MAX_SECONDS} heartbeat_sec={HEARTBEAT_SEC}", log_path)
    log_line(f"[KNOBS] workers={WORKERS} sample_starts_total={SAMPLE_STARTS_TOTAL} cycles_per_worker={CYCLES_PER_WORKER_TARGET}", log_path)
    log_line(f"[KNOBS] batch_disjoint_target={BATCH_DISJOINT_TARGET} max_dfs_edge_steps={MAX_DFS_EDGE_STEPS}", log_path)
    log_line(f"[KNOBS] core_trim={DO_CORE_TRIM} trim_cap_per_iter={TRIM_PROCESS_LIMIT_PER_ITER}", log_path)
    log_line(f"[KNOBS] scc_focus={DO_SCC_FOCUS} scc_every_sec={SCC_RECOMPUTE_EVERY_SEC} scc_every_deact={SCC_RECOMPUTE_EVERY_DEACT}", log_path)
    log_line(f"[KNOBS] hot_start_fraction={HOT_START_FRACTION} hot_buffer={HOT_NODE_BUFFER}", log_path)
    log_line(f"[KNOBS] add_back={DO_ADD_BACK} add_back_fraction_remaining={ADD_BACK_MAX_FRACTION_REMAINING}", log_path)

    rng = random.Random(12345)

    # -------------------- READ --------------------
    t_read0 = time.perf_counter()
    edges_indexed, node_to_index, index_to_node = read_graph_dimacs_agg(EDGE_FILE)
    n = len(node_to_index)
    m = len(edges_indexed)
    total_w = sum(w for _, _, w in edges_indexed)
    t_read1 = time.perf_counter()
    log_line(f"[READ] done in {human_time(t_read1 - t_read0)} n={fmt_int(n)} m={fmt_int(m)} total_w={total_w:.6f}", log_path)

    # -------------------- BUILD CSR --------------------
    t_build0 = time.perf_counter()
    U, V, W0, out_offsets, out_eids, in_offsets, in_eids = build_csr_from_edges(edges_indexed, n)
    t_build1 = time.perf_counter()
    log_line(f"[BUILD] csr built in {human_time(t_build1 - t_build0)}", log_path)

    W = W0.copy()
    active = np.ones(m, dtype=np.uint8)
    n_nodes_arr = np.array([n], dtype=np.int64)

    # Removed tracking for add-back (memory-safe)
    removed_mask = np.zeros(m, dtype=np.uint8)

    active_edges = int(m)
    removed_total = 0
    cycles_total = 0

    # -------------------- CORE TRIM INIT --------------------
    core = np.ones(n, dtype=np.uint8)
    indeg_core = np.zeros(n, dtype=np.int32)
    outdeg_core = np.zeros(n, dtype=np.int32)
    trim_queue = deque()

    if DO_CORE_TRIM:
        t_trim0 = time.perf_counter()
        core, indeg_core, outdeg_core, trimmed_init, work_init = init_core_trim(
            n, U, V, active, out_offsets, out_eids, in_offsets, in_eids
        )
        t_trim1 = time.perf_counter()
        log_line(f"[CORE] init_trim trimmed={fmt_int(trimmed_init)} core_nodes={fmt_int(int(core.sum()))} "
                 f"work_steps={fmt_int(work_init)} time={human_time(t_trim1-t_trim0)}", log_path)

    # -------------------- SCC FOCUS INIT --------------------
    focus_nodes = np.array([], dtype=np.int32)
    last_scc_time = -1e18
    last_scc_deact = 0

    if DO_SCC_FOCUS:
        t_scc0 = time.perf_counter()
        comp_id, comp_sizes, num_comps, cyclic, focus_comp, focus_nodes = kosaraju_scc_active_core(
            n, U, V, active, core, out_offsets, out_eids, in_offsets, in_eids
        )
        t_scc1 = time.perf_counter()
        last_scc_time = time.perf_counter()
        last_scc_deact = removed_total
        log_line(f"[SCC] reason=init comps={fmt_int(num_comps)} cyclic_comps={fmt_int(len(cyclic))} "
                 f"focus_comp={focus_comp} focus_nodes={fmt_int(len(focus_nodes))} time={human_time(t_scc1-t_scc0)}",
                 log_path)

    hot_nodes = deque(maxlen=HOT_NODE_BUFFER)

    # -------------------- SHARED MEMORY SETUP --------------------
    shm_blocks = []
    shm_infos = {}

    def add_shm(key, arr, prefix):
        shm, info = shm_create_from_np(arr, prefix)
        shm_blocks.append(shm)
        shm_infos[key] = info

    t_shm0 = time.perf_counter()
    add_shm("U", U, "U")
    add_shm("V", V, "V")
    add_shm("W0", W0, "W0")
    add_shm("W", W, "W")
    add_shm("active", active, "active")
    add_shm("core", core, "core")
    add_shm("out_offsets", out_offsets, "out_offsets")
    add_shm("out_eids", out_eids, "out_eids")
    add_shm("in_offsets", in_offsets, "in_offsets")
    add_shm("in_eids", in_eids, "in_eids")
    add_shm("n_nodes", n_nodes_arr, "n_nodes")
    t_shm1 = time.perf_counter()
    log_line(f"[SHM] built shared memory in {human_time(t_shm1 - t_shm0)}", log_path)

    # Attach coordinator to shm-backed arrays (must read updates from shm)
    shm_attach = {}
    for key, info in shm_infos.items():
        shm, arr = shm_attach_np(info)
        shm_attach[key] = (shm, arr)

    U_s = shm_attach["U"][1]
    V_s = shm_attach["V"][1]
    W0_s = shm_attach["W0"][1]
    W_s = shm_attach["W"][1]
    active_s = shm_attach["active"][1]
    core_s = shm_attach["core"][1]
    out_offsets_s = shm_attach["out_offsets"][1]
    out_eids_s = shm_attach["out_eids"][1]
    in_offsets_s = shm_attach["in_offsets"][1]
    in_eids_s = shm_attach["in_eids"][1]

    # -------------------- PROCESS POOL --------------------
    ctx = mp.get_context("fork")  # HPC Linux
    pool = ctx.Pool(processes=WORKERS, initializer=worker_init, initargs=(shm_infos,))

    # -------------------- MAIN LOOP --------------------
    deadline = start_wall + MAX_SECONDS
    next_hb = time.perf_counter() + HEARTBEAT_SEC

    t_find_tot = 0.0
    t_apply_tot = 0.0
    t_select_tot = 0.0
    t_trim_tot = 0.0
    t_scc_tot = 0.0

    stop_reason = "unknown"

    # aggregate cycle-len stats for HB
    cyc_len_stats = {"min": None, "max": 0, "sum": 0, "cnt": 0}

    log_line("[LOOP] entering main loop", log_path)

    try:
        while True:
            now = time.perf_counter()
            if now >= deadline:
                stop_reason = "time_budget_exhausted"
                break

            # Update core_nodes (cheap-ish)
            core_nodes = np.where(core_s == 1)[0].astype(np.int32)

            # Periodic SCC focus
            if DO_SCC_FOCUS:
                if (now - last_scc_time) >= SCC_RECOMPUTE_EVERY_SEC or (removed_total - last_scc_deact) >= SCC_RECOMPUTE_EVERY_DEACT:
                    t0 = time.perf_counter()
                    comp_id, comp_sizes, num_comps, cyclic, focus_comp, focus_nodes = kosaraju_scc_active_core(
                        n, U_s, V_s, active_s, core_s, out_offsets_s, out_eids_s, in_offsets_s, in_eids_s
                    )
                    t1 = time.perf_counter()
                    t_scc_tot += (t1 - t0)
                    last_scc_time = time.perf_counter()
                    last_scc_deact = removed_total
                    log_line(f"[SCC] reason=periodic comps={fmt_int(num_comps)} cyclic_comps={fmt_int(len(cyclic))} "
                             f"focus_comp={focus_comp} focus_nodes={fmt_int(len(focus_nodes))} time={human_time(t1-t0)}",
                             log_path)

            # Sample starts
            starts = sample_starts(rng, core_nodes, focus_nodes, hot_nodes, SAMPLE_STARTS_TOTAL)
            if not starts:
                stop_reason = "no_core_nodes_to_sample"
                log_line("[STOP] no core nodes left to sample (graph likely acyclic under core filter)", log_path)
                break

            # Split starts across workers
            per = max(1, len(starts) // WORKERS)
            chunks = []
            idx = 0
            for w in range(WORKERS):
                chunk = starts[idx: idx + per]
                idx += per
                if w == WORKERS - 1:
                    chunk += starts[idx:]
                chunks.append(chunk)

            # -------- find cycles (parallel) --------
            t_find0 = time.perf_counter()
            tasks = [(chunks[w], MAX_DFS_EDGE_STEPS, CYCLES_PER_WORKER_TARGET, MAX_CYCLE_LEN_STORE) for w in range(WORKERS)]
            results = pool.map(worker_find_cycles, tasks)
            t_find1 = time.perf_counter()
            t_find_tot += (t_find1 - t_find0)

            all_cycles = []
            attempts_sum = 0
            found_sum = 0
            steps_sum = 0
            trunc_sum = 0

            for r in results:
                cs = r["cycles"]
                all_cycles.extend(cs)
                attempts_sum += int(r["attempts"])
                found_sum += int(r["found"])
                steps_sum += int(r["steps_tot"])
                trunc_sum += int(r["trunc"])

            if not all_cycles:
                # near-acyclic in sampled region; confirm via SCC
                if DO_SCC_FOCUS:
                    t0 = time.perf_counter()
                    comp_id, comp_sizes, num_comps, cyclic, focus_comp, focus_nodes = kosaraju_scc_active_core(
                        n, U_s, V_s, active_s, core_s, out_offsets_s, out_eids_s, in_offsets_s, in_eids_s
                    )
                    t1 = time.perf_counter()
                    t_scc_tot += (t1 - t0)
                    last_scc_time = time.perf_counter()
                    last_scc_deact = removed_total
                    log_line(f"[SCC] reason=no_cycles_found comps={fmt_int(num_comps)} cyclic_comps={fmt_int(len(cyclic))} "
                             f"focus_comp={focus_comp} focus_nodes={fmt_int(len(focus_nodes))} time={human_time(t1-t0)}",
                             log_path)
                    if len(cyclic) == 0:
                        stop_reason = "acyclic_confirmed_by_scc"
                        log_line("[STOP] SCC says no cyclic components remain (acyclic)", log_path)
                        break
                # unlucky sampling: continue
                continue

            # Hot nodes update (lightweight)
            for cyc in all_cycles[:min(len(all_cycles), 64)]:
                for eid in cyc[:min(len(cyc), 256)]:
                    hot_nodes.append(int(U_s[eid]))
                    hot_nodes.append(int(V_s[eid]))

            # -------- select edge-disjoint subset --------
            t_sel0 = time.perf_counter()
            selected, sel_stats = greedy_select_edge_disjoint(all_cycles, BATCH_DISJOINT_TARGET, scan_all=False)
            t_sel1 = time.perf_counter()
            t_select_tot += (t_sel1 - t_sel0)

            if not selected:
                selected = [min(all_cycles, key=len)]
                log_line("[SELECT] selected empty due to heavy overlap; forcing one shortest cycle", log_path)

            # cycle length stats
            lens = [len(c) for c in selected]
            if lens:
                cyc_len_stats["min"] = min(lens) if cyc_len_stats["min"] is None else min(cyc_len_stats["min"], min(lens))
                cyc_len_stats["max"] = max(cyc_len_stats["max"], max(lens))
                cyc_len_stats["sum"] += sum(lens)
                cyc_len_stats["cnt"] += len(lens)

            # -------- apply reductions on disjoint cycles (parallel) --------
            buckets = [[] for _ in range(WORKERS)]
            for i, cyc in enumerate(selected):
                buckets[i % WORKERS].append(cyc)

            t_app0 = time.perf_counter()
            app_results = pool.map(worker_apply_disjoint_cycles, buckets)
            t_app1 = time.perf_counter()
            t_apply_tot += (t_app1 - t_app0)

            # Aggregate deactivations + eps stats
            deactivated_all = []
            cycles_applied_now = 0
            edges_touched_now = 0
            eps_min = None
            eps_max = None
            eps_sum = 0.0
            eps_cnt = 0

            for ar in app_results:
                deactivated_all.extend(ar["deactivated"])
                cycles_applied_now += int(ar["cycles_applied"])
                edges_touched_now += int(ar["edges_touched"])

                if ar["cycles_applied"] > 0:
                    eps_cnt += int(ar["cycles_applied"])
                    eps_sum += float(ar["eps_avg"]) * int(ar["cycles_applied"])
                    eps_min = float(ar["eps_min"]) if eps_min is None else min(eps_min, float(ar["eps_min"]))
                    eps_max = float(ar["eps_max"]) if eps_max is None else max(eps_max, float(ar["eps_max"]))

            if deactivated_all:
                deactivated_all = list(set(deactivated_all))

            cycles_total += cycles_applied_now

            # Count deactivations exactly (only those not already marked removed)
            deact_now = 0
            for eid in deactivated_all:
                if removed_mask[eid] == 0:
                    removed_mask[eid] = 1
                    deact_now += 1

            removed_total += deact_now
            active_edges = int(active_s.sum())  # exact count (cheap enough once/iter)

            # -------- update degrees and core trimming incrementally --------
            trimmed_now = 0
            if DO_CORE_TRIM and deact_now > 0:
                t_trim0 = time.perf_counter()

                for eid in deactivated_all:
                    u = int(U_s[eid]); v = int(V_s[eid])
                    if core_s[u]:
                        outdeg_core[u] -= 1
                        if outdeg_core[u] == 0:
                            trim_queue.append(u)
                    if core_s[v]:
                        indeg_core[v] -= 1
                        if indeg_core[v] == 0:
                            trim_queue.append(v)

                tcount, wsteps = core_trim_incremental(
                    trim_queue, core_s, indeg_core, outdeg_core,
                    U_s, V_s, active_s,
                    out_offsets_s, out_eids_s, in_offsets_s, in_eids_s,
                    work_cap=TRIM_PROCESS_LIMIT_PER_ITER
                )
                trimmed_now += tcount

                t_trim1 = time.perf_counter()
                t_trim_tot += (t_trim1 - t_trim0)

            # -------- heartbeat --------
            now = time.perf_counter()
            if now >= next_hb:
                elapsed = now - start_wall
                core_cnt = int(core_s.sum())
                focus_cnt = int(len(focus_nodes)) if (focus_nodes is not None) else 0

                avg_cycle_len = (cyc_len_stats["sum"] / cyc_len_stats["cnt"]) if cyc_len_stats["cnt"] > 0 else 0.0
                eps_avg = (eps_sum / eps_cnt) if eps_cnt > 0 else 0.0

                dfs_steps_per_found = (steps_sum / max(1, found_sum))
                disjoint_rate = (len(selected) / max(1, len(all_cycles)))

                log_line(
                    f"[HB] elapsed={human_time(elapsed)} "
                    f"cycles_applied_total={fmt_int(cycles_total)} "
                    f"deact_total={fmt_int(removed_total)} "
                    f"deact_rate={fmt_float(removed_total/max(1.0,elapsed),3)}/s "
                    f"active_edges={fmt_int(active_edges)} "
                    f"core_nodes={fmt_int(core_cnt)} focus_nodes={fmt_int(focus_cnt)} "
                    f"find={human_time(t_find_tot)} sel={human_time(t_select_tot)} apply={human_time(t_apply_tot)} "
                    f"trim={human_time(t_trim_tot)} scc={human_time(t_scc_tot)} "
                    f"last: found_cycles={fmt_int(len(all_cycles))} selected={fmt_int(len(selected))} "
                    f"disjoint_rate={disjoint_rate:.3f} rej_overlap_seen={fmt_int(sel_stats['rejected_overlap_seen'])} "
                    f"deact_now={fmt_int(deact_now)} trimmed_now={fmt_int(trimmed_now)} "
                    f"cycle_len[min/avg/max]={lens and min(lens) or 0}/{avg_cycle_len:.1f}/{lens and max(lens) or 0} "
                    f"eps[min/avg/max]={0.0 if eps_min is None else eps_min:.3g}/{eps_avg:.3g}/{0.0 if eps_max is None else eps_max:.3g} "
                    f"dfs_attempts={fmt_int(attempts_sum)} dfs_found={fmt_int(found_sum)} "
                    f"dfs_steps={fmt_int(steps_sum)} dfs_steps_per_found={dfs_steps_per_found:.1f} trunc={fmt_int(trunc_sum)}",
                    log_path
                )
                next_hb = now + HEARTBEAT_SEC

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        log_line("[STOP] keyboard interrupt", log_path)
    except Exception as e:
        stop_reason = f"exception:{type(e).__name__}"
        log_line(f"[ERROR] {type(e).__name__}: {e}", log_path)
        raise
    finally:
        # IMPORTANT: do NOT unlink shared memory here; we still need it for ranking/add-back
        pass

    # ============================================================
    # FINALIZATION: add-back (optional) + ranking + write outputs
    # ============================================================

    elapsed = time.perf_counter() - start_wall
    remaining = max(0.0, MAX_SECONDS - elapsed)
    log_line(f"[STOP] reason={stop_reason} elapsed={human_time(elapsed)} remaining={human_time(remaining)} "
             f"active_edges={fmt_int(int(active_s.sum()))} core_nodes={fmt_int(int(core_s.sum()))}",
             log_path)

    # If acyclic, optionally do add-back with time budget fraction
    did_add_back = False
    add_back_accepted = 0

    # Decide method: try topo (acyclic) first
    t_rank0 = time.perf_counter()
    log_line("[RANK] start", log_path)

    try:
        rank = topo_order_active(n, out_offsets_s, out_eids_s, V_s, active_s)
        log_line("[RANK] graph is acyclic (topo order available)", log_path)

        if DO_ADD_BACK and remaining > 1.0:
            did_add_back = True
            add_deadline = time.perf_counter() + (ADD_BACK_MAX_FRACTION_REMAINING * remaining)
            log_line(f"[ADD_BACK] budget={human_time(ADD_BACK_MAX_FRACTION_REMAINING * remaining)}", log_path)
            add_back_accepted = add_back_heavy_first(
                removed_mask=removed_mask,
                U=U_s, V=V_s, W0=W0_s, active=active_s,
                out_offsets=out_offsets_s, out_eids=out_eids_s,
                n=n, time_deadline=add_deadline, log_path=log_path
            )
            # re-topo after add-back
            rank = topo_order_active(n, out_offsets_s, out_eids_s, V_s, active_s)
            log_line(f"[RANK] topo recomputed after add-back accepted={fmt_int(add_back_accepted)}", log_path)

        rank_method = "topo"

    except Exception:
        log_line("[RANK] still cyclic -> using SCC-condensation ranking", log_path)
        rank, num_comps, cyclic_comps = scc_condensation_ranking(
            n, U_s, V_s, active_s, out_offsets_s, out_eids_s, in_offsets_s, in_eids_s
        )
        log_line(f"[RANK] scc_condensation comps={fmt_int(num_comps)} cyclic_comps={fmt_int(cyclic_comps)}", log_path)
        rank_method = "scc_condensation"

    t_rank1 = time.perf_counter()
    log_line(f"[RANK] done method={rank_method} time={human_time(t_rank1 - t_rank0)}", log_path)

    # Write CSV
    t_w0 = time.perf_counter()
    df = pd.DataFrame({
        "Node ID": [index_to_node[i] for i in range(n)],
        "Order": rank.astype(np.int64)
    })
    df.to_csv(out_csv, index=False)
    t_w1 = time.perf_counter()
    log_line(f"[WRITE] wrote ranking csv rows={fmt_int(n)} path={out_csv} time={human_time(t_w1 - t_w0)}", log_path)

    # ============================================================
    # CLEANUP: pool + shared memory
    # ============================================================

    log_line("[CLEANUP] starting", log_path)

    try:
        pool.close()
        pool.join()
    except Exception:
        try:
            pool.terminate()
            pool.join()
        except Exception:
            pass

    for shm, _arr in shm_attach.values():
        try:
            shm.close()
        except Exception:
            pass

    for shm in shm_blocks:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass

    log_line("[CLEANUP] done", log_path)

    print("Wrote log:", log_path)
    print("Wrote ranking:", out_csv)


if __name__ == "__main__":
    main()
