#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os

import random
import datetime
from collections import defaultdict

# !pip install networkx
# !pip install pandas
# !pip install scikit-learn
# !pip install scipy
# !pip install pyparsing
# !pip install fsspec
import networkx as nx
import multiprocessing as mp

import heapq
import time

# from google.colab import drive
# !pip install openpyxl
import pandas as pd


# === Mount Google Drive ===
# drive.mount('/content/drive', force_remount=True)

def read_graph(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={
        'Source Node  ID': 'source',
        'Target Node ID': 'target',
        'Edge Weight': 'weight'
    }, inplace=True)
    df['source'] = df['source'].astype(str)
    df['target'] = df['target'].astype(str)
    edges = list(df.itertuples(index=False, name=None))
    node_set = sorted(set(u for u, v, _ in edges).union(v for u, v, _ in edges))
    node_to_index = {node: i for i, node in enumerate(node_set)}
    index_to_node = {i: node for node, i in node_to_index.items()}
    edges_indexed = [(node_to_index[u], node_to_index[v], float(w)) for (u, v, w) in edges]
    return edges_indexed, node_to_index, index_to_node


def load_initial_scores(csv_path, node_to_index):
    df = pd.read_csv(csv_path)
    df['Node ID'] = df['Node ID'].astype(str).str.strip()
    rank_map = {row['Node ID']: row['Order'] for _, row in df.iterrows()}

    scores = {}
    for node_str, idx in node_to_index.items():
        if node_str in rank_map:
            scores[idx] = int(rank_map[node_str])

    # Assign unique ranks to unranked nodes
    max_rank = max(scores.values(), default=0) + 1
    for node_str, idx in node_to_index.items():
        if idx not in scores:
            scores[idx] = max_rank
            max_rank += 1

    return scores


def compute_forward_weight(edges, scores):
    return sum(w for u, v, w in edges if scores[u] < scores[v])


# In[6]:


def compare_forward_weights(edges, sbef, saft):
    """
    Given edges and two score dictionaries (sbef and saft),
    returns:
    - sum of forward edge weights in sbef that are NOT forward in saft
    - sum of forward edge weights in saft that are NOT forward in sbef
    """
    sum_sbef_only = 0.0
    sum_saft_only = 0.0
    sum_both_forward = 0.0
    sum_both_backward = 0.0
    only_before_gobal = set()
    for u, v, w in edges:
        is_forward_sbef = sbef[u] < sbef[v]
        is_forward_saft = saft[u] < saft[v]

        if is_forward_sbef and not is_forward_saft:
            sum_sbef_only += w
            only_before_gobal.add((u, v, w))
        if is_forward_saft and not is_forward_sbef:
            sum_saft_only += w
        if is_forward_sbef and is_forward_saft:
            sum_both_forward += w
        if not is_forward_sbef and not is_forward_saft:
            sum_both_backward += w
    return sum_sbef_only, sum_saft_only, sum_both_forward, sum_both_backward, only_before_gobal


def all_scores_unique(scores):
    """
    Returns True if all nodes have unique scores, otherwise False.
    """
    values = list(scores.values())
    return len(values) == len(set(values))


def global_scc_topo_reorder_scores(G, edges, scores, debug=False):
    """
    Reorder 'scores' so that each SCC's vertices are contiguous in rank and SCCs
    appear in a topological order of the condensation DAG. FW must not decrease.
    """
    if debug:
        print("🔧 Global SCC-based topological reorder of all vertices...")

    sccs = list(nx.strongly_connected_components(G))
    if debug:
        print(f"   Found {len(sccs)} SCCs")

    node_to_scc = {}
    for scc_id, scc in enumerate(sccs):
        for n in scc:
            node_to_scc[n] = scc_id

    scc_dag = nx.DiGraph()
    scc_dag.add_nodes_from(range(len(sccs)))
    for u, v, w in edges:
        su = node_to_scc[u]
        sv = node_to_scc[v]
        if su != sv:
            scc_dag.add_edge(su, sv)

    scc_order = list(nx.topological_sort(scc_dag))
    if debug:
        print(f"   Topological order of SCCs length = {len(scc_order)}")

    fw_before = compute_forward_weight(edges, scores)
    if debug:
        print(f"   FW BEFORE global SCC reorder = {fw_before:.6f}")

    new_scores = {}
    next_rank = 0
    for scc_id in scc_order:
        nodes_in_scc = list(sccs[scc_id])
        nodes_in_scc_sorted = sorted(nodes_in_scc, key=lambda n: scores[n])
        for n in nodes_in_scc_sorted:
            new_scores[n] = next_rank
            next_rank += 1

    fw_after = compute_forward_weight(edges, new_scores)
    if debug:
        print(f"   FW AFTER  global SCC reorder = {fw_after:.6f}")

    if fw_after + 1e-9 < fw_before:
        raise RuntimeError(
            f"❌ Global SCC topo reorder decreased FW: "
            f"before={fw_before:.6f}, after={fw_after:.6f}"
        )

    if debug:
        print("✅ Global SCC topo reorder done (no FW decrease). "
              "Each SCC is now contiguous in rank.")

    return new_scores


# In[7]:


def apply_new_strategy(scores, u, v, between, out_edges, in_edges, edges_dict, debug=False):
    import time
    import datetime
    import os

    PROFILE_THRESHOLD = 0.08  # seconds – log only if slower than this (or debug=True)

    pid = os.getpid()

    def now_str():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(msg):
        print(f"[{now_str()}][PID {pid}][apply_new_strategy] {msg}")

    t_total0 = time.time()

    # IMPORTANT: we now treat the incoming 'scores' as the "frozen" baseline
    # and DO NOT copy it. We only modify scores in the reorder step.
    scores_before = scores
    ranks_before = (scores_before[u], scores_before[v])

    # Optional: sanity check that 'between' is sorted by rank (only if debug)
    if debug and len(between) > 1:
        b_ranks = [scores_before[x] for x in between]
        for i in range(len(b_ranks) - 1):
            if b_ranks[i] >= b_ranks[i + 1]:
                log(
                    f"❌ RUNTIME ERROR: 'between' is not strictly sorted by rank "
                    f"at pos {i}: {b_ranks[i]} >= {b_ranks[i + 1]}"
                )
                raise RuntimeError("Between list not sorted by scores in apply_new_strategy.")

    # ----------------- 1) Define neighbors of v and u within 'between' -----------------
    t0 = time.time()
    v_neighbors = [node for node in between
                   if (node, v) in edges_dict or (v, node) in edges_dict]
    u_neighbors = [node for node in between
                   if (node, u) in edges_dict or (u, node) in edges_dict]
    t_neighbors_build = time.time() - t0

    # Sort neighbors by scores_before (in theory 'between' is already sorted,
    # but we keep the structure consistent). We do NOT call sorted(...) to avoid
    # extra O(k log k) unless needed; instead we assert sortedness.
    t0 = time.time()
    v_neighbors_sorted_before = v_neighbors  # same order as in 'between'

    u_neighbors_sorted_before = u_neighbors  # same order as in 'between'
    u_scores = [scores_before[node] for node in u_neighbors_sorted_before]
    # This should hold because 'between' is sorted and filtering preserves order.
    #  assert all(u_scores[i] <= u_scores[i + 1] for i in range(len(u_scores) - 1)), \
    #      "u_scores is not sorted!"
    t_neighbors_sort = time.time() - t0

    # ----------------- 2) Precompute prefix sums for v_neighbors -----------------
    t0 = time.time()
    outgoing_v = [0] * (len(v_neighbors_sorted_before) + 1)
    ingoing_v = [0] * (len(v_neighbors_sorted_before) + 1)
    for i, node in enumerate(v_neighbors_sorted_before):
        outgoing_v[i + 1] = outgoing_v[i] + edges_dict.get((v, node), 0)
        ingoing_v[i + 1] = ingoing_v[i] + edges_dict.get((node, v), 0)
    t_prefix_v = time.time() - t0

    # ----------------- 3) Precompute suffix sums for u_neighbors -----------------
    t0 = time.time()
    outgoing_u = [0] * (len(u_neighbors_sorted_before) + 1)
    ingoing_u = [0] * (len(u_neighbors_sorted_before) + 1)
    for i in reversed(range(len(u_neighbors_sorted_before))):
        node_i = u_neighbors_sorted_before[i]
        outgoing_u[i] = outgoing_u[i + 1] + edges_dict.get((u, node_i), 0)
        ingoing_u[i] = ingoing_u[i + 1] + edges_dict.get((node_i, u), 0)
    t_prefix_u = time.time() - t0

    # ----------------- 4) Find best cut (two-pointer sweep) -----------------
    t0 = time.time()
    best_delta = float("-inf")
    best_y = -1
    best_j = -1

    w_uv = edges_dict.get((u, v), 0.0)
    w_vu = edges_dict.get((v, u), 0.0)
    base = w_uv - w_vu

    us = u_scores
    len_u = len(us)
    fs = scores_before
    ing_v = ingoing_v
    out_v = outgoing_v
    ing_u = ingoing_u
    out_u = outgoing_u

    j = 0
    prev_s = -float("inf")

    for i, nv in enumerate(v_neighbors_sorted_before):
        s = fs[nv]

        # advance j only forward (most cases)
        if s >= prev_s:
            while j < len_u and us[j] <= s:
                j += 1
        else:
            # if sequence is not sorted, abort immediately
            raise RuntimeError(
                f"Non-monotone order detected at i={i}: "
                f"s={s:.6f} < prev_s={prev_s:.6f} (nv={nv})"
            )

        prev_s = s

        if j < len_u:
            delta = base + ing_v[i + 1] - out_v[i + 1] - ing_u[j] + out_u[j]
            if delta > best_delta:
                best_delta = delta
                best_y = i
                best_j = j

    t_sweep = time.time() - t0

    # ----------------- 5) Early exit if no gain -----------------
    if best_delta <= 0:
        t_total = time.time() - t_total0
        # if t_total > PROFILE_THRESHOLD or debug:
        #     log(
        #         f"SLOW no-gain: u={u}, v={v}, "
        #         f"|between|={len(between)}, |Nv|={len(v_neighbors)}, |Nu|={len(u_neighbors)}, "
        #         f"best_delta={best_delta:.3f}, "
        #         f"times(s): neighbors_build={t_neighbors_build:.4f}, "
        #         f"neighbors_sort={t_neighbors_sort:.4f}, "
        #         f"prefix_v={t_prefix_v:.4f}, prefix_u={t_prefix_u:.4f}, "
        #         f"sweep={t_sweep:.4f}, total={t_total:.4f}"
        #     )
        return False, 0.0, ranks_before, (scores_before[u], scores_before[v]), {}

    # ----------------- 6) Apply reordering (success case) -----------------
    t0 = time.time()
    # Apply reordering
    split_point = between.index(v_neighbors_sorted_before[best_y])
    left = between[:split_point + 1]
    right = between[split_point + 1:]
    new_order = left + [u, v] + right
    original_block = [v] + between + [u]
    base_score = scores_before[v]

    # Backup scores (only for this block, NOT the entire dict)
    old_scores_snapshot = {node: scores_before[node] for node in original_block}
    split_point_node = v_neighbors_sorted_before[best_y]
    old_split_score = old_scores_snapshot[split_point_node]

    # Now we actually modify 'scores'
    for i, node in enumerate(new_order):
        new_score = base_score + i
        old_score = scores[node]
        scores[node] = new_score

        if node not in {u, v}:
            delta_rank = abs(new_score - old_score)
            if delta_rank > 1:
                print("\n❌ Rank change violation for a non-u/v node!")
                print(f"🔸 Violating node: {node}")
                print(f"🔹 Old Score based on scores: {old_score}")
                print(f"🔹 Old Score based on old_scores_snapshot: {old_scores_snapshot[node]}")
                print(f"🔹 New Score: {new_score}")
                print(f"🔹 Delta: {delta_rank}")
                print(f"📌 Split point node: {split_point_node}")
                print(f"    ↳ Old rank = {old_split_score}")
                print(f"    ↳ New rank = {scores[split_point_node]}")
                print(f"📏 Length of LEFT block = {len(left)}")
                print(f"📏 Length of RIGHT block = {len(right)}")
                print(f"u = {u}, old rank = {old_scores_snapshot[u]}, new rank = {scores[u]}")
                print(f"v = {v}, old rank = {old_scores_snapshot[v]}, new rank = {scores[v]}")
                print(f"base_score = {base_score}")

                print("\n📜 Previous order with scores:")
                for n in original_block:
                    print(f"{n}: {old_scores_snapshot[n]}")

                print("\n📜 New order with scores:")
                for n in new_order:
                    print(f"{n}: {scores[n]}")

                raise ValueError(f"Node {node} changed score by {delta_rank}, which exceeds limit.")

    t_reorder = time.time() - t0

    ranks_after = (scores[u], scores[v])

    # ----------------- 7) Re-sort neighbors based on updated scores (optional debug) -----------------
    t0 = time.time()
    v_neighbors_sorted_after = sorted(v_neighbors, key=lambda x: scores[x])
    u_neighbors_sorted_after = sorted(u_neighbors, key=lambda x: scores[x])
    t_neighbors_after = time.time() - t0

    # Debug information (kept minimal; heavy parts commented out)
    debug_info = {
        # Placeholders if you want to re-enable rich debug later.
        # "v_neighbors_before": [(node, scores_before[node]) for node in v_neighbors_sorted_before],
        # "u_neighbors_before": [(node, scores_before[node]) for node in u_neighbors_sorted_before],
        # "v_neighbors_after": [(node, scores[node]) for node in v_neighbors_sorted_after],
        # "u_neighbors_after": [(node, scores[node]) for node in u_neighbors_sorted_after],
        # "best_y": best_y,
        # "best_j": best_j,
        # "best_delta": best_delta,
    }

    # ----------------- 8) Final profiling log for success case -----------------
    t_total = time.time() - t_total0
    # if t_total > PROFILE_THRESHOLD or debug:
    #     log(
    #         f"SLOW success: u={u}, v={v}, "
    #         f"|between|={len(between)}, |Nv|={len(v_neighbors)}, |Nu|={len(u_neighbors)}, "
    #         f"best_delta={best_delta:.3f}, "
    #         f"times(s): neighbors_build={t_neighbors_build:.4f}, "
    #         f"neighbors_sort={t_neighbors_sort:.4f}, "
    #         f"prefix_v={t_prefix_v:.4f}, prefix_u={t_prefix_u:.4f}, "
    #         f"sweep={t_sweep:.4f}, reorder={t_reorder:.4f}, "
    #         f"neighbors_after={t_neighbors_after:.4f}, total={t_total:.4f}"
    #     )

    return True, best_delta, ranks_before, ranks_after, debug_info


# In[8]:


def compute_all_gains_local(scores, u, v, out_edges, in_edges):
    idx_u = scores[u]
    idx_v = scores[v]
    if idx_u <= idx_v:
        raise ValueError(f"Invalid edge ({u}->{v}) is already forward!")

    lo, hi = idx_v, idx_u

    # Local bindings for speed
    scores_local = scores
    idx_u_local = idx_u
    idx_v_local = idx_v

    def new_rank(node, rank_node, strategy):
        """Compute new rank for 'node' under given strategy, using its original rank."""
        if strategy == 'swap':
            if node == u:
                return idx_v_local
            if node == v:
                return idx_u_local
            return rank_node

        elif strategy == 'mvvafu':  # move v after u
            if node == v:
                return idx_u_local
            if idx_v_local < rank_node <= idx_u_local:
                return rank_node - 1
            return rank_node

        elif strategy == 'mvubfv':  # move u before v
            if node == u:
                return idx_v_local
            if idx_v_local <= rank_node < idx_u_local:
                return rank_node + 1
            return rank_node

        # Should never happen
        return rank_node

    # Store just the gains, we don’t actually use the f2b/b2f breakdown externally
    swap_gain = 0.0
    mvvafu_gain = 0.0
    mvubfv_gain = 0.0

    # If you ever care again:
    # swap_f2b = swap_b2f = mvvafu_f2b = mvvafu_b2f = mvubfv_f2b = mvubfv_b2f = 0.0

    edges_checked = 0

    def process_edge(a, b, w):
        nonlocal edges_checked
        nonlocal swap_gain, mvvafu_gain, mvubfv_gain

        sa = scores_local[a]
        sb = scores_local[b]

        # Only edges touching [lo, hi] matter
        if not (lo <= sa <= hi or lo <= sb <= hi):
            return

        edges_checked += 1
        old_forward = (sa < sb)

        # ---- swap ----
        ra2 = new_rank(a, sa, 'swap')
        rb2 = new_rank(b, sb, 'swap')
        new_forward = (ra2 < rb2)
        if old_forward and not new_forward:
            swap_gain -= w
            # swap_f2b += w
        elif (not old_forward) and new_forward:
            swap_gain += w
            # swap_b2f += w

        # ---- mvvafu ----
        ra2 = new_rank(a, sa, 'mvvafu')
        rb2 = new_rank(b, sb, 'mvvafu')
        new_forward = (ra2 < rb2)
        if old_forward and not new_forward:
            mvvafu_gain -= w
            # mvvafu_f2b += w
        elif (not old_forward) and new_forward:
            mvvafu_gain += w
            # mvvafu_b2f += w

        # ---- mvubfv ----
        ra2 = new_rank(a, sa, 'mvubfv')
        rb2 = new_rank(b, sb, 'mvubfv')
        new_forward = (ra2 < rb2)
        if old_forward and not new_forward:
            mvubfv_gain -= w
            # mvubfv_f2b += w
        elif (not old_forward) and new_forward:
            mvubfv_gain += w
            # mvubfv_b2f += w

    # Collect candidate edges (neighbors of u or v within [lo, hi])
    # Using a set is still OK here to avoid double-counting; but we only
    # hash (a,b,w) once per neighbor.
    edges_to_check = set()

    # Local bindings for adjacency to avoid global lookups
    out_u = out_edges[u]
    in_u = in_edges[u]
    out_v = out_edges[v]
    in_v = in_edges[v]
    scores_vals = scores_local  # alias

    for x, w in out_u:
        rx = scores_vals[x]
        if lo <= rx <= hi:
            edges_to_check.add((u, x, w))

    for x, w in in_u:
        rx = scores_vals[x]
        if lo <= rx <= hi:
            edges_to_check.add((x, u, w))

    for x, w in out_v:
        rx = scores_vals[x]
        if lo <= rx <= hi:
            edges_to_check.add((v, x, w))

    for x, w in in_v:
        rx = scores_vals[x]
        if lo <= rx <= hi:
            edges_to_check.add((x, v, w))

    for a, b, w in edges_to_check:
        process_edge(a, b, w)

    return swap_gain, mvvafu_gain, mvubfv_gain


# In[9]:


def select_nonconflicting_edge_indices_dp(backward_edges):
    """
    Given a list of backward_edges = (u, v, w, lo, hi) with lo < hi (ranks),
    select a maximum-size subset of edges whose [lo, hi] intervals do not overlap.

    IMPORTANT:
      - We compress coordinates so that DP is defined only on ranks that are
        actually endpoints of some edge (lo or hi), not on [0..max_rank].

    Returns:
      selected_indices: set of LOCAL indices (0..len(backward_edges)-1)
                        of chosen edges.

    Complexity per call:
      O(E + K), where E = len(backward_edges),
      and K = number of distinct endpoints ≤ 2E → overall O(E).
    """
    if not backward_edges:
        return set()

    # 1) Collect all distinct endpoints (lo, hi)
    endpoints = set()
    for (_, _, _, lo, hi) in backward_edges:
        endpoints.add(lo)
        endpoints.add(hi)

    # Sort endpoints and build compression map
    sorted_endpoints = sorted(endpoints)
    coord_index = {rank: i for i, rank in enumerate(sorted_endpoints)}
    K = len(sorted_endpoints)  # number of DP states

    # 2) For each compressed "hi position", keep the edge with the largest compressed lo
    best_edge_idx_at_pos = [-1] * K
    best_edge_lo_pos_at_pos = [-1] * K

    for idx, (u, v, w, lo, hi) in enumerate(backward_edges):
        # Compress endpoints
        lo_pos = coord_index[lo]
        hi_pos = coord_index[hi]

        # For each hi_pos, choose the edge with maximal lo_pos
        if best_edge_idx_at_pos[hi_pos] == -1 or lo_pos > best_edge_lo_pos_at_pos[hi_pos]:
            best_edge_idx_at_pos[hi_pos] = idx
            best_edge_lo_pos_at_pos[hi_pos] = lo_pos

    # 3) DP over compressed positions 0..K-1
    dp = [0] * K
    choose_pos = [False] * K

    for pos in range(K):
        # Option 1: skip this position
        opt1 = dp[pos - 1] if pos > 0 else 0

        # Option 2: take the edge that ends at this position (if any)
        opt2 = -1
        if best_edge_idx_at_pos[pos] != -1:
            lo_pos = best_edge_lo_pos_at_pos[pos]
            before_lo = dp[lo_pos - 1] if lo_pos > 0 else 0
            opt2 = 1 + before_lo

        if opt2 > opt1:
            dp[pos] = opt2
            choose_pos[pos] = True
        else:
            dp[pos] = opt1
            choose_pos[pos] = False

    # 4) Reconstruct chosen edges (original local indices into backward_edges)
    selected_indices = set()
    pos = K - 1
    while pos >= 0:
        if choose_pos[pos] and best_edge_idx_at_pos[pos] != -1:
            edge_idx = best_edge_idx_at_pos[pos]
            selected_indices.add(edge_idx)
            lo_pos = best_edge_lo_pos_at_pos[pos]
            pos = lo_pos - 1
        else:
            pos -= 1

    return selected_indices


# In[10]:


def worker_loop(worker_idx, num_procs,
                backward_edges,
                scores_snapshot,
                out_edges, in_edges, edges_dict,
                active_intervals,  # unused now
                used_intervals,  # unused now
                lock,  # unused now
                edge_queue,
                result_queue):
    import os
    import time
    import datetime
    import random

    pid = os.getpid()

    def now_str():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(msg):
        print(f"[{now_str()}][PID {pid}][WORKER {worker_idx}/{num_procs}] {msg}")

    # Thresholds for logging
    BETWEEN_WARN = 0.05
    APPLY_WARN = 0.10
    GREEDY_WARN = 0.05
    EDGE_WARN = 0.20
    LOCK_FRACTION_WARN = 0.25

    BETWEEN_SAFETY_CHECK_LIMIT = 5
    between_checks_done = 0

    # -------------------- SNAPSHOT / RANK ARRAY --------------------
    # Build rank_to_node once from snapshot
    max_rank = max(scores_snapshot.values())
    rank_to_node = [None] * (max_rank + 1)
    for node, r in scores_snapshot.items():
        if r < 0 or r > max_rank:
            log(f"❌ RUNTIME ERROR: Invalid rank in scores_snapshot: node={node}, rank={r}")
            raise RuntimeError("Invalid rank in scores_snapshot.")
        if rank_to_node[r] is not None:
            log(f"❌ RUNTIME ERROR: Duplicate rank in scores_snapshot: rank={r}, "
                f"nodes={rank_to_node[r]} and {node}")
            raise RuntimeError("Non-injective ranks in scores_snapshot.")
        rank_to_node[r] = node

    # Quick structural sanity: ensure no None in used ranks
    used_ranks = set(scores_snapshot.values())
    for r in used_ranks:
        if rank_to_node[r] is None:
            log(f"❌ RUNTIME ERROR: rank_to_node[{r}] is None while some node has rank {r}")
            raise RuntimeError("rank_to_node inconsistent with scores_snapshot.")

    # Random sample consistency check
    sample_nodes = random.sample(list(scores_snapshot.keys()),
                                 min(10, len(scores_snapshot)))
    for n in sample_nodes:
        rr = scores_snapshot[n]
        if rank_to_node[rr] != n:
            log(f"❌ RUNTIME ERROR: rank_to_node[{rr}]={rank_to_node[rr]} != node {n}")
            raise RuntimeError("rank_to_node / scores_snapshot mismatch on sample check.")

    # log(f"🧩 Worker starting with {len(backward_edges)} edges in batch, max_rank={max_rank}")

    worker_start = time.time()

    total_lock_time = 0.0  # kept for stats (but essentially unused)
    total_apply_strategy_time = 0.0
    total_greedy_time = 0.0
    total_queue_wait_time = 0.0
    total_edge_processing_time = 0.0

    edges_processed = 0
    successes = 0
    failures = 0
    extended_successes = 0
    greedy_successes = 0

    # scores_base is the immutable snapshot for THIS worker/batch.
    # For each edge we make a fresh copy "scores" and only send back deltas.
    scores_base = scores_snapshot
    out_edges_base = out_edges
    in_edges_base = in_edges
    edges_array = backward_edges

    while True:
        # -----------------------------------------------
        # 1) POP EDGE INDEX FROM QUEUE (non-blocking)
        # -----------------------------------------------
        t0 = time.time()
        try:
            idx = edge_queue.get_nowait()
        except Exception:
            total_queue_wait_time += (time.time() - t0)
            # log("ℹ️ Edge queue empty; worker exiting processing loop.")
            break
        total_queue_wait_time += (time.time() - t0)

        edge_start = time.time()
        u, v, w, lo, hi = edges_array[idx]

        # -----------------------------------------------
        # DIRECTLY TRY extended + greedy (no conflicts)
        # -----------------------------------------------
        t_between = 0.0
        t_apply_dur = 0.0
        greedy_dur = 0.0
        mode = "none"
        success = False
        delta = 0.0
        changed_scores = {}

        try:
            idx_u = scores_base[u]
            idx_v = scores_base[v]

            # Sanity: ensure this really is backward in snapshot
            if not (idx_u > idx_v):
                log(f"⚠️ Edge ({u}->{v}) not backward in snapshot ranks: "
                    f"idx_u={idx_u}, idx_v={idx_v}. Skipping.")
                failures += 1
                continue

            # Build "between" via rank_to_node slice
            t_btw = time.time()
            if idx_u - idx_v > 1:
                slice_raw = rank_to_node[idx_v + 1: idx_u]
                between = [node for node in slice_raw if node is not None]
            else:
                between = []
            t_between = time.time() - t_btw

            if t_between > BETWEEN_WARN:
                log(f"⚠️ 'between' build slow: len={len(between)} time={t_between:.6f}s")

            # Optional safety check vs slow method (first few calls)
            if between_checks_done < BETWEEN_SAFETY_CHECK_LIMIT:
                between_checks_done += 1
                slow_between = [node for node, r in scores_base.items()
                                if idx_v < r < idx_u]
                if set(between) != set(slow_between):
                    log("❌ RUNTIME ERROR: Fast 'between' does not match old method!")
                    log(f"    fast_between_size={len(between)}, slow_between_size={len(slow_between)}")
                    raise RuntimeError("Fast 'between' implementation mismatch detected.")

            # Local copy for this edge
            scores = scores_base.copy()

            # ----------------- Extended strategy -----------------
            t_apply = time.time()
            try:
                ext_success, ext_delta, rB, rA, dbg = apply_new_strategy(
                    scores, u, v, between,
                    out_edges_base, in_edges_base, edges_dict,
                    debug=False
                )
                success = ext_success
                delta = ext_delta
                mode = "extended"
            except Exception as e:
                log(f"❌ ERROR apply_new_strategy({u}->{v}): {repr(e)}")
                success = False
                delta = 0.0
                mode = "extended_error"
            t_apply_dur = time.time() - t_apply
            total_apply_strategy_time += t_apply_dur

            if t_apply_dur > APPLY_WARN:
                log(f"⚠️ apply_new_strategy slow for edge ({u}->{v}): {t_apply_dur:.6f}s")

            # This should **never** happen: success but non-positive gain.
            if success and delta <= 0.0:
                log(f"❌ RUNTIME WARNING: apply_new_strategy returned "
                    f"success=True but delta={delta:.6f} for edge ({u}->{v}); "
                    "treating as failure and falling back to greedy.")
                success = False
                delta = 0.0
                mode = "extended_nonpositive_delta"

            # ----------------- Greedy fallback -----------------
            greedy_dur = 0.0
            if not (success and delta > 0):
                t_g = time.time()
                g1, g2, g3 = compute_all_gains_local(scores_base, u, v,
                                                     out_edges_base, in_edges_base)
                greedy_dur = time.time() - t_g
                total_greedy_time += greedy_dur

                # if greedy_dur > GREEDY_WARN:
                #     log(f"⚠️ greedy gain computation slow for edge ({u}->{v}): "
                #         f"{greedy_dur:.6f}s")

                best_gain = max(g1, g2, g3)
                if best_gain > 0:
                    scores = scores_base.copy()
                    delta = best_gain
                    success = True

                    if best_gain == g1:
                        scores[u], scores[v] = idx_v, idx_u
                        mode = "greedy/swap"
                    elif best_gain == g2:
                        # move v after u
                        for k, r in scores.items():
                            if idx_v < r <= idx_u:
                                scores[k] = r - 1
                        scores[v] = idx_u
                        mode = "greedy/move_v_after_u"
                    else:
                        # move u before v
                        for k, r in scores.items():
                            if idx_v <= r < idx_u:
                                scores[k] = r + 1
                        scores[u] = idx_v
                        mode = "greedy/move_u_before_v"
                else:
                    success = False
                    delta = 0.0
                    if mode == "extended_error":
                        mode = "none_after_error"
                    else:
                        mode = "none"
            else:
                # extended success; no greedy
                greedy_dur = 0.0

            # ----------------- Collect rank changes -----------------
            if success and delta > 0:
                for node, r in scores_base.items():
                    if lo <= r <= hi and scores[node] != r:
                        changed_scores[node] = scores[node]

                # Safety: ensure we actually have some changes
                if not changed_scores:
                    log(f"❌ RUNTIME ERROR: success with positive delta but "
                        f"no changed_scores for edge ({u}->{v}), mode={mode}")
                    success = False
                    delta = 0.0
                    mode = f"{mode}_no_changes"
                else:
                    # Check that all changed nodes were originally in [lo, hi]
                    for node in changed_scores.keys():
                        old_r = scores_base[node]
                        if not (lo <= old_r <= hi):
                            log(f"❌ RUNTIME ERROR: Node {node} changed but old rank {old_r} "
                                f"outside interval [{lo},{hi}] for edge ({u}->{v})")
                            raise RuntimeError("Changed rank outside edge interval.")

            edges_processed += 1
            if success and delta > 0:
                successes += 1
                if mode.startswith("extended"):
                    extended_successes += 1
                elif mode.startswith("greedy"):
                    greedy_successes += 1

                # log(
                #     f"✅ SUCCESS ({u}->{v}) gain={delta:.4f} mode={mode} "
                #     f"interval=({lo},{hi}) changed={len(changed_scores)}"
                # )
            else:
                failures += 1

            edge_total_time = time.time() - edge_start
            total_edge_processing_time += edge_total_time

            # if edge_total_time > EDGE_WARN:
            #     log(
            #         f"⚠️ Slow edge ({u}->{v}): total={edge_total_time:.6f}s "
            #         f"(between={t_between:.6f}s, apply={t_apply_dur:.6f}s, "
            #         f"greedy={greedy_dur:.6f}s)"
            #     )

            # Send result back
            result_queue.put({
                "worker_idx": worker_idx,
                "u": u,
                "v": v,
                "success": bool(success and delta > 0),
                "delta": float(delta),
                "changed_scores": changed_scores,
                "mode": mode,
                "interval": (lo, hi),
                "timing": {
                    "apply_new_strategy": t_apply_dur,
                    "greedy_time": greedy_dur,
                    "between_list_time": t_between,
                    "edge_total_time": edge_total_time,
                },
            })

        except Exception as e:
            # Catch-all for unexpected runtime errors on this edge
            failures += 1
            edge_total_time = time.time() - edge_start
            total_edge_processing_time += edge_total_time

            log(f"❌ UNHANDLED ERROR while processing edge index {idx} ({u}->{v}): {repr(e)}")
            log(f"   Edge processing aborted; reporting failure with delta=0. "
                f"edge_time={edge_total_time:.6f}s")

            result_queue.put({
                "worker_idx": worker_idx,
                "u": u,
                "v": v,
                "success": False,
                "delta": 0.0,
                "changed_scores": {},
                "mode": "error",
                "interval": (lo, hi),
                "timing": {
                    "apply_new_strategy": 0.0,
                    "greedy_time": 0.0,
                    "between_list_time": 0.0,
                    "edge_total_time": edge_total_time,
                },
            })

    # -----------------------------------------------
    # Worker summary
    # -----------------------------------------------
    runtime = time.time() - worker_start
    lock_fraction = (total_lock_time / runtime) if runtime > 0 else 0.0

    # log(
    #     f"🧮 Worker DONE: processed={edges_processed}, "
    #     f"success={successes} (extended={extended_successes}, "
    #     f"greedy={greedy_successes}), "
    #     f"fail={failures}, runtime={runtime:.2f}s"
    # )

    # log(
    #     f"⏱️ Time stats: queue_wait={total_queue_wait_time:.3f}s, "
    #     f"lock_time={total_lock_time:.3f}s ({lock_fraction * 100:.2f}%), "
    #     f"apply_total={total_apply_strategy_time:.3f}s, "
    #     f"greedy_total={total_greedy_time:.3f}s, "
    #     f"edge_processing_total={total_edge_processing_time:.3f}s"
    # )

    if lock_fraction > LOCK_FRACTION_WARN:
        log(
            f"⚠️ WARNING: High lock contention (unexpected now): "
            f"{lock_fraction * 100:.2f}% of worker time under lock"
        )

    # Final DONE message (with stats so MAIN can also log them if you want)
    result_queue.put({
        "worker_idx": worker_idx,
        "done": True,
        "edges_processed": edges_processed,
        "successes": successes,
        "extended_successes": extended_successes,
        "greedy_successes": greedy_successes,
        "failures": failures,
        "runtime": runtime,
        "timing_totals": {
            "queue_wait": total_queue_wait_time,
            "lock_time": total_lock_time,
            "apply_total": total_apply_strategy_time,
            "greedy_total": total_greedy_time,
            "edge_processing_total": total_edge_processing_time,
        },
    })


# In[11]:


# Global variables for worker processes
G_GLOBAL = None
EDGES_GLOBAL = None
SCORES_BEFORE_GLOBAL = None


def _init_refine_worker(G, edges, scores_before):
    """
    Initializer for multiprocessing workers.
    Each worker gets its own copy of:
      - G (networkx DiGraph)
      - edges (list of edges)
      - scores_before (dict of scores)
    """
    global G_GLOBAL, EDGES_GLOBAL, SCORES_BEFORE_GLOBAL
    G_GLOBAL = G
    EDGES_GLOBAL = edges
    SCORES_BEFORE_GLOBAL = scores_before


#

# In[12]:


def run_dynamic_round(edges,
                      scores,
                      out_edges,
                      in_edges,
                      edges_dict,
                      num_procs,
                      forward_weight,
                      improvement_counter,
                      index_to_node,
                      output_excel,
                      FW_SANITY_EVERY_IMPROVEMENTS=5,
                      edge_subset=None):
    import math
    import time
    import datetime
    import multiprocessing as mp

    def now_str():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    # ----------------- WARN THRESHOLDS -----------------
    BUILD_BACKWARD_WARN = 10.0  # seconds
    DP_PER_EDGE_WARN = 1e-4  # s/edge heuristic for O(E)-ish behavior
    SPAWN_WORKERS_WARN = 5.0  # seconds
    RESULT_LOOP_WARN = 60.0  # seconds
    JOIN_WARN = 5.0  # seconds
    ROUND_TIME_WARN = 600.0  # seconds (10 minutes)
    AVG_WORKER_EDGE_WARN = 0.20  # seconds per edge
    LOW_SUCCESS_RATIO_WARN = 0.02  # applied_edges / processed_results
    FW_MISMATCH_TOL = 1e-6  # tolerance for FW mismatch

    # ----------------- HELPER: SAVE RANKING -----------------
    def save_ranking_snapshot(scores_dict, index_to_node, path):
        import pandas as pd

        # Convert {node_index: rank} to rows sorted by rank
        items = sorted(scores_dict.items(), key=lambda kv: kv[1])

        rows = []
        for node_idx, rank in items:
            node_id_str = str(index_to_node[node_idx])
            rows.append({"Node ID": node_id_str, "Order": int(rank)})

        df = pd.DataFrame(rows)

        lower = path.lower()
        if lower.endswith(".csv"):
            df.to_csv(path, index=False)
        else:
            df.to_excel(path, index=False)

    round_start = time.time()
    FW_before_round = forward_weight
    # print(f"[{now_str()}][MAIN] 🚀 START dynamic round (DP-based independent subsets)")
    # print(f"[{now_str()}][MAIN]    FW_before_round={FW_before_round:.2f}")

    # ----------------- TIMING ACCUMULATORS (whole dynamic round) -----------------
    t_build_backward = 0.0
    t_spawn_workers_total = 0.0
    t_result_loop_total = 0.0
    t_join_total = 0.0
    t_dp_total = 0.0

    # Optional aggregation of worker-side timing (if present in res["timing"])
    total_apply_strategy_time_workers = 0.0
    total_greedy_time_workers = 0.0
    total_between_time_workers = 0.0
    total_edge_time_workers = 0.0
    counted_timing_results_total = 0

    total_delta_round = 0.0
    applied_edges_total = 0
    results_processed_total = 0

    # ----- Normalize edge_subset → subset_pairs of (u,v) and log sanity -----
    subset_pairs = None
    if edge_subset is not None:
        raw_len = len(edge_subset)
        norm_set = set()
        malformed = 0

        for e in edge_subset:
            # e might be (u,v), (u,v,w), or longer
            if isinstance(e, tuple) and len(e) >= 2:
                norm_set.add((e[0], e[1]))
            else:
                malformed += 1

        subset_pairs = norm_set
        # print(
        #     f"[{now_str()}][MAIN] 🔧 edge_subset provided: "
        #     f"raw_len={raw_len}, normalized_len={len(subset_pairs)}, malformed_entries={malformed}"
        # )

        # Sanity: sample a few normalized subset edges and check if they exist in the graph
        sample_pairs = list(subset_pairs)[:10]
        for (su, sv) in sample_pairs:
            exists = (su, sv) in edges_dict
            # print(
            #     f"[{now_str()}][MAIN]    subset sample edge ({su}->{sv}) "
            #     f"exists_in_graph={exists}"
            # )

    # ----- Build backward edges for this dynamic sweep -----
    t0 = time.time()

    backward_edges = []  # list of (u, v, w, lo, hi)
    skipped_by_subset = 0
    skipped_not_backward = 0

    for (u, v, w) in edges:
        if subset_pairs is not None and (u, v) not in subset_pairs:
            skipped_by_subset += 1
            continue
        if scores[u] > scores[v]:
            ru, rv = scores[u], scores[v]
            lo = min(ru, rv)
            hi = max(ru, rv)
            backward_edges.append((u, v, w, lo, hi))
        else:
            skipped_not_backward += 1

    # Sort by weight if you want heavier edges to be “seen” earlier,
    # note: DP itself will ignore this ordering and work on endpoints.
    backward_edges.sort(key=lambda x: x[2], reverse=True)
    n_edges = len(backward_edges)
    t_build_backward = time.time() - t0

    # print(f"[{now_str()}][MAIN] Built {n_edges} backward edges (sorted). "
    #       f"(skipped_by_subset={skipped_by_subset}, skipped_not_backward={skipped_not_backward}, "
    #       f"time={t_build_backward:.4f}s)")
    if t_build_backward > BUILD_BACKWARD_WARN:
        print(f"[{now_str()}][MAIN] ⚠️ WARNING: Building backward edges took {t_build_backward:.4f}s "
              f"(>{BUILD_BACKWARD_WARN}s)")

    if n_edges == 0:
        total_round_time = time.time() - round_start
        # print(f"[{now_str()}][MAIN] ❗ No backward edges; ending dynamic round. "
        #       f"total_round_time={total_round_time:.4f}s")
        # print(f"[{now_str()}][MAIN]    FW_after_round={forward_weight:.2f} "
        #       f"(ΔFW_round={forward_weight - FW_before_round:.3f})")
        return scores, forward_weight, 0.0, 0, improvement_counter

    # -------- shared structures for ALL DP batches in this dynamic round --------
    # NOTE: workers do not need conflicts anymore, but we keep the signature.
    manager = mp.Manager()
    active_intervals = manager.list()
    used_intervals = manager.list()
    lock = manager.Lock()

    scores_snapshot = scores.copy()

    # We will iteratively pick independent subsets via DP until all backward_edges
    # are "assessed" (either tried or skipped) in this dynamic round.
    remaining_indices = list(range(n_edges))
    batch_idx = 0

    while remaining_indices:
        batch_idx += 1
        dp_round_start = time.time()

        edges_in_pool = len(remaining_indices)
        # print(f"[{now_str()}][MAIN] 🔂 DP batch {batch_idx}: "
        #       f"edges_in_pool_before_batch={edges_in_pool}")

        # --- Build local list of edges for DP ---
        edges_for_dp = [backward_edges[i] for i in remaining_indices]

        # --- DP to select a maximum non-conflicting subset ---
        t_dp0 = time.time()
        local_selected = select_nonconflicting_edge_indices_dp(edges_for_dp)
        t_dp1 = time.time()
        dp_time = t_dp1 - t_dp0
        t_dp_total += dp_time

        E = len(edges_for_dp)
        if E > 0:
            dp_time_per_edge = dp_time / E
        else:
            dp_time_per_edge = float('nan')

        selected_global = [remaining_indices[i] for i in local_selected]
        num_selected = len(selected_global)

        # print(
        #     f"[{now_str()}][MAIN] (DP batch {batch_idx}) "
        #     f"DP selection: total_edges_in_pool={E}, selected_nonconflicting={num_selected}"
        # )
        # print(
        #     f"[{now_str()}][MAIN] (DP batch {batch_idx}) "
        #     f"DP time={dp_time:.4f}s (≈ {dp_time_per_edge:.3e}s per edge)"
        # )

        # O(E) compatibility check
        # if not math.isnan(dp_time_per_edge):
        #     if dp_time_per_edge <= DP_PER_EDGE_WARN:
        #         print(
        #             f"[{now_str()}][MAIN] (DP batch {batch_idx}) "
        #             f"O(E) check: dp_time_per_edge={dp_time_per_edge:.3e}s "
        #             f"→ compatible with linear-time behavior."
        #         )
        #     else:
        #         print(
        #             f"[{now_str()}][MAIN] (DP batch {batch_idx}) "
        #             f"⚠️ O(E) check: dp_time_per_edge={dp_time_per_edge:.3e}s "
        #             f"→ higher than expected; may be environment/constants."
        #         )

        if num_selected == 0:
            # print(
            #     f"[{now_str()}][MAIN] (DP batch {batch_idx}) ❗ DP did not select any edge; "
            #     f"marking remaining {len(remaining_indices)} edges as assessed with no moves."
            # )
            remaining_indices = []
            break

        # --- Build batch_edges list for workers ---
        batch_edges = [backward_edges[i] for i in selected_global]
        num_edges_batch = len(batch_edges)
        num_procs_effective = min(num_procs, num_edges_batch)

        # print(f"[{now_str()}][MAIN] (DP batch {batch_idx}) "
        #       f"Spawning {num_procs_effective} workers for {num_edges_batch} edges.")

        # Separate queues per batch
        edge_queue = manager.Queue()
        result_queue = manager.Queue()

        for idx in range(num_edges_batch):
            edge_queue.put(idx)

        # --- Spawn workers for this independent subset ---
        t_spawn0 = time.time()
        procs = []
        for worker_idx in range(1, num_procs_effective + 1):
            p = mp.Process(
                target=worker_loop,
                args=(
                    worker_idx,
                    num_procs_effective,
                    batch_edges,
                    scores_snapshot,
                    out_edges,
                    in_edges,
                    edges_dict,
                    active_intervals,
                    used_intervals,
                    lock,
                    edge_queue,
                    result_queue,
                )
            )
            p.start()
            procs.append(p)
            # print(f"[{now_str()}][MAIN] (DP batch {batch_idx}) "
            #       f"Worker {worker_idx}/{num_procs_effective} → PID={p.pid}")
        t_spawn_workers = time.time() - t_spawn0
        t_spawn_workers_total += t_spawn_workers

        # print(f"[{now_str()}][MAIN] (DP batch {batch_idx}) "
        #       f"Finished spawning workers in {t_spawn_workers:.4f}s")
        if t_spawn_workers > SPAWN_WORKERS_WARN:
            print(f"[{now_str()}][MAIN] ⚠️ WARNING: Spawning workers for DP batch {batch_idx} "
                  f"took {t_spawn_workers:.4f}s (>{SPAWN_WORKERS_WARN}s)")

        # --- Collect results for this batch ---
        t_res0 = time.time()
        alive = num_procs_effective
        total_delta_batch = 0.0
        applied_edges_batch = 0
        results_processed_batch = 0

        # worker timing for this batch
        batch_apply_time = 0.0
        batch_greedy_time = 0.0
        batch_between_time = 0.0
        batch_edge_total_time = 0.0
        counted_timing_results_batch = 0

        last_progress_log = time.time()
        FW_before_batch = forward_weight

        while alive > 0:
            res = result_queue.get()
            # NOTE: we count *all* messages, but "results_processed_batch" will only
            # count edge-result messages, not "done" messages.
            if res.get("done"):
                alive -= 1
                continue

            results_processed_batch += 1
            results_processed_total += 1

            # Aggregate worker timing if present
            timing = res.get("timing")
            if timing:
                ta = timing.get("apply_new_strategy", 0.0)
                tg = timing.get("greedy_time", 0.0)
                tb = timing.get("between_list_time", 0.0)
                te = timing.get("edge_total_time", 0.0)

                batch_apply_time += ta
                batch_greedy_time += tg
                batch_between_time += tb
                batch_edge_total_time += te
                counted_timing_results_batch += 1

                total_apply_strategy_time_workers += ta
                total_greedy_time_workers += tg
                total_between_time_workers += tb
                total_edge_time_workers += te
                counted_timing_results_total += 1

            if not res["success"]:
                # still count for rate stats, but no FW update
                if time.time() - last_progress_log > 5.0:
                    remaining_edges_pool = len(remaining_indices)
                    avg_edge_time_batch = (
                        batch_edge_total_time / counted_timing_results_batch
                        if counted_timing_results_batch > 0 else float('nan')
                    )
                    # print(
                    #     f"[{now_str()}][MAIN] (DP batch {batch_idx}) Progress: "
                    #     f"results_processed_batch={results_processed_batch}, "
                    #     f"remaining_edges_pool={remaining_edges_pool}, "
                    #     f"applied_edges_batch={applied_edges_batch}, "
                    #     f"alive_workers={alive}, "
                    #     f"avg_worker_edge_time_batch={avg_edge_time_batch:.4f}s"
                    # )
                    last_progress_log = time.time()
                continue

            # Success: apply changes
            delta = res["delta"]
            changed = res["changed_scores"]
            u = res["u"]
            v = res["v"]
            mode = res["mode"]

            for node, nrank in changed.items():
                scores[node] = nrank

            forward_weight += delta
            total_delta_batch += delta
            total_delta_round += delta
            applied_edges_batch += 1
            applied_edges_total += 1
            improvement_counter += 1

            # print(f"[{now_str()}][MAIN] (DP batch {batch_idx}) ✅ SUCCESS ({u}->{v}) "
            #       f"Δ={delta:.3f} mode={mode} FW={forward_weight:.2f}")

            # Periodic sanity check + checkpoint SAVE (per improvements)
            if FW_SANITY_EVERY_IMPROVEMENTS > 0 and \
                    improvement_counter % FW_SANITY_EVERY_IMPROVEMENTS == 0:

                t_fw0 = time.time()
                real_fw = compute_forward_weight(edges, scores)
                t_fw = time.time() - t_fw0
                # print(f"[{now_str()}][MAIN] 🔍 SANITY FW={real_fw} tracked={forward_weight} "
                #       f"(recompute_time={t_fw:.4f}s)")

                if abs(real_fw - forward_weight) > FW_MISMATCH_TOL:
                    print(f"[{now_str()}][MAIN] ❌ RUNTIME ERROR: Forward-weight mismatch before save! "
                          f"tracked={forward_weight:.6f}, recomputed={real_fw:.6f}, "
                          f"tol={FW_MISMATCH_TOL}")
                    raise RuntimeError("Forward-weight mismatch detected before saving ranking.")

                # sync tracked FW
                forward_weight = real_fw

                # safe to save snapshot
                try:
                    save_ranking_snapshot(scores, index_to_node, output_excel)
                # print(f"[{now_str()}][MAIN] 💾 Saved intermediate ranking to {output_excel}")
                except Exception as e:
                    print(f"[{now_str()}][MAIN] ⚠️ WARNING: Failed to save intermediate ranking "
                          f"to {output_excel}: {repr(e)}")

            # Periodic progress log independent of sanity FW
            if time.time() - last_progress_log > 5.0:
                avg_edge_time_batch = (
                    batch_edge_total_time / counted_timing_results_batch
                    if counted_timing_results_batch > 0 else float('nan')
                )
                # print(
                #     f"[{now_str()}][MAIN] (DP batch {batch_idx}) Progress: "
                #     f"results_processed_batch={results_processed_batch}, "
                #     f"applied_edges_batch={applied_edges_batch}, "
                #     f"total_delta_batch={total_delta_batch:.3f}, "
                #     f"alive_workers={alive}, "
                #     f"avg_worker_edge_time_batch={avg_edge_time_batch:.4f}s"
                # )
                last_progress_log = time.time()

        t_result_loop = time.time() - t_res0
        t_result_loop_total += t_result_loop

        # print(f"[{now_str()}][MAIN] (DP batch {batch_idx}) Result collection loop finished in "
        #       f"{t_result_loop:.4f}s (results_processed_batch={results_processed_batch}, "
        #       f"applied_edges_batch={applied_edges_batch})")
        if t_result_loop > RESULT_LOOP_WARN:
            print(f"[{now_str()}][MAIN] ⚠️ WARNING: Result collection loop for DP batch {batch_idx} "
                  f"took {t_result_loop:.4f}s (>{RESULT_LOOP_WARN}s)")

        # join workers for this batch
        t_join0 = time.time()
        for p in procs:
            p.join()
        t_join = time.time() - t_join0
        t_join_total += t_join

        #      print(f"[{now_str()}][MAIN] (DP batch {batch_idx}) Joined all workers in {t_join:.4f}s")
        if t_join > JOIN_WARN:
            print(f"[{now_str()}][MAIN] ⚠️ WARNING: Joining workers for DP batch {batch_idx} "
                  f"took {t_join:.4f}s (>{JOIN_WARN}s)")

        # --- Remove selected edges from remaining pool (they're assessed now) ---
        selected_global_set = set(selected_global)
        remaining_before = len(remaining_indices)
        remaining_indices = [idx for idx in remaining_indices if idx not in selected_global_set]
        # print(
        #     f"[{now_str()}][MAIN] (DP batch {batch_idx}) "
        #     f"Removed {len(selected_global_set)} edges from pool; "
        #     f"remaining_after_batch={len(remaining_indices)} (was {remaining_before})"
        # )

        # --- Batch summary (timing + FW delta) ---
        batch_total_time = time.time() - dp_round_start

        avg_worker_edge_time_batch = (
            batch_edge_total_time / counted_timing_results_batch
            if counted_timing_results_batch > 0 else float('nan')
        )

        FW_after_batch = forward_weight

        # print(f"[{now_str()}][MAIN] (DP batch {batch_idx}) 🧮 BATCH SUMMARY:")
        # print(f"    edges_in_batch={num_edges_batch}, "
        #       f"applied_edges_batch={applied_edges_batch}, "
        #       f"total_delta_batch={total_delta_batch:.3f}")
        # print(f"    FW_before_batch={FW_before_batch:.2f}, "
        #       f"FW_after_batch={FW_after_batch:.2f}, "
        #       f"ΔFW_batch={FW_after_batch - FW_before_batch:.3f}")
        # print(f"    timing_dp={dp_time:.4f}s")
        # print(f"    timing_spawn_workers={t_spawn_workers:.4f}s")
        # print(f"    timing_result_loop={t_result_loop:.4f}s")
        # print(f"    timing_join={t_join:.4f}s")
        # print(f"    batch_total_time={batch_total_time:.4f}s")
        # print(f"    worker_timing_batch: apply_strategy≈{batch_apply_time:.4f}s, "
        #       f"greedy≈{batch_greedy_time:.4f}s, between≈{batch_between_time:.4f}s, "
        #       f"edge_total≈{batch_edge_total_time:.4f}s, "
        #       f"avg_worker_edge_time_batch≈{avg_worker_edge_time_batch:.4f}s "
        #       f"over {counted_timing_results_batch} results")

    # --------------- DYNAMIC ROUND SUMMARY (over all DP batches) ---------------
    total_round_time = time.time() - round_start
    avg_gain = (total_delta_round / applied_edges_total) if applied_edges_total > 0 else 0.0
    edges_per_sec = (applied_edges_total / total_round_time) if total_round_time > 0 else 0.0
    avg_worker_edge_time = (
        total_edge_time_workers / counted_timing_results_total
        if counted_timing_results_total > 0 else float('nan')
    )

    FW_after_round = forward_weight

    # print(f"[{now_str()}][MAIN] 🧮 DYNAMIC ROUND SUMMARY:")
    # print(f"    FW_before_round={FW_before_round:.2f}, "
    #       f"FW_after_round={FW_after_round:.2f}, "
    #       f"ΔFW_round={FW_after_round - FW_before_round:.3f}")
    # print(f"    total_delta_round={total_delta_round:.3f}")
    # print(f"    applied_edges_total={applied_edges_total}, "
    #       f"avg_gain_per_edge={avg_gain:.4f}")
    # print(f"    total_round_time={total_round_time:.4f}s, "
    #       f"applied_edges_per_sec={edges_per_sec:.2f}")
    # print(f"    timing_build_backward={t_build_backward:.4f}s")
    # print(f"    timing_dp_total={t_dp_total:.4f}s")
    # print(f"    timing_spawn_workers_total={t_spawn_workers_total:.4f}s")
    # print(f"    timing_result_loop_total={t_result_loop_total:.4f}s")
    # print(f"    timing_join_total={t_join_total:.4f}s")
    # print(f"    worker_timing_total: apply_strategy≈{total_apply_strategy_time_workers:.4f}s, "
    #       f"greedy≈{total_greedy_time_workers:.4f}s, "
    #       f"between≈{total_between_time_workers:.4f}s, "
    #       f"edge_total≈{total_edge_time_workers:.4f}s, "
    #       f"avg_worker_edge_time≈{avg_worker_edge_time:.4f}s "
    #       f"over {counted_timing_results_total} results")

    # --------------- WARNINGS BASED ON SUMMARY ---------------
    if total_round_time > ROUND_TIME_WARN:
        print(f"[{now_str()}][MAIN] ⚠️ WARNING: Dynamic round took {total_round_time:.4f}s "
              f"(>{ROUND_TIME_WARN}s)")

    if not math.isnan(avg_worker_edge_time) and avg_worker_edge_time > AVG_WORKER_EDGE_WARN:
        print(f"[{now_str()}][MAIN] ⚠️ WARNING: avg_worker_edge_time={avg_worker_edge_time:.4f}s "
              f"(>{AVG_WORKER_EDGE_WARN}s)")

    if results_processed_total > 0:
        success_ratio = applied_edges_total / results_processed_total
        if success_ratio < LOW_SUCCESS_RATIO_WARN:
            print(f"[{now_str()}][MAIN] ⚠️ WARNING: Very low success ratio in this dynamic round: "
                  f"{success_ratio:.4%} (applied_edges_total={applied_edges_total}, "
                  f"results_processed_total={results_processed_total})")

    #   print(f"[{now_str()}][MAIN] ✅ END dynamic round (DP-based independent subsets)")

    return scores, forward_weight, total_delta_round, applied_edges_total, improvement_counter


# In[13]:


import csv


def save_scores_to_csv(scores, output_path, index_to_node):
    """
    Saves scores to CSV in the exact format required by load_initial_scores():
    columns = ['Node ID', 'Order'].
    Values:
        Node ID = original node label (string)
        Order   = rank (int)
    Rows sorted by Order.
    """

    # Build list of (order, node_index)
    rows = [(order, node_idx) for node_idx, order in scores.items()]
    rows.sort(key=lambda x: x[0])  # sort by Order

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Node ID", "Order"])  # required header

        for order, node_idx in rows:
            node_label = index_to_node[node_idx]  # original ID string
            writer.writerow([node_label, int(order)])


#

# In[14]:


def refine_ranking_parallel_dynamic(
        csv_path,
        initial_ranking_path,
        output_excel,
        MAX_HOURS: float = 12.0,
        LOG_EVERY_ROUND: int = 1,
        FW_CHECK_EVERY_ROUND: int = 5,
        edge_subset=None,  # None => global; otherwise list of edges; we normalize to (u,v)
):
    """
    Dynamic parallel backward refinement with multiple DP rounds over a fixed pool
    of candidate edges.

    Key behavior:
      - Build a fixed candidate list of edges (possibly a subset).
      - Maintain remaining_indices over this list.
      - Each DP round:
          * Take a fresh snapshot of scores.
          * Among remaining edges, keep those that are backward in snapshot and
            build (u,v,w,lo,hi).
          * Use DP to choose a maximum non-conflicting subset by [lo,hi].
          * Give only that subset to workers with THIS snapshot.
          * Collect results, apply changed_scores globally (sync point).
          * Remove those edges from remaining_indices (they're assessed).
      - Stop when:
          * no remaining backward edges, OR
          * DP selects zero edges, OR
          * global time limit (MAX_HOURS), OR
          * NO IMPROVEMENT for 10 minutes (wall clock).

    RETURN:
      best_scores, final_fw, remaining_pairs

      where:
        - best_scores: dict node -> rank of the best FW seen
        - final_fw: forward weight of best_scores
        - remaining_pairs: set of (u, v) for edges in the candidate pool that
          were never assessed (never selected in any DP batch).
    """

    import math
    import time
    import datetime
    import os
    import multiprocessing as mp

    pid_main = os.getpid()

    try:
        num_procs = len(os.sched_getaffinity(0))
    except Exception:
        num_procs = os.cpu_count() or 1

    def now_str():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(msg):
        print(f"[{now_str()}][PID {pid_main}][PROCS {num_procs}] {msg}")

    # ---------------------------------------------------
    # Helper: DP independence checker
    # ---------------------------------------------------
    def check_dp_interval_independence(edges_for_dp, local_selected, round_idx):
        """
        edges_for_dp: list of (u,v,w,lo,hi) for remaining edges (for THIS snapshot)
        local_selected: indices into edges_for_dp returned by DP
        Ensures selected intervals [lo,hi] are strictly non-overlapping.
        """
        if not local_selected:
            log(f"[MAIN] (Round {round_idx}) DP independence check: no edges selected.")
            return

        tmp = []
        for li in local_selected:
            try:
                u, v, w, lo, hi = edges_for_dp[li]
            except Exception as e:
                log(
                    f"[MAIN] ❌ INTERNAL ERROR in DP independence check: "
                    f"cannot unpack edge at local_idx={li}: {repr(e)}"
                )
                raise
            tmp.append((lo, hi, li, u, v))

        tmp.sort(key=lambda x: x[0])  # sort by lo

        last_lo, last_hi, last_li, last_u, last_v = tmp[0]
        # Intervals must be strictly disjoint: hi_prev < lo_curr
        # for i in range(1, len(tmp)):
        #     lo, hi, li, u, v = tmp[i]
        #     if lo <= last_hi:
        #         log(
        #             f"[MAIN] ❌ DP INDEPENDENCE VIOLATION in round {round_idx}: intervals overlap.\n"
        #             f"    Edge A (local_idx={last_li}, edge=({last_u}->{last_v}), "
        #             f"interval=[{last_lo},{last_hi}])\n"
        #             f"    Edge B (local_idx={li}, edge=({u}->{v}), interval=[{lo},{hi}])"
        #         )
        #         raise RuntimeError("DP selected overlapping intervals; independence broken.")
        #     if hi > last_hi:
        #         last_lo, last_hi, last_li, last_u, last_v = lo, hi, li, u, v
        #
        # log(
        #     f"[MAIN] (Round {round_idx}) DP independence check PASSED: "
        #     f"{len(local_selected)} intervals, no overlaps."
        # )

    # log("🚀 Starting DYNAMIC parallel backward refinement (multi-DP rounds over fixed edge pool)")
    # log(f"📥 Graph: {csv_path}")
    # log(f"📥 Initial ranking: {initial_ranking_path}")
    # log(f"💾 Output ranking: {output_excel}")
    # log(f"🧵 Number of available processors (os.cpu_count / sched_getaffinity) = {num_procs}")

    # ---------------------------------------------------
    # Load graph & initial scores
    # ---------------------------------------------------
    # log("🔧 Reading graph...")
    edges, node_to_index, index_to_node = read_graph(csv_path)
    # log(f"   Graph loaded: n_nodes={len(node_to_index)}, n_edges={len(edges)}")

    # log("🔧 Loading initial scores/ranking...")
    scores = load_initial_scores(initial_ranking_path, node_to_index)
    # log(f"   Loaded scores for {len(scores)} nodes.")

    fw_total = compute_forward_weight(edges, scores)
    total_weight = sum(w for (_, _, w) in edges)
    fw_ratio = fw_total / total_weight if total_weight > 0 else 0.0
    # log(f"✅ Initial FW / total = {fw_total:.2f} / {total_weight:.2f} = {fw_ratio:.6f}")
    #
    # # ---------------------------------------------------
    # # Build adjacency & edges_dict
    # # ---------------------------------------------------
    # log("🔧 Building adjacency lists and edges_dict...")
    out_edges = {}
    in_edges = {}
    edges_dict = {}
    for (u, v, w) in edges:
        out_edges.setdefault(u, []).append((v, w))
        in_edges.setdefault(v, []).append((u, w))
        edges_dict[(u, v)] = w
    # log(
    #     f"   Adjacency built: len(out_edges)={len(out_edges)}, "
    #     f"len(in_edges)={len(in_edges)}, len(edges_dict)={len(edges_dict)}"
    # )

    # ---------------------------------------------------
    # Normalize edge_subset → subset_pairs of form (u,v)
    # ---------------------------------------------------
    subset_pairs = None
    if edge_subset is not None:
        raw_len = len(edge_subset)
        norm_set = set()
        malformed = 0

        for e in edge_subset:
            # e might be (u,v), (u,v,extra), etc.
            if isinstance(e, tuple):
                if len(e) >= 2:
                    norm_set.add((e[0], e[1]))
                else:
                    malformed += 1
            else:
                malformed += 1

        subset_pairs = norm_set
        # log(
        #     f"🔧 Initial subset provided: raw_len={raw_len}, "
        #     f"normalized_len={len(subset_pairs)}, malformed_entries={malformed}"
        # )

        # Sanity: sample a few normalized subset edges and check if they exist in the graph
        sample_pairs = list(subset_pairs)[:10]
        for (su, sv) in sample_pairs:
            exists = (su, sv) in edges_dict
            log(f"[MAIN] subset sample edge ({su}->{sv}) exists_in_graph={exists}")

    # ---------------------------------------------------
    # Fixed candidate pool of edges for the whole run
    # ---------------------------------------------------
    candidate_edges = []  # list of (u, v, w)
    for (u, v, w) in edges:
        if subset_pairs is not None and (u, v) not in subset_pairs:
            continue
        candidate_edges.append((u, v, w))

    #  log(f"[MAIN] Candidate edge pool size = {len(candidate_edges)} (after subset filter).")

    # if subset_pairs is not None and len(candidate_edges) == 0:
    #     log("[MAIN] ❌ ERROR: edge_subset (after normalization) is non-empty but "
    #         "no edges from the graph matched it.")
    #     log("[MAIN]      Likely node-ID vs index mismatch or wrong edge directions.")
    #     raise RuntimeError("edge_subset does not match any edges in this graph.")

    # remaining_indices points into candidate_edges (fixed list)
    remaining_indices = list(range(len(candidate_edges)))

    best_scores = scores.copy()
    best_FW = fw_total
    FW_tracked = fw_total
    #   log(f"⭐ Initial best FW set to {best_FW:.2f}")

    start_ts = time.time()
    deadline = start_ts + MAX_HOURS * 3600.0

    # NEW: time-based no-improvement stopping rule
    NO_IMPROVEMENT_TIME_LIMIT_SEC = 600.0  # 10 minutes without improvement
    last_improvement_time = start_ts

    FW_MISMATCH_TOL = 1e-6
    DP_PER_EDGE_WARN = 1e-4  # heuristic threshold for O(E)-ish behavior

    round_idx = 0

    # ===================================================
    # MAIN LOOP: one DP round per iteration
    # ===================================================
    while remaining_indices:
        now = time.time()
        if now >= deadline:
            log("⏰ Time limit reached before starting new round; stopping.")
            break

        round_idx += 1
        elapsed_h = (now - start_ts) / 3600.0
        time_left_h = max(0.0, (deadline - now) / 3600.0)
        since_improvement = now - last_improvement_time

        # Track whether this round produces a *global* improvement
        improved_this_round = False

        # log("")
        # log(f"[MAIN] 🔁 ===== Dynamic DP round {round_idx} =====")
        # log(
        #     f"[MAIN]    Current FW(tracked)={FW_tracked:.2f}, best_FW={best_FW:.2f}, "
        #     f"elapsed={elapsed_h:.2f}h, time_left={time_left_h:.2f}h"
        # )
        # log(f"[MAIN]    Remaining candidate edges: {len(remaining_indices)}")
        # log(
        #     f"[MAIN]    Time since last improvement: {since_improvement:.1f}s "
        #     f"(limit={NO_IMPROVEMENT_TIME_LIMIT_SEC:.0f}s)"
        # )

        # ---------------------------------------------------
        # Fresh snapshot for THIS DP round
        # ---------------------------------------------------
        scores_snapshot = scores.copy()
        # log(
        #     f"[MAIN] Snapshot for round {round_idx} captured "
        #     f"(FW_tracked={FW_tracked:.2f}). All workers in this round share it."
        # )

        # ---------------------------------------------------
        # Build edges_for_dp from remaining_indices using THIS snapshot
        # ---------------------------------------------------
        edges_for_dp = []  # (u, v, w, lo, hi) for backward edges
        global_idx_for_local = []  # maps local index -> global index in candidate_edges

        num_non_backward = 0
        for gi in remaining_indices:
            u, v, w = candidate_edges[gi]
            ru = scores_snapshot[u]
            rv = scores_snapshot[v]
            if ru > rv:
                lo = rv
                hi = ru
                edges_for_dp.append((u, v, w, lo, hi))
                global_idx_for_local.append(gi)
            else:
                num_non_backward += 1

        # log(
        #     f"[MAIN] (Round {round_idx}) Among remaining {len(remaining_indices)} edges, "
        #     f"{len(edges_for_dp)} are backward in this snapshot, "
        #     f"{num_non_backward} are non-backward."
        # )

        if not edges_for_dp:
            log(
                f"[MAIN] (Round {round_idx}) ❗ No backward edges among remaining candidates. "
                f"Stopping refinement."
            )
            break

        # ---------------------------------------------------
        # DP selection on edges_for_dp
        # ---------------------------------------------------
        t_dp0 = time.time()
        local_selected = select_nonconflicting_edge_indices_dp(edges_for_dp)
        t_dp1 = time.time()
        dp_time = t_dp1 - t_dp0
        E = len(edges_for_dp)
        num_selected_local = len(local_selected)

        dp_time_per_edge = dp_time / E if E > 0 else float('nan')

        # log(
        #     f"[MAIN] (Round {round_idx}) DP selection: "
        #     f"total_backward_edges={E}, selected_nonconflicting={num_selected_local}"
        # )
        # log(
        #     f"[MAIN] (Round {round_idx}) DP time={dp_time:.4f}s "
        #     f"(≈ {dp_time_per_edge:.3e}s per edge)"
        # )

        # O(E) compatibility check
        # if not math.isnan(dp_time_per_edge):
        #     if dp_time_per_edge <= DP_PER_EDGE_WARN:
        #         log(
        #             f"[MAIN] (Round {round_idx}) "
        #             f"O(E) check: dp_time_per_edge={dp_time_per_edge:.3e}s "
        #             f"→ compatible with linear-time behavior."
        #         )
        #     else:
        #         log(
        #             f"[MAIN] (Round {round_idx}) "
        #             f"⚠️ O(E) check: dp_time_per_edge={dp_time_per_edge:.3e}s "
        #             f"→ higher than expected; may be environment/constants."
        #         )

        if num_selected_local == 0:
            log(
                f"[MAIN] (Round {round_idx}) ❗ DP did not select any edge from "
                f"{E} backward candidates; stopping refinement."
            )
            break

        # Independence check on this snapshot's intervals
        check_dp_interval_independence(edges_for_dp, local_selected, round_idx)

        # Map local-selected indices to global candidate indices
        selected_global_indices = [global_idx_for_local[li] for li in local_selected]
        batch_edges = [edges_for_dp[li] for li in local_selected]  # (u,v,w,lo,hi)
        num_edges_batch = len(batch_edges)

        # log(
        #     f"[MAIN] (Round {round_idx}) Will assess {num_edges_batch} edges "
        #     f"out of {len(remaining_indices)} remaining."
        # )

        # ------------------ Spawn workers for this batch ------------------
        num_procs_effective = min(num_procs, num_edges_batch)
        # log(
        #     f"[MAIN] (Round {round_idx}) Spawning {num_procs_effective} workers "
        #     f"for {num_edges_batch} selected edges."
        # )

        import multiprocessing as mp  # ensure in scope if moved
        edge_queue = mp.Queue()
        result_queue = mp.Queue()

        for idx in range(num_edges_batch):
            edge_queue.put(idx)

        dummy_manager = mp.Manager()
        active_intervals = dummy_manager.list()
        used_intervals = dummy_manager.list()
        lock = dummy_manager.Lock()

        workers = []
        t_spawn0 = time.time()
        for wi in range(num_procs_effective):
            p = mp.Process(
                target=worker_loop,
                args=(
                    wi + 1,
                    num_procs_effective,
                    batch_edges,
                    scores_snapshot,
                    out_edges,
                    in_edges,
                    edges_dict,
                    active_intervals,
                    used_intervals,
                    lock,
                    edge_queue,
                    result_queue,
                )
            )
            p.daemon = False
            p.start()
            # log(
            #     f"[MAIN] (Round {round_idx}) Worker {wi + 1}/{num_procs_effective} "
            #     f"→ PID={p.pid}"
            # )
            workers.append(p)
        t_spawn1 = time.time()
        t_spawn_workers = t_spawn1 - t_spawn0
        # log(
        #     f"[MAIN] (Round {round_idx}) Finished spawning workers in "
        #     f"{t_spawn_workers:.4f}s"
        # )

        # ------------------ Collect results for this batch ------------------
        t_results0 = time.time()
        alive_workers = num_procs_effective
        results_processed = 0

        agg_apply_strategy = 0.0
        agg_greedy_time = 0.0
        agg_between_time = 0.0
        agg_edge_total_time = 0.0

        used_nodes_this_batch = set()  # node-level independence check
        total_delta_round = 0.0
        applied_edges_round = 0

        last_progress_log = time.time()

        while alive_workers > 0:
            res = result_queue.get()

            if res.get("done"):
                alive_workers -= 1
                # log(
                #     f"[MAIN] (Round {round_idx}) Received DONE from WORKER "
                #     f"{res.get('worker_idx', '?')} "
                #     f"(alive_workers now={alive_workers})"
                # )
                continue

            results_processed += 1
            u = res["u"]
            v = res["v"]
            success = res["success"]
            delta = res["delta"]
            mode = res.get("mode", "none")
            changed_scores = res.get("changed_scores", {})
            timing = res.get("timing", {})

            ta = timing.get("apply_new_strategy", 0.0)
            tg = timing.get("greedy_time", 0.0)
            tb = timing.get("between_list_time", 0.0)
            te = timing.get("edge_total_time", 0.0)
            agg_apply_strategy += ta
            agg_greedy_time += tg
            agg_between_time += tb
            agg_edge_total_time += te

            if success and delta > 0.0:
                changed_nodes = set(changed_scores.keys())
                intersect = used_nodes_this_batch.intersection(changed_nodes)
                # if intersect:
                #     log(
                #         f"[MAIN] ❌ NODE-LEVEL INDEPENDENCE VIOLATION in round {round_idx}: "
                #         f"nodes changed by multiple edges in same DP batch.\n"
                #         f"    Example nodes: {list(intersect)[:10]}"
                #     )
                #     raise RuntimeError(
                #         "DP independence violated at node level (changed_scores)."
                #     )

                used_nodes_this_batch.update(changed_nodes)

                applied_edges_round += 1
                total_delta_round += delta
                FW_tracked += delta

                for node, new_r in changed_scores.items():
                    scores[node] = new_r

                last_improvement_time = time.time()

                # log(
                #     f"[MAIN] (Round {round_idx}) ✅ SUCCESS ({u}->{v}) "
                #     f"Δ={delta:.3f} mode={mode} "
                #     f"changed_nodes={len(changed_scores)}"
                # )
            else:
                # no improvement; just progress logging
                if time.time() - last_progress_log > 5.0:
                    avg_edge_t = (
                        agg_edge_total_time / results_processed
                        if results_processed > 0 else float('nan')
                    )
                    # log(
                    #     f"[MAIN] (Round {round_idx}) Progress: "
                    #     f"results_processed={results_processed}, "
                    #     f"applied_edges_round={applied_edges_round}, "
                    #     f"alive_workers={alive_workers}, "
                    #     f"avg_worker_edge_time≈{avg_edge_t:.4f}s"
                    # )
                    last_progress_log = time.time()

        t_results1 = time.time()
        t_result_loop = t_results1 - t_results0
        # log(
        #     f"[MAIN] (Round {round_idx}) Result collection finished in "
        #     f"{t_result_loop:.4f}s "
        #     f"(results_processed={results_processed}, "
        #     f"applied_edges_round={applied_edges_round})"
        # )

        # Join workers
        t_join0 = time.time()
        for p in workers:
            p.join()
        t_join1 = time.time()
        t_join = t_join1 - t_join0
        # log(
        #     f"[MAIN] (Round {round_idx}) Joined all workers in {t_join:.4f}s"
        # )

        # Remove selected edges from remaining_indices (they are now "assessed")
        selected_global_set = set(selected_global_indices)
        old_remaining_count = len(remaining_indices)
        remaining_indices = [idx for idx in remaining_indices if idx not in selected_global_set]
        # log(
        #     f"[MAIN] (Round {round_idx}) Removed {len(selected_global_set)} edges from remaining; "
        #     f"remaining now = {len(remaining_indices)} (was {old_remaining_count})."
        # )

        # Batch-level summary
        avg_worker_edge_time = (
            agg_edge_total_time / results_processed
            if results_processed > 0 else float('nan')
        )

        # log(
        #     f"[MAIN] Round {round_idx} summary: "
        #     f"total_delta_round={total_delta_round:.3f}, "
        #     f"applied_edges_round={applied_edges_round}"
        # )
        # log(
        #     f"[MAIN] Round {round_idx} worker_timing: "
        #     f"apply_strategy≈{agg_apply_strategy:.4f}s, "
        #     f"greedy≈{agg_greedy_time:.4f}s, "
        #     f"between≈{agg_between_time:.4f}s, "
        #     f"edge_total≈{agg_edge_total_time:.4f}s, "
        #     f"avg_worker_edge_time≈{avg_worker_edge_time:.4f}s "
        #     f"over {results_processed} results"
        # )

        # ---------- Update global best & decide whether to save ----------
        if FW_tracked > best_FW:
            best_FW = FW_tracked
            best_scores = scores.copy()
            improved_this_round = True
            # log(
            #     f"[MAIN]    ⬆️ New best FW={best_FW:.2f} "
            #     f"(updated best_scores after round {round_idx})"
            # )
      #  else:
        # log(
        #     f"[MAIN]    No new global best in round {round_idx} "
        #     f"(FW_tracked={FW_tracked:.2f}, best_FW={best_FW:.2f})"
        # )

        # Save intermediate ranking ONLY if this round improved global best
        if improved_this_round:
            save_scores_to_csv(best_scores, output_excel, index_to_node)
            # log(
            #     f"[MAIN] 💾 Saved intermediate BEST ranking (round {round_idx}, "
            #     f"FW={best_FW:.2f}) to {output_excel}"
            # )
        # else:
        # log(
        #     f"[MAIN] 💾 Skipped saving intermediate ranking (no FW improvement "
        #     f"in round {round_idx})"
        # )

        # FW sanity check every FW_CHECK_EVERY_ROUND rounds
        if FW_CHECK_EVERY_ROUND > 0 and (round_idx % FW_CHECK_EVERY_ROUND == 1):
            t_fw0 = time.time()
            fw_recompute = compute_forward_weight(edges, scores)
            t_fw1 = time.time()
            # if abs(fw_recompute - FW_tracked) > FW_MISMATCH_TOL:
            #     log(
            #         f"[MAIN] ❌ SANITY MISMATCH after round {round_idx}: "
            #         f"FW_recomputed={fw_recompute:.6f}, "
            #         f"FW_tracked={FW_tracked:.6f}"
            #     )
            #     raise RuntimeError("Forward-weight tracking mismatch.")
            # else:
            #     log(
            #         f"[MAIN] 🔍 SANITY after round {round_idx}: "
            #         f"FW_recomputed={fw_recompute:.6f}, "
            #         f"FW_tracked={FW_tracked:.6f}, "
            #         f"(recompute_time={t_fw1 - t_fw0:.4f}s)"
            #     )

        # --- New: time-based no-improvement stop (10 minutes) ---
        no_improvement_elapsed = time.time() - last_improvement_time
        if no_improvement_elapsed >= NO_IMPROVEMENT_TIME_LIMIT_SEC:
            # log(
            #     f"[MAIN] 🔚 Stopping dynamic refinement after round {round_idx}: "
            #     f"no improvement for {no_improvement_elapsed:.1f}s "
            #     f"(limit={NO_IMPROVEMENT_TIME_LIMIT_SEC:.0f}s)."
            # )
            break

        if not remaining_indices:
            log(f"[MAIN] All candidate edges have been assessed; stopping.")
            break

    # ---------------------------------------------------
    # Final write of best_scores
    # ---------------------------------------------------
    save_scores_to_csv(best_scores, output_excel, index_to_node)
    final_fw = compute_forward_weight(edges, best_scores)
    # log(
    #     f"[MAIN] ✅ Finished all rounds. Final best FW={final_fw:.2f}. "
    #     f"Ranking written to {output_excel}"
    # )

    # Build set of remaining (unassessed) edge pairs
    remaining_pairs = set()
    if remaining_indices:
        for gi in remaining_indices:
            u, v, w = candidate_edges[gi]
            remaining_pairs.add((u, v))
    log(f"[MAIN] ℹ️ Unassessed edges remaining: {len(remaining_pairs)}")

    return best_scores, final_fw, remaining_pairs


# In[15]:


# ============================================
# Block 5: func2 – SCC block refinement (largest SCC intervals)
# ============================================

def _compute_block_fw(G, scores, block_nodes):
    subG = G.subgraph(block_nodes)
    fw = 0.0
    for u, v, data in subG.edges(data=True):
        w = data.get('weight', 1.0)
        if scores[u] < scores[v]:
            fw += w
    return fw


def _worker_refine_block(args):
    (
        worker_id,
        block_nodes,
        brute_force_min_size,
        brute_force_max_size,
        debug
    ) = args

    import os, time
    from datetime import datetime

    # Use global objects initialized via _init_refine_worker
    global G_GLOBAL, EDGES_GLOBAL, SCORES_BEFORE_GLOBAL
    G = G_GLOBAL
    edges = EDGES_GLOBAL
    scores_before = SCORES_BEFORE_GLOBAL

    pid = os.getpid()
    block_nodes = list(block_nodes)

    # --- Start timestamp ---
    t_start = time.time()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # if debug:
    #     print(f"[{worker_id} | PID {pid} | START {start_ts}] 🧵 starting on block with {len(block_nodes)} nodes")

    fw_block_before = _compute_block_fw(G, scores_before, block_nodes)
    # if debug:
    #     print(f"[{worker_id} | PID {pid}] block FW BEFORE = {fw_block_before:.6f}")

    # Call refine_block_scc_interval
    new_scores, _, _, _ = refine_block_scc_interval(
        G=G,
        edges=edges,
        node_order=scores_before,
        block_nodes=block_nodes,
        brute_force_min_size=brute_force_min_size,
        brute_force_max_size=brute_force_max_size,
        debug=False
    )

    fw_block_after = _compute_block_fw(G, new_scores, block_nodes)
    # if debug:
    #     print(f"[{worker_id} | PID {pid}] block FW AFTER  = {fw_block_after:.6f}")
    #
    # if fw_block_after + 1e-9 < fw_block_before:
    #     raise RuntimeError(
    #         f"[{worker_id} | PID {pid}] ❌ block FW decreased: "
    #         f"before={fw_block_before:.6f}, after={fw_block_after:.6f}"
    #     )

    block_ranks = {n: new_scores[n] for n in block_nodes}

    # --- End timestamp ---
    t_end = time.time()
    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = t_end - t_start

    # if debug:
    #     print(
    #         f"[{worker_id} | PID {pid} | END {end_ts}] "
    #         f"✅ finished, ΔFW_block = {fw_block_after - fw_block_before:.6f}, "
    #         f"elapsed = {elapsed:.2f} s"
    #     )
    #
    return {
        "worker_id": worker_id,
        "pid": pid,
        "block_nodes": block_nodes,
        "block_ranks": block_ranks,
        "fw_block_before": fw_block_before,
        "fw_block_after": fw_block_after,
        "elapsed_worker": elapsed,
    }


def refine_block_scc_interval(
        G,
        edges,
        node_order,
        block_nodes,
        brute_force_min_size=2,
        brute_force_max_size=7,
        debug=False,
):
    """
    Refine a given interval (block) of nodes using SCC decomposition + topo sort.

    Returns
    -------
    new_node_order : dict[node -> int]
        Updated global ranking after refining this block.
    existing_fw : float
        Total forward weight before refinement (global).
    new_fw : float
        Total forward weight after refinement (global).
    improved : bool
        True iff new_fw > existing_fw.
    """
    import networkx as nx
    from itertools import permutations
    from collections import Counter

    block_nodes = list(block_nodes)
    # if debug:
    #     print(f"\n=== refine_block_scc_interval on block of size {len(block_nodes)} ===")

    # Sort block nodes by current rank for stability
    block_nodes = sorted(block_nodes, key=lambda n: node_order[n])

    # Copy ranking
    prev_order = node_order.copy()
    new_node_order = node_order.copy()

    # Subgraph induced by the block
    subG = G.subgraph(block_nodes).copy()
    # if debug:
    #     print(f"   subgraph: |V|={subG.number_of_nodes()}, |E|={subG.number_of_edges()}")

    # SCC decomposition inside block
    sub_sccs = list(nx.strongly_connected_components(subG))
    # if debug:
    #     print(f"   found {len(sub_sccs)} SCCs in block")

    # Map node -> SCC id
    node_to_subscc = {}
    for idx, scc in enumerate(sub_sccs):
        for n in scc:
            node_to_subscc[n] = idx

    # Build SCC DAG
    scc_dag = nx.DiGraph()
    scc_dag.add_nodes_from(range(len(sub_sccs)))
    for u, v in subG.edges():
        su = node_to_subscc[u]
        sv = node_to_subscc[v]
        if su != sv:
            scc_dag.add_edge(su, sv)

    # Topological order of SCCs
    scc_order = list(nx.topological_sort(scc_dag))
    # if debug:
    #     print(f"   topo order of SCCs length = {len(scc_order)}")

    # Construct new order for the block
    new_block_order = []
    for scc_id in scc_order:
        scc_nodes = list(sub_sccs[scc_id])
        scc_size = len(scc_nodes)

        if scc_size == 1:
            new_block_order.extend(scc_nodes)

        elif brute_force_min_size <= scc_size <= brute_force_max_size:
            if debug:
                print(f"   SCC {scc_id}: size={scc_size}, brute-forcing permutations")
            best_perm = None
            max_weight = float("-inf")
            for perm in permutations(scc_nodes):
                w_sum = 0.0
                for i, u in enumerate(perm):
                    for j in range(i + 1, scc_size):
                        v = perm[j]
                        if subG.has_edge(u, v):
                            w_sum += subG[u][v]['weight']
                if w_sum > max_weight:
                    max_weight = w_sum
                    best_perm = perm
            if debug:
                print(f"      best internal FW = {max_weight:.2f}")
            new_block_order.extend(best_perm)

        else:
            if debug:
                print(f"   SCC {scc_id}: size={scc_size}, keeping original order")
            sorted_original = sorted(scc_nodes, key=lambda n: prev_order[n])
            new_block_order.extend(sorted_original)

    # Sanity: permutation
    if set(new_block_order) != set(block_nodes) or len(new_block_order) != len(block_nodes):
        if debug:
            missing = set(block_nodes) - set(new_block_order)
            extra = set(new_block_order) - set(block_nodes)
            print(f"❌ mismatch in new_block_order: missing={len(missing)}, extra={len(extra)}")
        raise ValueError("Block nodes mismatch in new_block_order!")

    # Reassign ranks ONLY within the block, reusing original ranks
    orig_ranks = sorted(prev_order[n] for n in block_nodes)
    for i, node in enumerate(new_block_order):
        new_node_order[node] = orig_ranks[i]

    # Global duplicate-rank sanity check
    all_ranks = list(new_node_order.values())
    if len(all_ranks) != len(set(all_ranks)):
        if debug:
            duplicates = [r for r, cnt in Counter(all_ranks).items() if cnt > 1]
            print(f"❌ Duplicate ranks detected — {len(duplicates)} ranks are non-unique")
        raise ValueError("Duplicate ranks detected in new_node_order!")

    # Compute global FW before/after
    existing_fw = compute_forward_weight(edges, prev_order)
    new_fw = compute_forward_weight(edges, new_node_order)
    improved = new_fw > existing_fw

    # if debug:
    #     print(f"   FW before = {existing_fw:.2f}, FW after = {new_fw:.2f}, Δ={new_fw - existing_fw:.2f}")
    #     if improved:
    #         print("   ✅ block refinement improved global FW")
    #     else:
    #         print("   ⏭️  no global improvement from this block")

    return new_node_order, existing_fw, new_fw, improved


def parallel_refine_largest_scc_intervals(
        block_size=530,  # kept for API compatibility, NOT used for size range
        brute_force_min_size=2,
        brute_force_max_size=7,
        max_backward_flips=100,
        verify_every=5,
        csv_path="/content/drive/MyDrive/connectome_graph.csv",
        initial_ranking_path="/content/drive/MyDrive/bader.csv",
        output_path=None,
        save_every=5,  # save ranking to output_path every 'save_every' batches
        debug=False,
        max_no_improvement_batches=10,  # ⭐ NEW: patience on non-improving batches
):
    import time
    import random
    import os
    import math
    from datetime import datetime
    from collections import deque

    # --- hyper-params for learning-based interval selection ---
    P_EXPLOIT = 0.4  # prob. to exploit around previous good intervals
    P_UCB = 0.4  # prob. to use bucket UCB
    P_RANDOM = 0.2  # prob. to sample purely at random

    GOOD_INTERVAL_MAX = 50  # how many good intervals to remember
    COOL_DOWN_BATCHES = 10  # how long to avoid cold intervals
    COOL_MIN_DELTA_FW = -1e-9  # ⭐ NEW: only ΔFW < 0 -> mark cold

    # --- 0) detect number of processors ---
    try:
        num_procs = len(os.sched_getaffinity(0))
    except Exception:
        num_procs = mp.cpu_count()

    # if debug:
    #     print("🚀 Starting parallel refinement on largest SCC (rank-interval blocks)")
    #     print(f"   🧵 Detected {num_procs} processors")
    #     print(f"   brute_force SCC sizes = [{brute_force_min_size}, {brute_force_max_size}]")
    #     print(f"   max_backward_flips = {max_backward_flips}")
    #     print(f"   Graph CSV: {csv_path}")
    #     print(f"   Initial ranking CSV: {initial_ranking_path}")

    # --- 1) Read graph and initial scores ---
    edges_indexed, node_to_index, index_to_node = read_graph(csv_path)
    scores = load_initial_scores(initial_ranking_path, node_to_index)

    # if debug:
    #     print(f"   Loaded {len(edges_indexed)} edges and {len(scores)} nodes")
    #
    # if not all_scores_unique(scores):
    #     raise RuntimeError("Initial scores are not unique")

    # Build DiGraph
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges_indexed)

    # --- 1.5) Determine output_path & its components ---
    if output_path is None:
        dir_path, base_name = os.path.split(initial_ranking_path)
        base_root, ext = os.path.splitext(base_name)
        out_name = f"{base_root}_largest_scc_parallel_blocks{ext}"
        output_path = os.path.join(dir_path, out_name)
    else:
        dir_path, base_name = os.path.split(output_path)
        base_root, ext = os.path.splitext(base_name)
        if dir_path == "":
            parent_dir, _ = os.path.split(initial_ranking_path)
            output_path = os.path.join(parent_dir, output_path)
            dir_path = parent_dir

    if debug:
        print(f"   Output ranking will be written to: {output_path}")

    # --- 2) Global SCC-based topo reorder (so SCCs are contiguous in rank) ---
    scores = global_scc_topo_reorder_scores(G, edges_indexed, scores, debug=debug)

    # --- 3) Largest SCC ---
    # if debug:
    #     print("🔍 Finding largest strongly connected component...")
    sccs = list(nx.strongly_connected_components(G))
    if not sccs:
        raise RuntimeError("No SCCs found in graph (graph empty?).")

    largest_scc = max(sccs, key=len)
    largest_scc_set = set(largest_scc)

    if debug:
        print(f"   Largest SCC size = {len(largest_scc)} nodes")

    largest_scc_nodes_sorted = sorted(list(largest_scc), key=lambda n: scores[n])
    n_scc = len(largest_scc_nodes_sorted)

    # if debug:
    #     print("   Sample of largest SCC nodes with ranks (first 10):")
    #     for n in largest_scc_nodes_sorted[:10]:
    #         print(f"      node {n} -> rank {scores[n]}")

    # --- 4) Random block size range in terms of SCC nodes per interval ---
    min_scc_block_size = 2000
    max_scc_block_size = 13000

    if n_scc < min_scc_block_size:
        min_scc_block_size = max(1, n_scc // 2)
        max_scc_block_size = n_scc

    # if debug:
    #     print(
    #         f"   SCC-interval sizes in [min_scc_block_size={min_scc_block_size}, "
    #         f"max_scc_block_size={max_scc_block_size}]"
    #     )

    # Helper: is there any backward edge inside rank interval [r_low, r_high]?
    def _rank_interval_has_backward_edge(edges, scores_loc, r_low, r_high):
        for u, v, w in edges:
            ru = scores_loc[u]
            rv = scores_loc[v]
            in_u = (r_low <= ru <= r_high)
            in_v = (r_low <= rv <= r_high)
            if in_u and in_v and ru > rv:
                return True
        return False

    # --- 4.5) Bucketization over rank space for learned interval selection ---
    scc_ranks = [scores[n] for n in largest_scc_nodes_sorted]
    min_rank_scc = min(scc_ranks)
    max_rank_scc = max(scc_ranks)

    target_buckets = max(num_procs * 4, 8)
    total_rank_span = max_rank_scc - min_rank_scc + 1
    bucket_size = max(1, total_rank_span // target_buckets)

    num_buckets = (total_rank_span + bucket_size - 1) // bucket_size
    bucket_fw_gain = [0.0] * num_buckets
    bucket_counts = [0] * num_buckets

    bucket_to_indices = [[] for _ in range(num_buckets)]
    for idx, node in enumerate(largest_scc_nodes_sorted):
        r = scores[node]
        b = (r - min_rank_scc) // bucket_size
        if 0 <= b < num_buckets:
            bucket_to_indices[b].append(idx)

    # if debug:
    #     print(f"   Bucketization: bucket_size={bucket_size}, num_buckets={num_buckets}")
    #     non_empty = sum(1 for lst in bucket_to_indices if lst)
    #     print(f"   Buckets with at least one SCC node: {non_empty}")

    # Counters and memory for learning
    total_blocks_sampled = 0
    good_intervals = deque(maxlen=GOOD_INTERVAL_MAX)  # (r_low, r_high, delta_fw, batch_idx)
    cool_intervals = []  # list of (r_low, r_high, batch_idx)
    no_improve_batches = 0  # ⭐ NEW: consecutive non-improving batches

    def _pick_bucket_index():
        """UCB-style selection of a bucket to center a new interval in."""
        nonlocal total_blocks_sampled

        candidate_buckets = [b for b in range(num_buckets) if bucket_to_indices[b]]
        if not candidate_buckets:
            return None

        total_blocks_sampled += 1
        t = total_blocks_sampled

        if all(bucket_counts[b] == 0 for b in candidate_buckets):
            return random.choice(candidate_buckets)

        c = 1.0
        best_b = None
        best_score = float("-inf")

        for b in candidate_buckets:
            n_b = bucket_counts[b]
            if n_b == 0:
                bonus = math.sqrt(2.0 * math.log(t + 1.0))
                mean = 0.0
            else:
                mean = bucket_fw_gain[b] / n_b
                bonus = c * math.sqrt(math.log(t + 1.0) / n_b)
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_b = b

        return best_b

    def _interval_buckets(r_low, r_high):
        """Return list of bucket indices intersecting [r_low, r_high]."""
        if r_high < min_rank_scc or r_low > max_rank_scc:
            return []
        b_start = max(0, (max(r_low, min_rank_scc) - min_rank_scc) // bucket_size)
        b_end = min(
            num_buckets - 1,
            (min(r_high, max_rank_scc) - min_rank_scc) // bucket_size,
        )
        return list(range(b_start, b_end + 1))

    def _interval_overlaps(a_low, a_high, b_low, b_high):
        return not (a_high < b_low or b_high < a_low)

    def _in_cooldown(r_low, r_high, current_batch_idx):
        """Check if [r_low, r_high] overlaps a 'cold' interval still under cooldown."""
        for (cl, ch, b_idx) in cool_intervals:
            if current_batch_idx - b_idx <= COOL_DOWN_BATCHES:
                if _interval_overlaps(r_low, r_high, cl, ch):
                    return True
        return False

    current_scores = scores.copy()
    fw_current = compute_forward_weight(edges_indexed, current_scores)
    fw_history = [fw_current]

    if debug:
        print(f"⚖️ Initial total FW = {fw_current:.6f}")

    f2b_edges_global = set()
    batch_idx = 0
    t0 = time.time()

    # ================= MAIN BATCH LOOP =================
    while True:
        # stopping condition 1: F→B budget exhausted
        if len(f2b_edges_global) >= max_backward_flips:
            if debug:
                print(f"⛔ Stopping: reached max_backward_flips={max_backward_flips}")
            break

        # ⭐ NEW: stopping condition 2: too many non-improving batches in a row
        if (
                max_no_improvement_batches is not None
                and no_improve_batches >= max_no_improvement_batches
        ):
            if debug:
                print(
                    f"⛔ Stopping: no FW improvement in the last "
                    f"{no_improve_batches} batches (threshold={max_no_improvement_batches})"
                )
            break

        batch_blocks = []  # list of (r_low, r_high, block_nodes)
        used_rank_ranges = []  # list of (r_low, r_high)
        max_attempts = num_procs * 20
        attempts = 0

        # ---------- build one batch of up to num_procs blocks ----------
        while len(batch_blocks) < num_procs and attempts < max_attempts:
            attempts += 1

            # Decide strategy: exploit / UCB / random
            r_strategy = random.random()
            candidate_interval = None  # (r_low, r_high)

            # 1) Exploit around successful intervals
            if good_intervals and r_strategy < P_EXPLOIT:
                base_r_low, base_r_high, _, _ = random.choice(good_intervals)
                width = max(1, base_r_high - base_r_low)

                expand_frac = random.uniform(-0.3, 0.5)
                shift_frac = random.uniform(-0.3, 0.3)

                new_width = int(max(1, width * (1.0 + expand_frac)))
                r_center = (base_r_low + base_r_high) / 2.0 + shift_frac * width

                r_low = int(round(r_center - new_width / 2.0))
                r_high = int(round(r_center + new_width / 2.0))

                if r_low < min_rank_scc:
                    r_low = min_rank_scc
                if r_high > max_rank_scc:
                    r_high = max_rank_scc
                if r_high > r_low:
                    candidate_interval = (r_low, r_high)

            # 2) UCB-based bucket selection
            elif r_strategy < P_EXPLOIT + P_UCB:
                chosen_bucket = _pick_bucket_index()
                if chosen_bucket is None:
                    candidate_interval = None
                else:
                    indices_in_bucket = bucket_to_indices[chosen_bucket]
                    if not indices_in_bucket:
                        candidate_interval = None
                    else:
                        center_idx = random.choice(indices_in_bucket)
                        size_scc = random.randint(min_scc_block_size, max_scc_block_size)
                        if size_scc > n_scc:
                            size_scc = n_scc
                        if size_scc <= 0:
                            candidate_interval = None
                        else:
                            start_idx_scc = center_idx - size_scc // 2
                            if start_idx_scc < 0:
                                start_idx_scc = 0
                            end_idx_scc = start_idx_scc + size_scc
                            if end_idx_scc > n_scc:
                                end_idx_scc = n_scc
                                start_idx_scc = max(0, end_idx_scc - size_scc)

                            if end_idx_scc > start_idx_scc:
                                first_node = largest_scc_nodes_sorted[start_idx_scc]
                                last_node = largest_scc_nodes_sorted[end_idx_scc - 1]
                                r_low = scores[first_node]
                                r_high = scores[last_node]
                                if r_high < r_low:
                                    r_low, r_high = r_high, r_low
                                candidate_interval = (r_low, r_high)

            # 3) Pure random interval over largest SCC ranks
            else:
                size_scc = random.randint(min_scc_block_size, max_scc_block_size)
                if size_scc > n_scc:
                    size_scc = n_scc
                if size_scc <= 0:
                    candidate_interval = None
                else:
                    start_idx_scc = random.randint(0, n_scc - size_scc)
                    end_idx_scc = start_idx_scc + size_scc
                    first_node = largest_scc_nodes_sorted[start_idx_scc]
                    last_node = largest_scc_nodes_sorted[end_idx_scc - 1]
                    r_low = scores[first_node]
                    r_high = scores[last_node]
                    if r_high < r_low:
                        r_low, r_high = r_high, r_low
                    candidate_interval = (r_low, r_high)

            if candidate_interval is None:
                continue

            r_low, r_high = candidate_interval

            # Skip intervals under cooldown (recently "cold")
            if _in_cooldown(r_low, r_high, batch_idx + 1):
                continue

            # Ensure no overlap with intervals already in this batch
            overlap = False
            for (rl, rh) in used_rank_ranges:
                if _interval_overlaps(r_low, r_high, rl, rh):
                    overlap = True
                    break
            if overlap:
                continue

            # Determine block nodes based on *current* scores
            block_nodes = [n for n in current_scores if r_low <= current_scores[n] <= r_high]
            if not block_nodes:
                continue

            # Skip blocks with no backward edges inside the rank interval
            if not _rank_interval_has_backward_edge(edges_indexed, current_scores, r_low, r_high):
                continue

            batch_blocks.append((r_low, r_high, block_nodes))
            used_rank_ranges.append((r_low, r_high))

        # --- end building batch ---

        if not batch_blocks:
            if debug:
                print("🛑 No more rank-intervals with backward edges found. Stopping.")
            break

        # Sanity: intervals should be strictly non-overlapping
        used_rank_ranges_sorted = sorted(used_rank_ranges)
        for i in range(1, len(used_rank_ranges_sorted)):
            prev_l, prev_h = used_rank_ranges_sorted[i - 1]
            cur_l, cur_h = used_rank_ranges_sorted[i]
            if not (cur_l > prev_h):
                raise RuntimeError(
                    f"Rank-interval overlap detected between "
                    f"{used_rank_ranges_sorted[i - 1]} and {used_rank_ranges_sorted[i]}"
                )

        batch_idx += 1
        if debug:
            print(f"\n========== BATCH {batch_idx} ==========")
            for b_id, (r_low, r_high, block_nodes) in enumerate(batch_blocks):
                print(
                    f"  - Block {b_id}: rank-range=[{r_low}:{r_high}] "
                    f"size={len(block_nodes)}"
                )

        scores_before_batch = current_scores.copy()

        worker_args = []
        for b_id, (r_low, r_high, block_nodes) in enumerate(batch_blocks):
            worker_id = f"batch{batch_idx}_blk{b_id}"
            worker_args.append(
                (
                    worker_id,
                    block_nodes,
                    brute_force_min_size,
                    brute_force_max_size,
                    debug,
                )
            )

        if debug:
            print(f"🔥 Spawning {len(batch_blocks)} block tasks with a pool of {num_procs} workers")

        batch_start = time.time()
        with mp.Pool(
                processes=num_procs,
                initializer=_init_refine_worker,
                initargs=(G, edges_indexed, scores_before_batch)
        ) as pool:
            results = pool.map(_worker_refine_block, worker_args)
        batch_end = time.time()
        batch_elapsed = batch_end - batch_start

        # if debug:
        #     print("🧬 All workers in this batch finished. Merging results...")
        #     worker_times = [res["elapsed_worker"] for res in results]
        #     max_worker = max(worker_times) if worker_times else 0.0
        #     sum_worker = sum(worker_times)
        #     print(
        #         f"⏱️ Batch {batch_idx} timing: "
        #         f"wall = {batch_elapsed:.2f} s, "
        #         f"max worker = {max_worker:.2f} s, "
        #         f"sum worker = {sum_worker:.2f} s"
        #     )
        #     for res in results:
        #         print(
        #             f"   - {res['worker_id']} (PID {res['pid']}) "
        #             f"elapsed = {res['elapsed_worker']:.2f} s"
        #         )

        # Merge block ranks into global scores
        for res in results:
            worker_id = res["worker_id"]
            block_nodes = res["block_nodes"]
            block_ranks = res["block_ranks"]
            if debug:
                print(f"[Merge {worker_id}] updating {len(block_nodes)} nodes in global scores")
            for n, r in block_ranks.items():
                current_scores[n] = r

        if not all_scores_unique(current_scores):
            raise RuntimeError(
                f"❌ Duplicate scores detected after batch {batch_idx}"
            )

        sbef = scores_before_batch
        saft = current_scores
        (
            sum_sbef_only,
            sum_saft_only,
            sum_both_forward,
            sum_both_backward,
            only_before_global
        ) = compare_forward_weights(edges_indexed, sbef, saft)

        fw_old = fw_current
        fw_new = fw_old - sum_sbef_only + sum_saft_only
        delta_fw = fw_new - fw_old

        # ⭐ NEW: update non-improvement streak
        if delta_fw > 1e-9:
            no_improve_batches = 0
        else:
            no_improve_batches += 1

        # Cross-boundary invariant: edges crossing union of all block nodes
        B = set()
        for _, _, block_nodes in batch_blocks:
            B.update(block_nodes)
        bad_cross = 0
        for (u, v, w) in edges_indexed:
            in_u = (u in B)
            in_v = (v in B)
            if in_u ^ in_v:
                was_fwd = sbef[u] < sbef[v]
                now_fwd = saft[u] < saft[v]
                if was_fwd != now_fwd:
                    bad_cross += 1
                    if bad_cross <= 5 and debug:
                        print("❌ Cross-boundary edge changed direction:", (u, v, w), was_fwd, "→", now_fwd)
        # if bad_cross > 0:
        #     raise RuntimeError(
        #         f"❌ {bad_cross} cross-boundary edges changed direction. "
        #         "Block rank-interval invariant broken."
        #     )

        fw_current = fw_new
        fw_history.append(fw_current)

        for e in only_before_global:
            f2b_edges_global.add(e)

        # if debug:
        #     print(
        #         f"📈 Batch {batch_idx}: FW old={fw_old:.6f}, new={fw_new:.6f}, "
        #         f"Δ={delta_fw:.6f}"
        #     )
        #     print(
        #         f"   F→B weight this batch = {sum_sbef_only:.6f}, "
        #         f"B→F weight this batch = {sum_saft_only:.6f}"
        #     )
        #     print(
        #         f"   New F→B edges this batch = {len(only_before_global)}, "
        #         f"Total F→B edges so far = {len(f2b_edges_global)}"
        #     )

        # --- Update bucket statistics with this batch's FW change ---
        affected_buckets = set()
        for (r_low, r_high, _) in batch_blocks:
            for b in _interval_buckets(r_low, r_high):
                affected_buckets.add(b)

        if affected_buckets:
            reward_per_bucket = delta_fw / len(affected_buckets)
            for b in affected_buckets:
                bucket_fw_gain[b] += reward_per_bucket
                bucket_counts[b] += 1

        # --- Update good / cool interval memories ---
        if delta_fw > 1e-9:
            for (r_low, r_high, _) in batch_blocks:
                good_intervals.append((r_low, r_high, delta_fw, batch_idx))
        elif delta_fw < COOL_MIN_DELTA_FW:  # ⭐ NEW: only strictly negative batches are cold
            for (r_low, r_high, _) in batch_blocks:
                cool_intervals.append((r_low, r_high, batch_idx))

        if save_every is not None and save_every > 0 and batch_idx % save_every == 0:
            rows_temp = []
            for idx, order_val in current_scores.items():
                node_id_str = index_to_node[idx]
                rows_temp.append((node_id_str, order_val))
            rows_temp_sorted = sorted(rows_temp, key=lambda x: x[1])
            pd.DataFrame(rows_temp_sorted, columns=["Node ID", "Order"]).to_csv(output_path, index=False)
            if debug:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"💾 [{ts}] Saved intermediate ranking at batch {batch_idx} → {output_path}")

        if verify_every is not None and verify_every > 0 and batch_idx % verify_every == 0:
            fw_manual = compute_forward_weight(edges_indexed, current_scores)
            if abs(fw_manual - fw_current) > 1e-6:
                raise RuntimeError(
                    f"❌ FW mismatch at verification batch {batch_idx}: "
                    f"tracked={fw_current:.6f}, manual={fw_manual:.6f}"
                )
            if debug:
                print(
                    f"✅ Verification at batch {batch_idx}: "
                    f"FW tracked={fw_current:.6f}, manual={fw_manual:.6f}"
                )

    elapsed = time.time() - t0
    # if debug:
    #     print("\n======== PARALLEL REFINEMENT DONE ========")
    #     print(f"⏱️ Total time = {elapsed:.2f} seconds")
    #     print(f"📊 Final FW = {fw_current:.6f}")
    #     print(f"🔁 Total batches run = {batch_idx}")
    #     print(
    #         f"↘️ Total unique edges that went forward→backward = {len(f2b_edges_global)}"
    #     )

    rows = []
    for idx, order_val in current_scores.items():
        node_id_str = index_to_node[idx]
        rows.append((node_id_str, order_val))
    rows_sorted = sorted(rows, key=lambda x: x[1])

    df_out = pd.DataFrame(rows_sorted, columns=["Node ID", "Order"])
    df_out.to_csv(output_path, index=False)

    if debug:
        ts_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"💾 [{ts_end}] Final ranking written to: {output_path}")

    return current_scores, fw_history, f2b_edges_global, output_path


# In[16]:


def compute_direction_change_stats(
        edges,
        before_scores,
        after_scores,
        restrict_to_edges=None,
):
    """
    Compute how many edges changed direction between two rankings and
    the total weight of those changes.

    Parameters
    ----------
    edges : iterable of (u, v, w)
        All directed edges with weights.
    before_scores : dict
        node -> rank BEFORE the phase.
    after_scores : dict
        node -> rank AFTER the phase.
    restrict_to_edges : iterable of (u, v) or None
        If not None, we only consider edges whose (u, v) is in this set.

    Returns
    -------
    num_B2F : int
        Number of edges that went from backward to forward.
    wt_B2F : float
        Total weight of edges that went from backward to forward.
    num_F2B : int
        Number of edges that went from forward to backward.
    wt_F2B : float
        Total weight of edges that went from forward to backward.
    """
    if restrict_to_edges is not None:
        restrict_set = set(restrict_to_edges)
    else:
        restrict_set = None

    num_B2F = 0
    wt_B2F = 0.0
    num_F2B = 0
    wt_F2B = 0.0

    for (u, v, w) in edges:
        if restrict_set is not None and (u, v) not in restrict_set:
            continue

        bu = before_scores[u]
        bv = before_scores[v]
        au = after_scores[u]
        av = after_scores[v]

        # Orientation before
        before_forward = bu < bv
        before_backward = bu > bv

        # Orientation after
        after_forward = au < av
        after_backward = au > av

        # backward -> forward
        if before_backward and after_forward:
            num_B2F += 1
            wt_B2F += w

        # forward -> backward
        elif before_forward and after_backward:
            num_F2B += 1
            wt_F2B += w

        # If it was neither strictly forward nor backward (tie),
        # or stayed in the same class, we ignore it.

    return num_B2F, wt_B2F, num_F2B, wt_F2B


# In[17]:


def hybrid_refine_func1_func2(
        csv_path,
        initial_ranking_path,
        output_ranking_path,  # 👈 single CSV that is always overwritten
        log_path,  # 👈 text log file (explicit)
        h_hours: float = 4.0,
        func2_f2b_limit: int = 200,
        func1_full_every: int = 50,
        func1_log_every_round: int = 1,
        func1_fw_check_every_round: int = 5,
        func2_verify_every: int = 5,
        c: float = 1.0,  # 👈 kept for compatibility, but IGNORED now
):
    """
    Hybrid process with a SINGLE ranking CSV:

      - All rankings are always written to `output_ranking_path`.
      - Each phase overwrites that file with the new ranking.

    Behavior:

      1) Initial func1 phase (parallel dynamic refinement):

         - Uses **all backward edges**, i.e. `edge_subset=None`.
         - Runs until internal stopping:
             * time limit (remaining_h), OR
             * 10 minutes of no improvement (inside func1), OR
             * no more useful edges.
         - Returns a set of edges that were **not assessed** by func1
           (unassessed_edges_initial).

      2) Main loop (repeat until total time h_hours exhausted):

           (a) Run func2 (parallel_refine_largest_scc_intervals) on the current
               ranking; it returns the set of edges that changed direction
               from forward->backward: f2b_edges_global.

           (b) Run func1 again with:

               - If there are still unassessed edges from previous func1 runs:
                     S = (unassessed_edges_global ∪ f2b_edges_global)
                 (targeted subset)

               - If unassessed_edges_global is EMPTY:
                     run func1 **globally** again (edge_subset=None),
                     i.e., consider all backward edges under current ranking.

               func1 returns:
                 - updated scores / FW internally (and writes ranking CSV),
                 - a new set of edges that it still did not assess in this phase;
                   we update unassessed_edges_global to that.

           (c) Whenever func1_runs % func1_full_every == 0, run an extra
               **global func1** with edge_subset=None. After that run, the
               unassessed_edges_global becomes whatever edges that global
               run left unassessed.

      - `c` is now ignored; it is kept only for backward compatibility
        in the signature.
    """

    import os
    import time
    import datetime

    # ------------------------------
    # Load graph once for statistics
    # ------------------------------
    edges, node_to_index, index_to_node = read_graph(csv_path)

    # ------------------------------
    # Ensure output directory exists
    # ------------------------------
    out_dir = os.path.dirname(output_ranking_path)
    if out_dir == "":
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------
    # Processor counts (for log)
    # ------------------------------
    try:
        func2_procs_aff = len(os.sched_getaffinity(0))
    except Exception:
        func2_procs_aff = None

    func1_procs = os.cpu_count() or 1
    func2_procs = func2_procs_aff if func2_procs_aff is not None else func1_procs

    # ------------------------------
    # Helpers to write log
    # ------------------------------
    def log_line(text: str):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {text}\n")

    # ------------------------------
    # Write header in log file
    # ------------------------------
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# HYBRID func1/func2 LOG (all rankings in ONE CSV)\n")
        f.write(f"# ranking_csv = {output_ranking_path}\n")
        f.write("# This file contains per-PHASE statistics:\n")
        f.write("#   - FW before/after each phase\n")
        f.write("#   - Counts & total weight of edges that changed direction\n")
        f.write("#   - Extra info (subset sizes, limits, etc.)\n")
        f.write(f"# func1_processors (dynamic round) = {func1_procs}\n")
        f.write(f"# func2_processors (largest SCC intervals) = {func2_procs}\n")
        f.write(
            "# COLUMNS:\n"
            "# phase_idx, phase_type, duration_sec, "
            "fw_before, fw_after, "
            "num_B2F, wt_B2F, num_F2B, wt_F2B, extra\n"
        )
        f.write("# NOTE: parameter c is ignored; initial func1 uses ALL backward edges.\n")

    # ------------------------------
    # Time limit
    # ------------------------------
    start_ts = time.time()
    deadline = start_ts + h_hours * 3600.0

    phase_idx = 0
    func1_runs = 0
    func2_runs = 0

    # global pool of unassessed edges from func1 phases
    # represented as (u,v) pairs
    unassessed_edges_global = set()

    # Helper: load scores + FW from a ranking CSV
    def fw_from_ranking_path(ranking_path: str):
        scores = load_initial_scores(ranking_path, node_to_index)
        fw = compute_forward_weight(edges, scores)
        return scores, fw

    current_input_path = initial_ranking_path

    # =====================================================
    # PHASE 1: func1 on ALL backward edges (edge_subset=None)
    # =====================================================
    if time.time() >= deadline:
        log_line("No time left to start initial func1; exiting.")
        return output_ranking_path

    phase_idx += 1
    func1_runs += 1
    phase_type = "func1_initial_global"

    before_scores, fw_before = fw_from_ranking_path(current_input_path)

    t0 = time.time()
    remaining_h = max(0.0, (deadline - t0) / 3600.0)

    # refine_ranking_parallel_dynamic is expected to return:
    #   best_scores, final_fw, remaining_pairs_unassessed
    _, _, remaining_pairs_unassessed = refine_ranking_parallel_dynamic(
        csv_path=csv_path,
        initial_ranking_path=current_input_path,
        output_excel=output_ranking_path,
        MAX_HOURS=remaining_h,
        LOG_EVERY_ROUND=func1_log_every_round,
        FW_CHECK_EVERY_ROUND=func1_fw_check_every_round,
        edge_subset=None,  # 👈 ALL backward edges
    )
    t1 = time.time()

    after_scores, fw_after = fw_from_ranking_path(output_ranking_path)

    num_B2F, wt_B2F, num_F2B, wt_F2B = compute_direction_change_stats(
        edges, before_scores, after_scores
    )

    unassessed_edges_global = set(remaining_pairs_unassessed or [])
    extra_subset_info = (
        f"initial_global_func1, unassessed_edges={len(unassessed_edges_global)}"
    )

    log_line(
        f"{phase_idx}, {phase_type}, {t1 - t0:.2f}, "
        f"{fw_before:.2f}, {fw_after:.2f}, "
        f"{num_B2F}, {wt_B2F:.2f}, {num_F2B}, {wt_F2B:.2f}, "
        f"{extra_subset_info}"
    )

    current_input_path = output_ranking_path

    # =====================================================
    # MAIN LOOP: func2 ↔ func1 until time runs out
    # =====================================================
    last_f2b_edges = None

    while True:
        # ------------------------------------
        # (A) func2: run until func2_f2b_limit F→B edges
        # ------------------------------------
        if time.time() >= deadline:
            log_line("Time limit reached before starting next func2; stopping.")
            break

        phase_idx += 1
        func2_runs += 1
        phase_type = f"func2_run{func2_runs}"

        before_scores, fw_before = fw_from_ranking_path(current_input_path)

        t0 = time.time()
        scores_after_func2, fw_history, f2b_edges_global, out_path_used = (
            parallel_refine_largest_scc_intervals(
                max_backward_flips=func2_f2b_limit,
                verify_every=func2_verify_every,
                csv_path=csv_path,
                initial_ranking_path=current_input_path,
                output_path=output_ranking_path,  # 👈 always overwrite the same file
                debug=False,
            )
        )
        t1 = time.time()

        after_scores, fw_after = fw_from_ranking_path(output_ranking_path)

        num_B2F, wt_B2F, num_F2B, wt_F2B = compute_direction_change_stats(
            edges, before_scores, after_scores
        )

        last_f2b_edges = f2b_edges_global or []

        extra = (
            f"func2_max_F2B={func2_f2b_limit}, "
            f"func2_actual_F2B={len(last_f2b_edges)}"
        )
        log_line(
            f"{phase_idx}, {phase_type}, {t1 - t0:.2f}, "
            f"{fw_before:.2f}, {fw_after:.2f}, "
            f"{num_B2F}, {wt_B2F:.2f}, {num_F2B}, {wt_F2B:.2f}, "
            f"{extra}"
        )

        current_input_path = output_ranking_path

        # If func2 did not create any forward->backward edges AND
        # we have no leftover unassessed edges, nothing to fix
        if not last_f2b_edges and not unassessed_edges_global:
            log_line(
                "func2 produced 0 forward->backward edges and no unassessed edges remain; stopping."
            )
            break

        # ------------------------------------
        # (B) func1: on (F2B from func2) ∪ (unassessed edges so far),
        #     BUT if unassessed_edges_global is EMPTY, run GLOBAL func1.
        # ------------------------------------
        if time.time() >= deadline:
            log_line("Time limit reached before starting func1-after-func2; stopping.")
            break

        # Normalize F2B edges to (u,v) pairs
        f2b_pairs = set()
        for e in last_f2b_edges:
            if isinstance(e, tuple) and len(e) >= 2:
                f2b_pairs.add((e[0], e[1]))

        unassessed_before = len(unassessed_edges_global)

        # Decide mode: targeted vs global
        if unassessed_before == 0:
            # 🔁 Backlog is empty → consider ALL backward edges again
            use_global_for_this_func1 = True
            subset_for_func1 = None
        else:
            use_global_for_this_func1 = False
            subset_for_func1 = set(unassessed_edges_global)
            subset_for_func1.update(f2b_pairs)

        if not use_global_for_this_func1 and not subset_for_func1:
            log_line(
                "No edges to refine in func1-after-func2 (subset empty). Skipping func1."
            )
            # still allow another func2 in next loop
            continue

        phase_idx += 1
        func1_runs += 1
        phase_type = f"func1_after_func2_run{func1_runs}"

        before_scores, fw_before = fw_from_ranking_path(current_input_path)

        t0 = time.time()
        remaining_h = max(0.0, (deadline - t0) / 3600.0)

        if use_global_for_this_func1:
            # GLOBAL func1
            _, _, remaining_pairs_unassessed = refine_ranking_parallel_dynamic(
                csv_path=csv_path,
                initial_ranking_path=current_input_path,
                output_excel=output_ranking_path,  # overwrite same file
                MAX_HOURS=remaining_h,
                LOG_EVERY_ROUND=func1_log_every_round,
                FW_CHECK_EVERY_ROUND=func1_fw_check_every_round,
                edge_subset=None,  # 👈 global
            )
        else:
            # TARGETED func1 on unassessed ∪ F2B
            _, _, remaining_pairs_unassessed = refine_ranking_parallel_dynamic(
                csv_path=csv_path,
                initial_ranking_path=current_input_path,
                output_excel=output_ranking_path,  # overwrite same file
                MAX_HOURS=remaining_h,
                LOG_EVERY_ROUND=func1_log_every_round,
                FW_CHECK_EVERY_ROUND=func1_fw_check_every_round,
                edge_subset=list(subset_for_func1),  # targeted
            )

        t1 = time.time()

        after_scores, fw_after = fw_from_ranking_path(output_ranking_path)

        num_B2F, wt_B2F, num_F2B, wt_F2B = compute_direction_change_stats(
            edges, before_scores, after_scores
        )

        # Stats restricted to those F→B edges from func2
        num_B2F_sub, wt_B2F_sub, num_F2B_sub, wt_F2B_sub = compute_direction_change_stats(
            edges, before_scores, after_scores, restrict_to_edges=list(f2b_pairs)
        )

        # Update global unassessed pool from this func1 run
        unassessed_edges_global = set(remaining_pairs_unassessed or [])
        unassessed_after = len(unassessed_edges_global)

        if use_global_for_this_func1:
            subset_size_str = "ALL_BACKWARD"
        else:
            subset_size_str = len(subset_for_func1)

        extra = (
            f"mode={'global' if use_global_for_this_func1 else 'targeted'}, "
            f"subset_total={subset_size_str}, "
            f"subset_F2B_size={len(f2b_pairs)}, "
            f"subset_B2F={num_B2F_sub}, subset_wt_B2F={wt_B2F_sub:.2f}, "
            f"subset_F2B={num_F2B_sub}, subset_wt_F2B={wt_F2B_sub:.2f}, "
            f"unassessed_before={unassessed_before}, "
            f"unassessed_after_func1={unassessed_after}"
        )

        log_line(
            f"{phase_idx}, {phase_type}, {t1 - t0:.2f}, "
            f"{fw_before:.2f}, {fw_after:.2f}, "
            f"{num_B2F}, {wt_B2F:.2f}, {num_F2B}, {wt_F2B:.2f}, "
            f"{extra}"
        )

        current_input_path = output_ranking_path

        # ------------------------------------
        # (C) every func1_full_every: extra global func1
        # ------------------------------------
        if func1_runs % func1_full_every == 0:
            if time.time() >= deadline:
                log_line("Time limit reached before extra-every-N-func1; stopping.")
                break

            phase_idx += 1
            phase_type = f"func1_every{func1_full_every}_global"
            before_scores, fw_before = fw_from_ranking_path(current_input_path)

            t0 = time.time()
            remaining_h = max(0.0, (deadline - t0) / 3600.0)

            _, _, remaining_pairs_unassessed = refine_ranking_parallel_dynamic(
                csv_path=csv_path,
                initial_ranking_path=current_input_path,
                output_excel=output_ranking_path,  # global overwrite
                MAX_HOURS=remaining_h,
                LOG_EVERY_ROUND=func1_log_every_round,
                FW_CHECK_EVERY_ROUND=func1_fw_check_every_round,
                edge_subset=None,  # global
            )
            t1 = time.time()

            after_scores, fw_after = fw_from_ranking_path(output_ranking_path)
            num_B2F, wt_B2F, num_F2B, wt_F2B = compute_direction_change_stats(
                edges, before_scores, after_scores
            )

            # after a global func1, reset unassessed pool to what that run left
            unassessed_edges_global = set(remaining_pairs_unassessed or [])

            fw_delta = fw_after - fw_before

            log_line(
                f"{phase_idx:3d}  |  {phase_type:28s}  |  "
                f"dur={t1 - t0:7.2f}s  |  "
                f"FW: {fw_before:10.2f} -> {fw_after:10.2f}  (Δ={fw_delta:+8.2f})  |  "
                f"B2F: {num_B2F:5d} (wt={wt_B2F:8.2f})  |  "
                f"F2B: {num_F2B:5d} (wt={wt_F2B:8.2f})  |  "
                f"note=every_{func1_full_every}  |  "
                f"unassessed={len(unassessed_edges_global)}"
            )

            # extra blank line between rows for readability
            log_line("")

            current_input_path = output_ranking_path

    log_line(
        f"Hybrid process finished. Total func1_runs={func1_runs}, "
        f"func2_runs={func2_runs}, "
        f"final_unassessed_edges={len(unassessed_edges_global)}"
    )
    return output_ranking_path


# In[ ]:


final_ranking_path = hybrid_refine_func1_func2(
    csv_path="/mmfs1/home/sv96/Feedback-arc-set-paper/datasets/connectome_graph.csv",
    initial_ranking_path="/mmfs1/home/sv96/Feedback-arc-set-paper/datasets/35435948_ranking-24cpu.csv",
    output_ranking_path="/mmfs1/home/sv96/Feedback-arc-set-paper/datasets/35435948_ranking-24cpu-second72.csv",
    # 👈 ONE CSV
    log_path="/mmfs1/home/sv96/Feedback-arc-set-paper/datasets//connectome_35435948_changes_log-24cpu-second72.txt",
    h_hours=72.0,
    c=1  # use top 10% heaviest backward edges in the initial func1
)

# In[ ]:




