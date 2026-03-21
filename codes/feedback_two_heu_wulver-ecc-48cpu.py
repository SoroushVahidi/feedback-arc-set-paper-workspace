#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

#from google.colab import drive
#!pip install openpyxl
import pandas as pd



# === Mount Google Drive ===
#drive.mount('/content/drive', force_remount=True)

def read_graph(file_path):
    edges = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Strip whitespace and skip empty lines
            line = line.strip()
            if not line:
                continue
            
            # Skip comments ('c') and problem definition ('p') lines
            # We derive the nodes dynamically from the 'a' lines, 
            # so strict parsing of 'p' is not required for the topology.
            if line.startswith('c') or line.startswith('p'):
                continue
            
            # Parse arc lines: a <source> <target> <weight> <transit_time>
            if line.startswith('a'):
                parts = line.split()
                # parts[0] is 'a'
                # parts[1] is source
                # parts[2] is target
                # parts[3] is weight
                # parts[4] is transit_time (ignored)
                
                u = parts[1]
                v = parts[2]
                w = float(parts[3])
                
                edges.append((u, v, w))

    # The following logic remains identical to the original function
    # to ensure the return types (dictionaries, lists) are consistent.
    
    # Create a sorted set of unique nodes to establish a deterministic index mapping
    node_set = sorted(set(u for u, v, _ in edges).union(v for u, v, _ in edges))
    
    # Map node IDs (strings) to 0-based integers
    node_to_index = {node: i for i, node in enumerate(node_set)}
    index_to_node = {i: node for node, i in node_to_index.items()}
    
    # Rebuild edges using the internal 0-based indices
    edges_indexed = [(node_to_index[u], node_to_index[v], w) for (u, v, w) in edges]
    
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

import random




def compute_forward_weight(edges, scores):
    return sum(w for u, v, w in edges if scores[u] < scores[v])

import datetime

def log_message(message, log_file_path):
    """
    Appends a message with a timestamp to the specified log file.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}\n"
    
    # 'a' mode opens the file for appending; creates the file if it doesn't exist.
    with open(log_file_path, "a") as f:
        f.write(formatted_message)
    
    # Optional: Print to console as well so you can see progress in real-time
    #print(formatted_message.strip())

# Example usage within your workflow:
# log_message("SCC decomposition started.", log_path)
# log_message("Hybrid function iteration 1 complete.", log_path)



# In[2]:


def compare_backward_weights(edges, sbef, saft):
    """
    Given edges and two score dictionaries (sbef and saft),
    returns:
    - sum of backward edge weights in sbef that are NOT backward in saft (Improved)
    - sum of backward edge weights in saft that are NOT backward in sbef (Regressed)
    - sum of backward edges in both
    - sum of forward edges in both
    """
    sum_sbef_backward_only = 0.0
    sum_saft_backward_only = 0.0
    sum_both_backward = 0.0
    sum_both_forward = 0.0
    
    # Set to track edges that were backward ONLY in the 'before' state
    only_backward_in_before = set()

    for u, v, w in edges:
        # Check backward condition: score[u] > score[v]
        is_backward_sbef = sbef[u] > sbef[v]
        is_backward_saft = saft[u] > saft[v]

        if is_backward_sbef and not is_backward_saft:
            # It was backward, now it is forward (Improvement)
            sum_sbef_backward_only += w
            only_backward_in_before.add((u, v, w))
            
        elif is_backward_saft and not is_backward_sbef:
            # It was forward, now it is backward (Regression)
            sum_saft_backward_only += w
            
        elif is_backward_sbef and is_backward_saft:
            # Still backward
            sum_both_backward += w
            
        else:
            # Not backward in either (Forward in both)
            sum_both_forward += w

    return sum_sbef_backward_only, sum_saft_backward_only, sum_both_backward, sum_both_forward, only_backward_in_before

def all_scores_unique(scores):
    """
    Returns True if all nodes have unique scores, otherwise False.
    """
    values = list(scores.values())
    return len(values) == len(set(values))

def global_scc_topo_reorder_scores(G, edges, scores, log_path, debug=False, total_weight=None):
    """
    Reorder 'scores' so that each SCC's vertices are contiguous in rank and SCCs
    appear in a topological order of the condensation DAG.

    CRITERION: Backward Weight (BW) must NOT increase.

    Logging:
      - Uses ONLY log_message(message, log_path). No prints.

    BW computation:
      - If total_weight is provided, BW is computed as BW = total_weight - FW
        using compute_forward_weight (fast + consistent).
      - Otherwise, falls back to compute_backward_weight(edges, scores).
    """
    import networkx as nx

    # -------------------- Helpers: BW via total_weight - FW --------------------
    def _bw_of(scores_map):
        if total_weight is not None:
            fw = float(compute_forward_weight(edges, scores_map))
            return float(total_weight) - fw
        return float(compute_backward_weight(edges, scores_map))

    if debug:
        log_message("global_scc_topo_reorder_scores: running Global SCC topo reorder...", log_path)

    # 1) Identify SCCs
    sccs = list(nx.strongly_connected_components(G))
    if not sccs:
        # Empty graph: keep scores as-is
        if debug:
            log_message("global_scc_topo_reorder_scores: no SCCs found (empty graph). Returning original scores.", log_path)
        return scores.copy()

    # 2) Map nodes to SCC IDs
    node_to_scc = {}
    for scc_id, scc in enumerate(sccs):
        for n in scc:
            node_to_scc[n] = scc_id

    # 3) Build the Condensation DAG (SCC Graph)
    scc_dag = nx.DiGraph()
    scc_dag.add_nodes_from(range(len(sccs)))

    # Add edges between SCCs based on original edges
    for u, v, _w in edges:
        su = node_to_scc[u]
        sv = node_to_scc[v]
        if su != sv:
            scc_dag.add_edge(su, sv)

    # 4) Topological Sort of SCCs
    scc_order = list(nx.topological_sort(scc_dag))

    # 5) BW BEFORE reordering
    bw_before = _bw_of(scores)

    # 6) Generate New Scores:
    #    Preserve order inside SCC by old rank; order SCC blocks by topo order
    new_scores = {}
    next_rank = 0

    for scc_id in scc_order:
        nodes_in_scc = list(sccs[scc_id])
        nodes_in_scc_sorted = sorted(nodes_in_scc, key=lambda n: scores[n])
        for n in nodes_in_scc_sorted:
            new_scores[n] = next_rank
            next_rank += 1

    # 7) BW AFTER reordering
    bw_after = _bw_of(new_scores)

    # 8) Validation: BW must NOT increase
    if bw_after > bw_before + 1e-9:
        error_msg = (
            "❌ global_scc_topo_reorder_scores: Global SCC topo reorder INCREASED BW: "
            f"before={bw_before:.6f}, after={bw_after:.6f}"
        )
        log_message(error_msg, log_path)
        raise RuntimeError(error_msg)

    if debug:
        log_message(
            f"global_scc_topo_reorder_scores: done. BW_before={bw_before:.6f}, BW_after={bw_after:.6f} (non-increasing).",
            log_path
        )

    return new_scores


# In[3]:


def apply_new_strategy(scores, u, v, between, out_edges, in_edges, edges_dict, log_path, debug=False):
    """
    Optimizes the position of u and v within the block [v] + between + [u].

    Notes on objective:
      - Increasing forward-edge weight by Δ implies decreasing backward-edge weight by the same Δ,
        because total edge weight is constant. So we treat the improvement value returned here as
        "backward-weight reduction" (BW reduction).
    """
    import time

    PROFILE_THRESHOLD = 0.08  # seconds
    t_total0 = time.time()

    # Scores are modified in-place during reorder step.
    # We only snapshot the ranks we need (do NOT copy full dict for performance).
    ranks_before = (scores[u], scores[v])

    # ----------------- 1) Define neighbors of v and u within 'between' -----------------
    # 'between' is assumed sorted by current rank, and list comprehension preserves that order.
    v_neighbors_sorted = [
        node for node in between
        if (node, v) in edges_dict or (v, node) in edges_dict
    ]
    u_neighbors_sorted = [
        node for node in between
        if (node, u) in edges_dict or (u, node) in edges_dict
    ]

    # ----------------- 2) Prefix/Suffix sums -----------------
    # Prefix sums over v_neighbors (from left to right)
    outgoing_v = [0.0] * (len(v_neighbors_sorted) + 1)
    ingoing_v = [0.0] * (len(v_neighbors_sorted) + 1)
    for i, node in enumerate(v_neighbors_sorted):
        outgoing_v[i + 1] = outgoing_v[i] + edges_dict.get((v, node), 0.0)
        ingoing_v[i + 1] = ingoing_v[i] + edges_dict.get((node, v), 0.0)

    # Suffix sums over u_neighbors (from right to left)
    outgoing_u = [0.0] * (len(u_neighbors_sorted) + 1)
    ingoing_u = [0.0] * (len(u_neighbors_sorted) + 1)
    for i in reversed(range(len(u_neighbors_sorted))):
        node_i = u_neighbors_sorted[i]
        outgoing_u[i] = outgoing_u[i + 1] + edges_dict.get((u, node_i), 0.0)
        ingoing_u[i] = ingoing_u[i + 1] + edges_dict.get((node_i, u), 0.0)

    # ----------------- 3) Two-pointer sweep to find best cut -----------------
    best_bw_reduction = float("-inf")
    best_y = -1
    best_j = -1

    w_uv = edges_dict.get((u, v), 0.0)
    w_vu = edges_dict.get((v, u), 0.0)

    # ΔFW for swapping u/v relative to each other; equals BW reduction by same amount.
    base_bw_reduction = w_uv - w_vu

    u_scores_list = [scores[node] for node in u_neighbors_sorted]
    len_u = len(u_scores_list)
    j = 0

    for i, nv in enumerate(v_neighbors_sorted):
        s = scores[nv]

        # Advance j to maintain rank monotonicity
        while j < len_u and u_scores_list[j] <= s:
            j += 1

        if j < len_u:
            # BW reduction = ΔFW (because total weight is constant)
            bw_reduction = (
                base_bw_reduction
                + ingoing_v[i + 1] - outgoing_v[i + 1]
                - ingoing_u[j] + outgoing_u[j]
            )

            if bw_reduction > best_bw_reduction:
                best_bw_reduction = bw_reduction
                best_y = i
                best_j = j

    # ----------------- 4) Early exit if no improvement -----------------
    if best_bw_reduction <= 0:
        ranks_after = (scores[u], scores[v])
        return False, 0.0, ranks_before, ranks_after, {}

    # ----------------- 5) Apply reordering (success case) -----------------
    # best_y is index in v_neighbors_sorted; map to index in 'between'
    split_node = v_neighbors_sorted[best_y]
    split_index = between.index(split_node)

    left = between[:split_index + 1]
    right = between[split_index + 1:]
    new_order = left + [u, v] + right

    # Assign new scores sequentially starting from the original score of v
    base_score = scores[v]
    for k, node in enumerate(new_order):
        scores[node] = base_score + k

    ranks_after = (scores[u], scores[v])

    # ----------------- 6) Profiling Log -----------------
    t_total = time.time() - t_total0
    if t_total > PROFILE_THRESHOLD:
        log_message(
            f"SLOW success: u={u}, v={v}, |between|={len(between)}, "
            f"bw_reduction={best_bw_reduction:.3f}, total_time={t_total:.4f}s",
            log_path
        )

    return True, best_bw_reduction, ranks_before, ranks_after, {}


# In[4]:


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
    in_u  = in_edges[u]
    out_v = out_edges[v]
    in_v  = in_edges[v]
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


# In[5]:


def select_nonconflicting_edge_indices_dp(backward_edges):
    """
    Given a list of backward_edges = (u, v, w, lo, hi) with lo < hi (ranks),
    select a maximum-size subset of edges whose [lo, hi] intervals do not overlap.

    IMPORTANT:
      - This selects a maximum-cardinality set (not weight-based). The tuple
        contains w only because callers often carry it around for later
        backward-weight accounting, but it is NOT used here.
      - We compress coordinates so DP is defined only on ranks that are
        endpoints of some edge (lo or hi), not on [0..max_rank].

    Returns:
      selected_indices: set of LOCAL indices (0..len(backward_edges)-1)
                        of chosen edges.

    Complexity per call:
      O(E log E) dominated by sorting endpoints (≤ 2E) and sorting edges by hi.
      DP itself is O(E + K), K ≤ 2E.
    """
    if not backward_edges:
        return set()

    # 1) Collect all distinct endpoints (lo, hi)
    endpoints = set()
    for (_, _, _, lo, hi) in backward_edges:
        endpoints.add(lo)
        endpoints.add(hi)

    sorted_endpoints = sorted(endpoints)
    coord_index = {rank: i for i, rank in enumerate(sorted_endpoints)}
    K = len(sorted_endpoints)

    # 2) Convert edges to compressed coordinates and sort by hi_pos
    #    We keep local index so we can return indices into backward_edges.
    edges = []
    for idx, (u, v, w, lo, hi) in enumerate(backward_edges):
        lo_pos = coord_index[lo]
        hi_pos = coord_index[hi]
        edges.append((hi_pos, lo_pos, idx))

    # Sort by end coordinate
    edges.sort(key=lambda x: x[0])

    # Helper: for each edge in sorted order, find the last compatible edge index
    # (i.e., previous edge with hi_pos < current lo_pos). We use binary search on hi_pos list.
    hi_list = [e[0] for e in edges]

    def _rightmost_compatible_edge_index(lo_pos):
        # Return largest j such that hi_list[j] < lo_pos, or -1 if none.
        # (strict < avoids overlap at a point; if you want allow touching, change to <=)
        import bisect
        j = bisect.bisect_left(hi_list, lo_pos) - 1
        return j

    # 3) Weighted-interval-scheduling DP but with weight=1 for every edge
    n = len(edges)
    dp = [0] * (n + 1)         # dp[i] = best using first i edges (edges[0..i-1])
    take = [False] * (n + 1)   # whether we take edge i-1

    # Precompute p(i): index of last compatible edge for edge i-1 (0-based in edges)
    p = [-1] * n
    for i in range(n):
        hi_pos, lo_pos, _ = edges[i]
        p[i] = _rightmost_compatible_edge_index(lo_pos)

    for i in range(1, n + 1):
        # Option 1: skip edge i-1
        opt1 = dp[i - 1]

        # Option 2: take edge i-1
        # then we add 1 + dp[p(i-1)+1]
        j = p[i - 1]
        opt2 = 1 + (dp[j + 1] if j >= 0 else 0)

        if opt2 > opt1:
            dp[i] = opt2
            take[i] = True
        else:
            dp[i] = opt1
            take[i] = False

    # 4) Reconstruct chosen edges (original local indices into backward_edges)
    selected_indices = set()
    i = n
    while i > 0:
        if take[i]:
            hi_pos, lo_pos, local_idx = edges[i - 1]
            selected_indices.add(local_idx)
            j = p[i - 1]
            i = j + 1
        else:
            i -= 1

    return selected_indices


# In[6]:


def worker_loop(worker_idx, num_procs,
                backward_edges,
                scores_snapshot,
                out_edges, in_edges, edges_dict,
                active_intervals,   # unused now
                used_intervals,     # unused now
                lock,               # unused now
                edge_queue,
                result_queue,
                log_path):
    """
    Worker processes backward_edges indices from edge_queue, tries:
      1) apply_new_strategy (returns BW reduction)
      2) greedy fallback (assumed to return BW reduction; if it returns FW gain, it's identical)

    Logging:
      - MUST use only log_message(..., log_path)
      - No other printing/logging is used here.
    """
    import time

    # Thresholds for profiling logs (seconds)
    BETWEEN_WARN = 0.05
    APPLY_WARN = 0.10
    LOCK_FRACTION_WARN = 0.25  # lock is unused now; kept for compatibility
    BETWEEN_SAFETY_CHECK_LIMIT = 5
    between_checks_done = 0

    # -------------------- SNAPSHOT / RANK ARRAY --------------------
    max_rank = max(scores_snapshot.values())
    rank_to_node = [None] * (max_rank + 1)

    for node, r in scores_snapshot.items():
        if r < 0 or r > max_rank:
            log_message(f"RUNTIME ERROR: Invalid rank in scores_snapshot: node={node}, rank={r}", log_path)
            raise RuntimeError("Invalid rank in scores_snapshot.")
        if rank_to_node[r] is not None:
            log_message(
                f"RUNTIME ERROR: Duplicate rank in scores_snapshot: rank={r}, "
                f"nodes={rank_to_node[r]} and {node}",
                log_path
            )
            raise RuntimeError("Non-injective ranks in scores_snapshot.")
        rank_to_node[r] = node

    # Quick sanity: ensure used ranks map to non-None nodes
    for r in set(scores_snapshot.values()):
        if rank_to_node[r] is None:
            log_message(f"RUNTIME ERROR: rank_to_node[{r}] is None while some node has rank {r}", log_path)
            raise RuntimeError("rank_to_node inconsistent with scores_snapshot.")

    worker_start = time.time()

    total_lock_time = 0.0  # lock is unused now, kept for summary compatibility
    total_apply_strategy_time = 0.0
    total_greedy_time = 0.0
    total_queue_wait_time = 0.0
    total_edge_processing_time = 0.0

    edges_processed = 0
    successes = 0
    failures = 0
    extended_successes = 0
    greedy_successes = 0

    scores_base = scores_snapshot
    out_edges_base = out_edges
    in_edges_base = in_edges
    edges_array = backward_edges

    # Helper: call compute_all_gains_local with/without log_path, depending on its signature.
    def _compute_gains(scores_base_local, u0, v0, out_e, in_e, log_file_path):
        try:
            # Preferred: allow function to log using log_message via passed path, if it supports it.
            return compute_all_gains_local(scores_base_local, u0, v0, out_e, in_e, log_file_path)
        except TypeError:
            # Backward compatible: original signature without log_path
            return compute_all_gains_local(scores_base_local, u0, v0, out_e, in_e)

    while True:
        # -----------------------------------------------
        # 1) POP EDGE INDEX FROM QUEUE (non-blocking)
        # -----------------------------------------------
        t0 = time.time()
        try:
            idx = edge_queue.get_nowait()
        except Exception:
            total_queue_wait_time += (time.time() - t0)
            break
        total_queue_wait_time += (time.time() - t0)

        edge_start = time.time()
        edges_processed += 1

        # Default values for safe exception handling
        u = v = None
        lo = hi = None
        mode = "none"
        success = False
        delta_bw_reduction = 0.0
        changed_scores = {}
        t_between = 0.0
        t_apply_dur = 0.0
        greedy_dur = 0.0

        try:
            u, v, w, lo, hi = edges_array[idx]

            idx_u = scores_base[u]
            idx_v = scores_base[v]

            # This worker only acts on edges that are backward in this snapshot
            # (u appears after v -> idx_u > idx_v).
            if not (idx_u > idx_v):
                failures += 1
                continue

            # -----------------------------------------------
            # Build "between" using rank_to_node slice
            # -----------------------------------------------
            t_btw = time.time()
            if idx_u - idx_v > 1:
                slice_raw = rank_to_node[idx_v + 1: idx_u]
                between = [node for node in slice_raw if node is not None]
            else:
                between = []
            t_between = time.time() - t_btw

            if t_between > BETWEEN_WARN:
                log_message(f"'between' build slow: len={len(between)} time={t_between:.6f}s", log_path)

            # Optional safety check vs slow method (first few calls)
            if between_checks_done < BETWEEN_SAFETY_CHECK_LIMIT:
                between_checks_done += 1
                slow_between = [node for node, r in scores_base.items() if idx_v < r < idx_u]
                if set(between) != set(slow_between):
                    log_message("RUNTIME ERROR: Fast 'between' does not match slow method.", log_path)
                    raise RuntimeError("Fast 'between' implementation mismatch detected.")

            # Local copy for this edge attempt
            scores = scores_base.copy()

            # -----------------------------------------------
            # 2) Extended strategy (returns BW reduction)
            # -----------------------------------------------
            t_apply = time.time()
            try:
                ext_success, ext_bw_reduction, rB, rA, dbg = apply_new_strategy(
                    scores, u, v, between,
                    out_edges_base, in_edges_base, edges_dict,
                    log_path,
                    debug=False
                )
                success = bool(ext_success)
                delta_bw_reduction = float(ext_bw_reduction)
                mode = "extended"
            except Exception as e:
                log_message(f"ERROR apply_new_strategy({u}->{v}): {repr(e)}", log_path)
                success = False
                delta_bw_reduction = 0.0
                mode = "extended_error"

            t_apply_dur = time.time() - t_apply
            total_apply_strategy_time += t_apply_dur

            if t_apply_dur > APPLY_WARN:
                log_message(f"apply_new_strategy slow for edge ({u}->{v}): {t_apply_dur:.6f}s", log_path)

            if success and delta_bw_reduction <= 0.0:
                log_message(
                    f"RUNTIME WARNING: success=True but bw_reduction={delta_bw_reduction:.6f} for ({u}->{v}). "
                    f"Treating as failure.",
                    log_path
                )
                success = False
                delta_bw_reduction = 0.0
                mode = "extended_nonpositive_delta"

            # -----------------------------------------------
            # 3) Greedy fallback
            # -----------------------------------------------
            if not (success and delta_bw_reduction > 0.0):
                t_g = time.time()
                g1, g2, g3 = _compute_gains(scores_base, u, v, out_edges_base, in_edges_base, log_path)
                greedy_dur = time.time() - t_g
                total_greedy_time += greedy_dur

                best_gain = max(g1, g2, g3)
                if best_gain > 0:
                    scores = scores_base.copy()
                    delta_bw_reduction = float(best_gain)
                    success = True

                    if best_gain == g1:
                        # swap u and v
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
                    delta_bw_reduction = 0.0
                    mode = "none" if mode != "extended_error" else "none_after_error"

            # -----------------------------------------------
            # 4) Collect rank changes within [lo, hi]
            # -----------------------------------------------
            if success and delta_bw_reduction > 0.0:
                for node, r_old in scores_base.items():
                    if lo <= r_old <= hi:
                        r_new = scores[node]
                        if r_new != r_old:
                            changed_scores[node] = r_new

                if not changed_scores:
                    log_message(
                        f"RUNTIME ERROR: success with positive bw_reduction but no changed_scores for edge ({u}->{v})",
                        log_path
                    )
                    success = False
                    delta_bw_reduction = 0.0
                    mode = f"{mode}_no_changes"

                # Validate: any changed node must have old rank inside [lo,hi]
                for node in changed_scores.keys():
                    old_r = scores_base[node]
                    if not (lo <= old_r <= hi):
                        log_message(
                            f"RUNTIME ERROR: Node {node} changed but old rank {old_r} outside interval [{lo},{hi}]",
                            log_path
                        )
                        raise RuntimeError("Changed rank outside edge interval.")

            # -----------------------------------------------
            # 5) Accounting + result send
            # -----------------------------------------------
            if success and delta_bw_reduction > 0.0:
                successes += 1
                if mode.startswith("extended"):
                    extended_successes += 1
                elif mode.startswith("greedy"):
                    greedy_successes += 1

                # log_message(
                #     f"SUCCESS ({u}->{v}) BW_Reduct={delta_bw_reduction:.4f} mode={mode} "
                #     f"interval=({lo},{hi}) changed={len(changed_scores)}",
                #     log_path
                # )
            else:
                failures += 1

            edge_total_time = time.time() - edge_start
            total_edge_processing_time += edge_total_time

            result_queue.put({
                "worker_idx": worker_idx,
                "u": u,
                "v": v,
                "success": bool(success and delta_bw_reduction > 0.0),
                "delta": float(delta_bw_reduction),   # BW reduction
                "changed_scores": changed_scores,
                "mode": mode,
                "interval": (lo, hi),
                "timing": {
                    "apply_new_strategy": float(t_apply_dur),
                    "greedy_time": float(greedy_dur),
                    "between_list_time": float(t_between),
                    "edge_total_time": float(edge_total_time),
                },
            })

        except Exception as e:
            failures += 1
            edge_total_time = time.time() - edge_start
            total_edge_processing_time += edge_total_time

            log_message(
                f"UNHANDLED ERROR processing edge index {idx} ({u}->{v}): {repr(e)}",
                log_path
            )

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
                    "edge_total_time": float(edge_total_time),
                },
            })

    # -----------------------------------------------
    # Worker summary
    # -----------------------------------------------
    runtime = time.time() - worker_start
    lock_fraction = (total_lock_time / runtime) if runtime > 0 else 0.0

    if lock_fraction > LOCK_FRACTION_WARN:
        log_message(f"WARNING: High lock contention: {lock_fraction * 100:.2f}% of time", log_path)

    result_queue.put({
        "worker_idx": worker_idx,
        "done": True,
        "edges_processed": edges_processed,
        "successes": successes,
        "extended_successes": extended_successes,
        "greedy_successes": greedy_successes,
        "failures": failures,
        "runtime": float(runtime),
        "timing_totals": {
            "queue_wait": float(total_queue_wait_time),
            "lock_time": float(total_lock_time),
            "apply_total": float(total_apply_strategy_time),
            "greedy_total": float(total_greedy_time),
            "edge_processing_total": float(total_edge_processing_time),
        },
    })


# In[7]:


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

# In[8]:


# def run_dynamic_round(edges,
#                       scores,
#                       out_edges,
#                       in_edges,
#                       edges_dict,
#                       num_procs,
#                       backward_weight,     # tracked BW (sum of weights of backward edges)
#                       improvement_counter,
#                       index_to_node,
#                       output_excel,
#                       log_path,
#                       BW_SANITY_EVERY_IMPROVEMENTS=5,
#                       edge_subset=None):
#     """
#     Runs one dynamic round:
#       - Build current backward edges (u,v,w) where rank(u) > rank(v)
#       - Repeatedly select a non-overlapping subset via DP
#       - Dispatch each subset to workers to propose local reorders
#       - Apply successful changes and update *backward_weight* by subtracting BW reductions

#     Logging:
#       - Only uses log_message(..., log_path)
#       - No other prints/loggers

#     Sanity:
#       - Periodically recompute BW as: total_edge_weight - forward_weight(scores)
#         (since total weight is constant and ranks are injective)
#     """
#     import math
#     import time
#     import multiprocessing as mp

#     # ----------------- WARN THRESHOLDS -----------------
#     BUILD_BACKWARD_WARN = 10.0
#     SPAWN_WORKERS_WARN = 5.0
#     RESULT_LOOP_WARN = 60.0
#     JOIN_WARN = 5.0
#     ROUND_TIME_WARN = 600.0
#     AVG_WORKER_EDGE_WARN = 0.20
#     LOW_SUCCESS_RATIO_WARN = 0.02
#     BW_MISMATCH_TOL = 1e-6

#     # Total weight is constant → BW = total - FW
#     total_edge_weight = 0.0
#     for (_, _, w) in edges:
#         total_edge_weight += float(w)

#     # ----------------- HELPER: SAVE RANKING -----------------
#     def save_ranking_snapshot_local(scores_dict, idx_to_node, path):
#         # Lazy import to avoid paying import cost if never called
#         import pandas as pd

#         items = sorted(scores_dict.items(), key=lambda kv: kv[1])  # sort by rank
#         rows = []
#         for node_idx, rank in items:
#             rows.append({"Node ID": str(idx_to_node[node_idx]), "Order": int(rank)})

#         df = pd.DataFrame(rows)
#         lower = path.lower()
#         if lower.endswith(".csv"):
#             df.to_csv(path, index=False)
#         else:
#             df.to_excel(path, index=False)

#     round_start = time.time()
#     BW_before_round = float(backward_weight)

#     # ----------------- TIMING ACCUMULATORS -----------------
#     t_build_backward = 0.0
#     t_spawn_workers_total = 0.0
#     t_result_loop_total = 0.0
#     t_join_total = 0.0
#     t_dp_total = 0.0

#     # Worker timing aggregation
#     total_apply_strategy_time_workers = 0.0
#     total_greedy_time_workers = 0.0
#     total_between_time_workers = 0.0
#     total_edge_time_workers = 0.0
#     counted_timing_results_total = 0

#     total_delta_round = 0.0  # total BW reduction in this round (positive)
#     applied_edges_total = 0
#     results_processed_total = 0

#     # ----- Normalize edge_subset → subset_pairs of (u,v) -----
#     subset_pairs = None
#     if edge_subset is not None:
#         norm_set = set()
#         for e in edge_subset:
#             if isinstance(e, tuple) and len(e) >= 2:
#                 norm_set.add((e[0], e[1]))
#         subset_pairs = norm_set

#     # ----- Build backward edges for this sweep -----
#     t0 = time.time()

#     backward_edges = []  # list of (u, v, w, lo, hi)
#     skipped_by_subset = 0
#     skipped_not_backward = 0

#     for (u, v, w) in edges:
#         if subset_pairs is not None and (u, v) not in subset_pairs:
#             skipped_by_subset += 1
#             continue

#         ru, rv = scores[u], scores[v]
#         if ru > rv:
#             lo = rv
#             hi = ru
#             backward_edges.append((u, v, float(w), lo, hi))
#         else:
#             skipped_not_backward += 1

#     backward_edges.sort(key=lambda x: x[2], reverse=True)
#     n_edges = len(backward_edges)
#     t_build_backward = time.time() - t0

#     if t_build_backward > BUILD_BACKWARD_WARN:
#         log_message(
#             f"WARNING: Building backward edges took {t_build_backward:.4f}s (>{BUILD_BACKWARD_WARN}s)",
#             log_path
#         )

#     if n_edges == 0:
#         total_round_time = time.time() - round_start
#         # log_message(f"No backward edges; ending dynamic round. total_round_time={total_round_time:.4f}s", log_path)
#         # log_message(
#         #     f"BW_after_round={backward_weight:.2f} (ΔBW_round={BW_before_round - backward_weight:.3f})",
#         #     log_path
#         # )
#         return scores, backward_weight, 0.0, 0, improvement_counter

#     # -------- shared structures (kept for signature compatibility) --------
#     manager = mp.Manager()
#     active_intervals = manager.list()  # unused now
#     used_intervals = manager.list()    # unused now
#     lock = manager.Lock()              # unused now

#     scores_snapshot = scores.copy()

#     # Iteratively pick independent subsets via DP
#     remaining_indices = list(range(n_edges))
#     batch_idx = 0

#     while remaining_indices:
#         batch_idx += 1
#         batch_start = time.time()

#         edges_in_pool = len(remaining_indices)
#        # log_message(f"DP batch {batch_idx}: edges_in_pool_before_batch={edges_in_pool}", log_path)

#         # --- Build local list of edges for DP ---
#         edges_for_dp = [backward_edges[i] for i in remaining_indices]

#         # --- DP to select a maximum non-conflicting subset ---
#         t_dp0 = time.time()
#         local_selected = select_nonconflicting_edge_indices_dp(edges_for_dp)
#         dp_time = time.time() - t_dp0
#         t_dp_total += dp_time

#         selected_global = [remaining_indices[i] for i in local_selected]
#         num_selected = len(selected_global)

#         # log_message(
#         #     f"(DP batch {batch_idx}) DP selection: total_edges={len(edges_for_dp)}, selected={num_selected}",
#         #     log_path
#         # )

#         if num_selected == 0:
#             # log_message(
#             #     f"(DP batch {batch_idx}) DP selected none; stopping with remaining={len(remaining_indices)}",
#             #     log_path
#             # )
#             break

#         # --- Build batch_edges list for workers ---
#         batch_edges = [backward_edges[i] for i in selected_global]
#         num_edges_batch = len(batch_edges)
#         num_procs_effective = min(num_procs, num_edges_batch)

#         # log_message(
#         #     f"(DP batch {batch_idx}) Spawning {num_procs_effective} workers for {num_edges_batch} edges.",
#         #     log_path
#         # )

#         edge_queue = manager.Queue()
#         result_queue = manager.Queue()

#         for i in range(num_edges_batch):
#             edge_queue.put(i)

#         # --- Spawn workers ---
#         t_spawn0 = time.time()
#         procs = []
#         for worker_idx in range(1, num_procs_effective + 1):
#             p = mp.Process(
#                 target=worker_loop,
#                 args=(
#                     worker_idx,
#                     num_procs_effective,
#                     batch_edges,
#                     scores_snapshot,
#                     out_edges,
#                     in_edges,
#                     edges_dict,
#                     active_intervals,
#                     used_intervals,
#                     lock,
#                     edge_queue,
#                     result_queue,
#                     log_path
#                 )
#             )
#             p.start()
#             procs.append(p)

#         t_spawn_workers = time.time() - t_spawn0
#         t_spawn_workers_total += t_spawn_workers
#         if t_spawn_workers > SPAWN_WORKERS_WARN:
#             log_message(
#                 f"WARNING: Spawning workers batch {batch_idx} took {t_spawn_workers:.4f}s (>{SPAWN_WORKERS_WARN}s)",
#                 log_path
#             )

#         # --- Collect results ---
#         t_res0 = time.time()
#         alive = num_procs_effective

#         total_delta_batch = 0.0  # BW reduction in this batch
#         applied_edges_batch = 0
#         results_processed_batch = 0

#         # Worker timing aggregation for this batch
#         batch_apply_time = 0.0
#         batch_greedy_time = 0.0
#         batch_between_time = 0.0
#         batch_edge_total_time = 0.0
#         counted_timing_results_batch = 0

#         last_progress_log = time.time()
#         BW_before_batch = float(backward_weight)

#         while alive > 0:
#             res = result_queue.get()

#             if res.get("done"):
#                 alive -= 1
#                 continue

#             results_processed_batch += 1
#             results_processed_total += 1

#             timing = res.get("timing")
#             if timing:
#                 ta = float(timing.get("apply_new_strategy", 0.0))
#                 tg = float(timing.get("greedy_time", 0.0))
#                 tb = float(timing.get("between_list_time", 0.0))
#                 te = float(timing.get("edge_total_time", 0.0))

#                 batch_apply_time += ta
#                 batch_greedy_time += tg
#                 batch_between_time += tb
#                 batch_edge_total_time += te
#                 counted_timing_results_batch += 1

#                 total_apply_strategy_time_workers += ta
#                 total_greedy_time_workers += tg
#                 total_between_time_workers += tb
#                 total_edge_time_workers += te
#                 counted_timing_results_total += 1

#             if not res.get("success"):
#                 if time.time() - last_progress_log > 5.0:
#                     last_progress_log = time.time()
#                 continue

#             # Success: apply changes
#             delta = float(res["delta"])  # POSITIVE BW reduction
#             changed = res["changed_scores"]
#             u = res["u"]
#             v = res["v"]
#             mode = res.get("mode", "unknown")

#             for node, nrank in changed.items():
#                 scores[node] = nrank

#             backward_weight -= delta
#             total_delta_batch += delta
#             total_delta_round += delta
#             applied_edges_batch += 1
#             applied_edges_total += 1
#             improvement_counter += 1

#             # log_message(
#             #     f"(DP batch {batch_idx}) SUCCESS ({u}->{v}) BW_Reduct={delta:.3f} mode={mode} BW={backward_weight:.2f}",
#             #     log_path
#             # )

#             # Periodic sanity check + checkpoint save
#             if BW_SANITY_EVERY_IMPROVEMENTS > 0 and (improvement_counter % BW_SANITY_EVERY_IMPROVEMENTS == 0):
#                 # Compute forward weight, then derive backward weight from total
#                 fw = 0.0
#                 for (u2, v2, w2) in edges:
#                     if scores[u2] < scores[v2]:
#                         fw += float(w2)
#                 real_bw = total_edge_weight - fw

#                 if abs(real_bw - backward_weight) > BW_MISMATCH_TOL:
#                     log_message(
#                         f"RUNTIME ERROR: BW mismatch! tracked={backward_weight:.6f}, recomputed={real_bw:.6f}",
#                         log_path
#                     )
#                     raise RuntimeError("Backward-weight mismatch detected before saving ranking.")

#                 backward_weight = real_bw  # sync

#                 try:
#                     save_ranking_snapshot_local(scores, index_to_node, output_excel)
#                 except Exception as e:
#                     log_message(
#                         f"WARNING: Failed to save intermediate ranking to {output_excel}: {repr(e)}",
#                         log_path
#                     )

#             if time.time() - last_progress_log > 5.0:
#                 last_progress_log = time.time()

#         t_result_loop = time.time() - t_res0
#         t_result_loop_total += t_result_loop
#         if t_result_loop > RESULT_LOOP_WARN:
#             log_message(
#                 f"WARNING: Result collection loop batch {batch_idx} took {t_result_loop:.4f}s (>{RESULT_LOOP_WARN}s)",
#                 log_path
#             )

#         # Join workers
#         t_join0 = time.time()
#         for p in procs:
#             p.join()
#         t_join = time.time() - t_join0
#         t_join_total += t_join
#         if t_join > JOIN_WARN:
#             log_message(
#                 f"WARNING: Joining workers batch {batch_idx} took {t_join:.4f}s (>{JOIN_WARN}s)",
#                 log_path
#             )

#         # Remove selected edges from remaining pool
#         selected_global_set = set(selected_global)
#         remaining_indices = [i for i in remaining_indices if i not in selected_global_set]

#         # Batch summary
#         batch_total_time = time.time() - batch_start
#         # log_message(f"(DP batch {batch_idx}) BATCH SUMMARY:", log_path)
#         # log_message(f"    edges={num_edges_batch}, applied={applied_edges_batch}, BW_Reduct_total={total_delta_batch:.3f}", log_path)
#         # log_message(f"    BW_before={BW_before_batch:.2f}, BW_after={backward_weight:.2f}", log_path)
#         # log_message(f"    batch_total_time={batch_total_time:.4f}s", log_path)

#     # --------------- DYNAMIC ROUND SUMMARY ---------------
#     total_round_time = time.time() - round_start
#     avg_gain = (total_delta_round / applied_edges_total) if applied_edges_total > 0 else 0.0
#     edges_per_sec = (applied_edges_total / total_round_time) if total_round_time > 0 else 0.0
#     avg_worker_edge_time = (
#         total_edge_time_workers / counted_timing_results_total
#         if counted_timing_results_total > 0 else float('nan')
#     )

#     # log_message("DYNAMIC ROUND SUMMARY:", log_path)
#     # log_message(
#     #     f"    BW_before={BW_before_round:.2f}, BW_after={backward_weight:.2f}, BW_Reduct_round={total_delta_round:.3f}",
#     #     log_path
#     # )
#     # log_message(f"    applied_edges={applied_edges_total}, avg_bw_reduct_per_edge={avg_gain:.4f}", log_path)
#     # log_message(f"    total_round_time={total_round_time:.4f}s, edges_per_sec={edges_per_sec:.2f}", log_path)
#     # log_message(
#     #     f"    worker_timing_total: apply≈{total_apply_strategy_time_workers:.4f}s, greedy≈{total_greedy_time_workers:.4f}s",
#     #     log_path
#     # )

#     # --------------- WARNINGS ---------------
#     if total_round_time > ROUND_TIME_WARN:
#         log_message(f"WARNING: Dynamic round took {total_round_time:.4f}s (>{ROUND_TIME_WARN}s)", log_path)

#     if not math.isnan(avg_worker_edge_time) and avg_worker_edge_time > AVG_WORKER_EDGE_WARN:
#         log_message(f"WARNING: avg_worker_edge_time={avg_worker_edge_time:.4f}s (>{AVG_WORKER_EDGE_WARN}s)", log_path)

#     if results_processed_total > 0:
#         success_ratio = applied_edges_total / results_processed_total
#         if success_ratio < LOW_SUCCESS_RATIO_WARN:
#             log_message(
#                 f"WARNING: Very low success ratio: {success_ratio:.4%} ({applied_edges_total}/{results_processed_total})",
#                 log_path
#             )

#    # log_message("END dynamic round (DP-based independent subsets)", log_path)

#     return scores, backward_weight, total_delta_round, applied_edges_total, improvement_counter


# In[9]:


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
    rows.sort(key=lambda x: x[0])   # sort by Order

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Node ID", "Order"])  # required header

        for order, node_idx in rows:
            node_label = index_to_node[node_idx]  # original ID string
            writer.writerow([node_label, int(order)])


# 

# In[10]:


def refine_ranking_parallel_dynamic(
    csv_path,
    initial_ranking_path,
    output_excel,
    log_path,
    MAX_HOURS: float = 72.0,
    LOG_EVERY_ROUND: int = 300,
    BW_CHECK_EVERY_ROUND: int = 400,
    edge_subset=None,   # None => global; otherwise list of edges; we normalize to (u,v)
    # -------- NEW --------
    MAX_DP_BATCHES_PER_CALL: int | None = None,  # None => unlimited (original behavior)
    NO_IMPROVEMENT_TIME_LIMIT_SEC: float = 600.0,
    EPS_IMPROVE: float = 1e-12,
):
    """
    Same behavior as your current version, BUT:
      - MAX_DP_BATCHES_PER_CALL lets the caller force “only K DP batches then return”
        (needed for your hybrid cycle logic).
      - Adds a couple of high-signal logs so you can confirm:
          * batch sizes,
          * whether DP keeps selecting new intervals,
          * whether we’re improving at all.
    """
    import time
    import os
    import multiprocessing as mp

    pid_main = os.getpid()
    try:
        num_procs = len(os.sched_getaffinity(0))
    except Exception:
        num_procs = os.cpu_count() or 1

    # ------------------------------
    # Load graph & initial scores
    # ------------------------------
    edges, node_to_index, index_to_node = read_graph(csv_path)
    scores = load_initial_scores(initial_ranking_path, node_to_index)

    total_weight = 0.0
    for (_, _, w) in edges:
        total_weight += float(w)

    fw0 = compute_forward_weight(edges, scores)
    bw0 = total_weight - float(fw0)

    ratio_bw = (bw0 / total_weight) if total_weight > 0 else 0.0
    log_message(f"[FUNC1] pid={pid_main} procs={num_procs} | init BW/total={bw0:.2f}/{total_weight:.2f}={ratio_bw:.6f}", log_path)

    # ------------------------------
    # Build adjacency & edges_dict
    # ------------------------------
    out_edges = {}
    in_edges = {}
    edges_dict = {}
    for (u, v, w) in edges:
        out_edges.setdefault(u, []).append((v, float(w)))
        in_edges.setdefault(v, []).append((u, float(w)))
        edges_dict[(u, v)] = float(w)

    # ------------------------------
    # Normalize edge_subset → subset_pairs of form (u,v)
    # ------------------------------
    subset_pairs = None
    if edge_subset is not None:
        raw_len = len(edge_subset)
        norm_set = set()
        malformed = 0
        for e in edge_subset:
            if isinstance(e, tuple) and len(e) >= 2:
                norm_set.add((e[0], e[1]))
            else:
                malformed += 1
        subset_pairs = norm_set
        log_message(f"[FUNC1] edge_subset raw={raw_len} normalized={len(subset_pairs)} malformed={malformed}", log_path)

    # ------------------------------
    # Fixed candidate pool for whole call
    # ------------------------------
    candidate_edges = []
    for (u, v, w) in edges:
        if subset_pairs is not None and (u, v) not in subset_pairs:
            continue
        candidate_edges.append((u, v, float(w)))

    if subset_pairs is not None and len(candidate_edges) == 0:
        log_message("[FUNC1] ERROR: edge_subset normalized non-empty but matches 0 edges in graph.", log_path)
        raise RuntimeError("edge_subset does not match any edges in this graph.")

    remaining_indices = list(range(len(candidate_edges)))

    best_scores = scores.copy()
    best_BW = float(bw0)
    BW_tracked = float(bw0)

    start_ts = time.time()
    deadline = start_ts + MAX_HOURS * 3600.0
    last_improvement_time = start_ts

    BW_MISMATCH_TOL = 1e-6
    round_idx = 0
    dp_batches_done = 0

    # ===================================================
    # MAIN LOOP: one DP batch per iteration (as before)
    # ===================================================
    while remaining_indices:
        now = time.time()
        if now >= deadline:
            log_message("[FUNC1] Time limit reached; stopping.", log_path)
            break

        # NEW: caller-forced cap on number of DP batches
        if (MAX_DP_BATCHES_PER_CALL is not None) and (dp_batches_done >= MAX_DP_BATCHES_PER_CALL):
            log_message(f"[FUNC1] Reached MAX_DP_BATCHES_PER_CALL={MAX_DP_BATCHES_PER_CALL}; returning early.", log_path)
            break

        round_idx += 1

        if LOG_EVERY_ROUND > 0 and (round_idx % LOG_EVERY_ROUND == 0):
            elapsed_h = (now - start_ts) / 3600.0
            time_left_h = max(0.0, (deadline - now) / 3600.0)
            since_improvement = now - last_improvement_time
            log_message(
                f"[FUNC1] round={round_idx} batches={dp_batches_done} | BW={BW_tracked:.2f} best={best_BW:.2f} | "
                f"remaining_candidates={len(remaining_indices)} | elapsed={elapsed_h:.2f}h left={time_left_h:.2f}h | "
                f"since_improve={since_improvement:.1f}s",
                log_path
            )

        scores_snapshot = scores.copy()

        # Build edges_for_dp among remaining, using snapshot
        edges_for_dp = []
        global_idx_for_local = []
        for gi in remaining_indices:
            u, v, w = candidate_edges[gi]
            ru = scores_snapshot[u]
            rv = scores_snapshot[v]
            if ru > rv:
                lo = rv
                hi = ru
                edges_for_dp.append((u, v, w, lo, hi))
                global_idx_for_local.append(gi)

        if not edges_for_dp:
            log_message(f"[FUNC1] round={round_idx} | no backward edges among remaining candidates; stopping.", log_path)
            break

        # DP selection
        local_selected = select_nonconflicting_edge_indices_dp(edges_for_dp)
        if not local_selected:
            log_message(f"[FUNC1] round={round_idx} | DP selected 0 edges; stopping.", log_path)
            break

        selected_global_indices = [global_idx_for_local[li] for li in local_selected]
        batch_edges = [edges_for_dp[li] for li in local_selected]  # (u,v,w,lo,hi)

        dp_batches_done += 1
        log_message(
            f"[FUNC1] round={round_idx} batch={dp_batches_done} | selected_intervals={len(batch_edges)} "
            f"(remaining_backward_in_snapshot={len(edges_for_dp)})",
            log_path
        )

        # Spawn workers
        num_edges_batch = len(batch_edges)
        num_procs_effective = min(num_procs, num_edges_batch)

        edge_queue = mp.Queue()
        result_queue = mp.Queue()
        for i in range(num_edges_batch):
            edge_queue.put(i)

        dummy_manager = mp.Manager()
        active_intervals = dummy_manager.list()
        used_intervals = dummy_manager.list()
        lock = dummy_manager.Lock()

        workers = []
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
                    log_path,
                )
            )
            p.daemon = False
            p.start()
            workers.append(p)

        alive_workers = num_procs_effective
        total_bw_reduct_round = 0.0
        applied_edges_round = 0

        while alive_workers > 0:
            res = result_queue.get()
            if res.get("done"):
                alive_workers -= 1
                continue

            success = bool(res.get("success"))
            delta = float(res.get("delta", 0.0))
            changed_scores = res.get("changed_scores", {})

            if success and delta > EPS_IMPROVE:
                applied_edges_round += 1
                total_bw_reduct_round += delta
                BW_tracked -= delta
                for node, new_r in changed_scores.items():
                    scores[node] = new_r
                last_improvement_time = time.time()

        for p in workers:
            p.join()

        # Remove selected edges from remaining_indices (assessed once per call)
        selected_global_set = set(selected_global_indices)
        remaining_indices = [idx for idx in remaining_indices if idx not in selected_global_set]

        log_message(
            f"[FUNC1] round={round_idx} batch={dp_batches_done} summary | applied={applied_edges_round}/{num_edges_batch} "
            f"| dBW={-total_bw_reduct_round:.6f} | BW_now={BW_tracked:.2f} best={best_BW:.2f} | remaining={len(remaining_indices)}",
            log_path
        )

        improved_this_round = False
        if BW_tracked + EPS_IMPROVE < best_BW:
            best_BW = BW_tracked
            best_scores = scores.copy()
            improved_this_round = True
            save_scores_to_csv(best_scores, output_excel, index_to_node)
            log_message(f"[FUNC1] NEW_BEST | BW={best_BW:.2f} | saved={output_excel}", log_path)

        # Sanity BW check sometimes
        if BW_CHECK_EVERY_ROUND > 0 and (round_idx % BW_CHECK_EVERY_ROUND == 1):
            fw_recompute = compute_forward_weight(edges, scores)
            bw_recompute = total_weight - float(fw_recompute)
            if abs(bw_recompute - BW_tracked) > BW_MISMATCH_TOL:
                log_message(
                    f"[FUNC1] SANITY_MISMATCH round={round_idx} | BW_recompute={bw_recompute:.6f} BW_tracked={BW_tracked:.6f}",
                    log_path
                )
                raise RuntimeError("Backward-weight tracking mismatch.")

        # No-improvement stop (for this func1 call only)
        if (time.time() - last_improvement_time) >= NO_IMPROVEMENT_TIME_LIMIT_SEC:
            log_message(
                f"[FUNC1] Stop: no improvement for {time.time() - last_improvement_time:.1f}s "
                f"(limit={NO_IMPROVEMENT_TIME_LIMIT_SEC:.0f}s).",
                log_path
            )
            break

    # Final save best_scores for this call
    save_scores_to_csv(best_scores, output_excel, index_to_node)
    final_fw = compute_forward_weight(edges, best_scores)
    final_bw = total_weight - float(final_fw)

    remaining_pairs = set()
    for gi in remaining_indices:
        u, v, _ = candidate_edges[gi]
        remaining_pairs.add((u, v))

    log_message(f"[FUNC1] END | bestBW={final_bw:.2f} | unassessed_pairs={len(remaining_pairs)} | wrote={output_excel}", log_path)
    return best_scores, final_bw, remaining_pairs



def build_arc_ordering_L1_L2(edges, ordering: str, log_path=None):
    """
    edges: list of (u,v,w)
    ordering: "L1" or "L2"
    returns: list of arcs (u,v,w) sorted ascending by the chosen key
    """
    ordering = ordering.upper().strip()
    if ordering not in ("L1", "L2"):
        raise ValueError("ordering must be 'L1' or 'L2'")

    # Precompute W_in(v), W_out(v)
    W_in = {}
    W_out = {}
    for (u, v, w) in edges:
        w = float(w)
        W_out[u] = W_out.get(u, 0.0) + w
        W_in[v] = W_in.get(v, 0.0) + w

    if log_path:
        log_message(f"[WMSF] build_ordering {ordering} | n_edges={len(edges)} | n_nodes_inW={len(W_in)} n_nodes_outW={len(W_out)}", log_path)

    if ordering == "L1":
        return sorted([(u, v, float(w)) for (u, v, w) in edges], key=lambda t: t[2])

    # L2 key
    def key_L2(t):
        u, v, w = t
        denom = W_in.get(u, 0.0) + W_out.get(v, 0.0)
        if denom <= 0:
            denom = 1.0
        return float(w) / denom

    return sorted([(u, v, float(w)) for (u, v, w) in edges], key=key_L2)

# In[11]:


def kahn_toposort(nodes, out_adj):
    """
    nodes: iterable of nodes
    out_adj: dict u -> list of v
    returns (is_acyclic, topo_list)
    """
    nodes = list(nodes)
    indeg = {v: 0 for v in nodes}
    for u in nodes:
        for v in out_adj.get(u, []):
            if v in indeg:
                indeg[v] += 1

    q = [v for v in nodes if indeg[v] == 0]
    topo = []
    head = 0
    while head < len(q):
        x = q[head]
        head += 1
        topo.append(x)
        for y in out_adj.get(x, []):
            if y in indeg:
                indeg[y] -= 1
                if indeg[y] == 0:
                    q.append(y)

    return (len(topo) == len(nodes)), topo


def wmsf_remove_arcs(nodes, edges, ordering="L2", log_path=None, MAX_SEC=None):
    """
    WMSF phase-1: remove arcs until DAG, tracking removed set F.
    ordering: "L1" or "L2" (paper variants).
    MAX_SEC: optional soft time budget; if exceeded, returns current F.
    """
    import time

    start = time.time()
    if ordering not in ("L1", "L2"):
        ordering = "L2"

    # Build adjacency
    out_adj = {u: [] for u in nodes}
    in_adj = {u: [] for u in nodes}
    for (u, v, w) in edges:
        out_adj[u].append((v, float(w)))
        in_adj[v].append((u, float(w)))

    alive = set(nodes)
    F = set()

    def out_weight(u):
        return sum(w for (_, w) in out_adj[u] if _ in alive)

    def in_weight(u):
        return sum(w for (_, w) in in_adj[u] if _ in alive)

    # Precompute Kahn-like elimination with greedy source/sink selection
    removed_vertices = 0
    while alive:
        if MAX_SEC is not None and (time.time() - start) > MAX_SEC:
            if log_path:
                log_message(f"[WMSF] remove_arcs: hit MAX_SEC={MAX_SEC}s | alive={len(alive)} | F={len(F)}", log_path)
            break

        # remove all sinks (out=0)
        progressed = True
        while progressed:
            progressed = False
            sinks = [u for u in list(alive) if all(v not in alive for (v, _) in out_adj[u])]
            if sinks:
                progressed = True
                for u in sinks:
                    alive.remove(u)
                    removed_vertices += 1

        # remove all sources (in=0)
        progressed = True
        while progressed:
            progressed = False
            sources = [u for u in list(alive) if all(v not in alive for (v, _) in in_adj[u])]
            if sources:
                progressed = True
                for u in sources:
                    alive.remove(u)
                    removed_vertices += 1

        if not alive:
            break

        # choose a vertex by L1/L2 rule
        best_u = None
        best_score = None
        for u in alive:
            ow = out_weight(u)
            iw = in_weight(u)
            score = (ow - iw) if ordering == "L1" else (ow - iw) / (ow + iw + 1e-12)
            if best_score is None or score > best_score:
                best_score = score
                best_u = u

        # remove incoming edges to best_u into F (paper’s “remove arcs” step)
        u = best_u
        for (p, w) in in_adj[u]:
            if p in alive:
                F.add((p, u))
        alive.remove(u)
        removed_vertices += 1

    if log_path:
        log_message(f"[WMSF] remove_arcs done | ordering={ordering} | removed_vertices={removed_vertices} | F={len(F)}", log_path)
    return F


def wmsf_minimize_fas(nodes, edges, F, log_path=None, MAX_CHECKS=None, MAX_SEC=None):
    """
    WMSF phase-3: try to reinsert arcs from F back into E if it doesn't create a cycle.
    This is expensive; budgets control worst-case runtime.
    """
    import time
    from collections import defaultdict, deque

    start = time.time()

    out_adj = defaultdict(list)
    in_adj = defaultdict(list)
    for (u, v, w) in edges:
        out_adj[u].append((v, float(w)))
        in_adj[v].append((u, float(w)))

    # Build a working edge-set E' = E \ F
    F = set(F)
    Eprime = set((u, v) for (u, v, _) in edges if (u, v) not in F)

    def is_acyclic(Eset):
        indeg = {u: 0 for u in nodes}
        g = {u: [] for u in nodes}
        for (a, b) in Eset:
            g[a].append(b)
            indeg[b] += 1
        q = deque([u for u in nodes if indeg[u] == 0])
        seen = 0
        while q:
            u = q.popleft()
            seen += 1
            for v in g[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return seen == len(nodes)

    checks = 0
    # Greedy reinsertion: try each removed arc; keep it if DAG remains
    for (u, v) in list(F):
        checks += 1
        if MAX_CHECKS is not None and checks > MAX_CHECKS:
            if log_path:
                log_message(f"[WMSF] minimize_fas: hit MAX_CHECKS={MAX_CHECKS} | keptF={len(F)}", log_path)
            break
        if MAX_SEC is not None and (time.time() - start) > MAX_SEC:
            if log_path:
                log_message(f"[WMSF] minimize_fas: hit MAX_SEC={MAX_SEC}s | keptF={len(F)}", log_path)
            break

        trial = set(Eprime)
        trial.add((u, v))
        if is_acyclic(trial):
            # can reinsert: remove from F, add to E'
            F.remove((u, v))
            Eprime.add((u, v))

        if log_path and (checks % 5000 == 0):
            log_message(f"[WMSF] minimize_fas progress | checks={checks} | keptF={len(F)}", log_path)

    if log_path:
        log_message(f"[WMSF] minimize_fas done | checks={checks} | keptF={len(F)}", log_path)
    return F


def wmsf_stabilize_fas(nodes, edges, F, log_path=None):
    """
    Stabilize according to the paper's weight-stability inequalities.
    We apply their described swap ideas, BUT we verify DAG after each swap and rollback if it breaks.
    """
    import math

    nodes = list(nodes)
    A = set((u, v) for (u, v, _) in edges)

    w = {(u, v): float(ww) for (u, v, ww) in edges}

    # Precompute W_in(v,G), W_out(v,G) on original G
    W_in_G = {v: 0.0 for v in nodes}
    W_out_G = {v: 0.0 for v in nodes}
    for (u, v, ww) in edges:
        W_out_G[u] += float(ww)
        W_in_G[v] += float(ww)

    max_passes = max(1, int(math.log2(max(2, len(nodes)))))

    def build_present_from_F(Fset):
        present = A - set(Fset)
        out_adj = {x: [] for x in nodes}
        for (a, b) in present:
            out_adj[a].append(b)
        return present, out_adj

    changed_any = False
    Fset = set(F)

    for p in range(1, max_passes + 1):
        present, out_adj = build_present_from_F(Fset)
        acyclic, topo = kahn_toposort(nodes, out_adj)
        if not acyclic:
            if log_path:
                log_message(f"[WMSF] stabilizeFAS | PASS{p} | WARNING: input G* not acyclic (should not happen).", log_path)
            break

        # Compute W_in(v,G*), W_out(v,G*) from present
        W_in_Gs = {v: 0.0 for v in nodes}
        W_out_Gs = {v: 0.0 for v in nodes}
        for (a, b) in present:
            ww = w.get((a, b), 0.0)
            W_out_Gs[a] += ww
            W_in_Gs[b] += ww

        local_changes = 0
        rollbacks = 0

        # for quick incident arc lists
        in_arcs = {v: [] for v in nodes}
        out_arcs = {v: [] for v in nodes}
        for (a, b, ww) in edges:
            out_arcs[a].append((a, b))
            in_arcs[b].append((a, b))

        for v in topo:
            # eliminated incoming/outgoing weights
            elim_in = W_in_G[v] - W_in_Gs[v]
            elim_out = W_out_G[v] - W_out_Gs[v]
            rem_in = W_in_Gs[v]
            rem_out = W_out_Gs[v]

            violated1 = elim_in > rem_out
            violated2 = elim_out > rem_in

            if not (violated1 or violated2):
                continue

            # snapshot F to rollback if needed
            F_before = set(Fset)

            if violated1:
                # eliminate remaining outgoing arcs, restore eliminated incoming arcs
                # => add outgoing present arcs to F; remove incoming eliminated arcs from F
                for (a, b) in out_arcs[v]:
                    if (a, b) in A and (a, b) not in Fset:
                        Fset.add((a, b))
                for (a, b) in in_arcs[v]:
                    if (a, b) in Fset:
                        Fset.remove((a, b))
            else:
                # violated2: eliminate remaining incoming arcs, restore eliminated outgoing arcs
                for (a, b) in in_arcs[v]:
                    if (a, b) in A and (a, b) not in Fset:
                        Fset.add((a, b))
                for (a, b) in out_arcs[v]:
                    if (a, b) in Fset:
                        Fset.remove((a, b))

            # verify acyclic, else rollback
            present2, out_adj2 = build_present_from_F(Fset)
            acyclic2, _ = kahn_toposort(nodes, out_adj2)
            if not acyclic2:
                Fset = F_before
                rollbacks += 1
                if log_path:
                    log_message(f"[WMSF] stabilizeFAS | PASS{p} | rollback(v={v}) because cycle introduced.", log_path)
            else:
                local_changes += 1

        if log_path:
            log_message(f"[WMSF] stabilizeFAS | PASS{p}/{max_passes} | changes={local_changes} rollbacks={rollbacks}", log_path)

        if local_changes == 0:
            break

        changed_any = changed_any or (local_changes > 0)

    return Fset



def wmsf_produce_ranking(csv_path, input_ranking_path, output_ranking_path, log_path,
                        ordering="L2", MAX_SEC=None, MAX_NODES=None):
    import time

    t0 = time.time()
    edges, node_to_index, index_to_node = read_graph(csv_path)
    n = len(node_to_index)

    if MAX_NODES is not None and n > int(MAX_NODES):
        log_message(f"[WMSF] SKIP produce_ranking: n={n} > MAX_NODES={MAX_NODES}", log_path)
        return None, None

    scores0 = load_initial_scores(input_ranking_path, node_to_index)

    # budgets split (soft)
    if MAX_SEC is None:
        t_remove = t_min = None
    else:
        MAX_SEC = float(MAX_SEC)
        t_remove = 0.35 * MAX_SEC
        t_min    = 0.60 * MAX_SEC

    nodes = list(range(n))
    log_message(f"[WMSF] start | ordering={ordering} | n={n} | MAX_SEC={MAX_SEC}", log_path)

    # 1) remove arcs -> F
    F = wmsf_remove_arcs(nodes, edges, ordering=ordering, log_path=log_path, MAX_SEC=t_remove)
    log_message(f"[WMSF] remove_arcs done | ordering={ordering} | F={len(F)}", log_path)

    # 2) stabilize FAS -> still F
    F = wmsf_stabilize_fas(nodes, edges, F, log_path=log_path)

    # 3) derive scores from E\F (topo), tie-broken by seed scores0
    scores1 = wmsf_scores_from_E_minus_F(nodes, edges, F, seed_scores=scores0, log_path=log_path)
    if scores1 is None:
        log_message("[WMSF] produce_ranking abort: E\\F cyclic after stabilize.", log_path)
        return None, None

    # 4) minimize FAS
    F2 = wmsf_minimize_fas(nodes, edges, F, log_path=log_path, MAX_SEC=t_min)

    # 5) stabilize again + derive final scores
    F2 = wmsf_stabilize_fas(nodes, edges, F2, log_path=log_path)
    scores2 = wmsf_scores_from_E_minus_F(nodes, edges, F2, seed_scores=scores1, log_path=log_path)
    if scores2 is None:
        log_message("[WMSF] produce_ranking abort: E\\F cyclic after minimize+stabilize.", log_path)
        return None, None

    save_scores_to_csv(scores2, output_ranking_path, index_to_node)

    dt = time.time() - t0
    log_message(f"[WMSF] done | ordering={ordering} | dt={dt:.2f}s | F={len(F2)} | out={output_ranking_path}", log_path)
    return scores2, F2

def _normalize_F_to_pairs(F, log_path=None):
    """
    Ensure F is a set of (u,v) pairs. Accepts (u,v) or (u,v,w).
    Drops malformed items and logs count.
    """
    out = set()
    malformed = 0
    for e in F:
        try:
            if isinstance(e, tuple) and len(e) >= 2:
                out.add((e[0], e[1]))
            else:
                malformed += 1
        except TypeError:
            malformed += 1
    if malformed and log_path:
        log_message(f"[WMSF] normalize_F: dropped malformed={malformed} kept={len(out)}", log_path)
    return out


def wmsf_scores_from_E_minus_F(nodes, edges, F, seed_scores, log_path=None):
    """
    Build a topo order of G* = (V, E\\F) and convert it to scores dict.
    Uses seed_scores only for tie-breaking among zero-indegree nodes (stable ranking).
    """
    nodes = list(nodes)
    Fset = _normalize_F_to_pairs(F, log_path=log_path)

    # adjacency of present edges
    out_adj = {u: [] for u in nodes}
    indeg = {u: 0 for u in nodes}

    for (u, v, _) in edges:
        if (u, v) in Fset:
            continue
        out_adj[u].append(v)
        indeg[v] += 1

    # tie-break: smaller seed rank first
    BIG = 10**18
    def key(u):
        return int(seed_scores.get(u, BIG))

    heap = []
    for u in nodes:
        if indeg[u] == 0:
            heapq.heappush(heap, (key(u), u))

    topo = []
    while heap:
        _, u = heapq.heappop(heap)
        topo.append(u)
        for v in out_adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (key(v), v))

    if len(topo) != len(nodes):
        if log_path:
            log_message(f"[WMSF] ERROR: E\\F is still cyclic (topo={len(topo)}/{len(nodes)}).", log_path)
        return None

    scores = {u: i for i, u in enumerate(topo)}
    return scores



# --- OPTIONAL: WMSF reseed phase (call whenever you want) ---
def maybe_run_wmsf_reseed(csv_path, current_input_path, output_ranking_path, log_path,
                         ordering="L2", MAX_SEC=None, MAX_NODES=None):
    """
    Runs WMSF reseed and returns True if it produced a ranking file.
    All exceptions are logged and return False.
    """
    import os
    try:
        scores, _ = wmsf_produce_ranking(
            csv_path=csv_path,
            input_ranking_path=current_input_path,
            output_ranking_path=output_ranking_path,
            log_path=log_path,
            ordering=ordering,
            MAX_SEC=MAX_SEC,
            MAX_NODES=MAX_NODES,
        )
        if scores is None:
            return False
        if not os.path.exists(output_ranking_path):
            log_message(f"[WMSF] reseed failed: output not created: {output_ranking_path}", log_path)
            return False
        return True
    except Exception as e:
        log_message(f"[WMSF] reseed exception: {e}", log_path)
        return False



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
    
    # 1. Compute Baseline
    # Optimization: Ensure _compute_block_fw only scans edges relevant to the block
    fw_block_before = _compute_block_fw(G, scores_before, block_nodes)

    if debug:
        start_ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{worker_id} | {start_ts}] 🧵 Start. Nodes={len(block_nodes)}, FW_Pre={fw_block_before:.2f}")

    # 2. Run Refinement (Solver)
    # CRITICAL FIX: Pass the 'debug' flag down so inner logic is visible if requested
    new_scores, _, _, _ = refine_block_scc_interval(
    G=G,
    edges=edges,
    node_order=scores_before,
    block_nodes=block_nodes,
    brute_force_min_size=brute_force_min_size,
    brute_force_max_size=brute_force_max_size,
    debug=debug,
    dp_exact_max_size=18,
    large_scc_ls_iters=200,
    eps_improve=1e-12,
    log_path=log_path,  # <-- IMPORTANT
)

    # 3. Compute Result
    fw_block_after = _compute_block_fw(G, new_scores, block_nodes)
    
    # CRITICAL FIX: Calculate Gain Explicitly
    # This allows the Controller to see if we improved (positive) or regressed (negative)
    local_gain = fw_block_after - fw_block_before

    # CRITICAL FIX: REMOVED THE CRASH ON NEGATIVE GAIN.
    # We now allow the solver to return a worse score (exploration/annealing).
    # We only log a warning if it's a massive drop during debugging.
    if local_gain < -1e-9 and debug:
        print(f"[{worker_id}] ⚠️ Negative gain ({local_gain:.4f}). Exploration or regression?")

    block_ranks = {n: new_scores[n] for n in block_nodes}

    # --- End timestamp ---
    elapsed = time.time() - t_start

    if debug and abs(local_gain) > 1e-9:
        print(
            f"[{worker_id}] "
            f"✅ Done. ΔFW={local_gain:+.4f}, "
            f"Elapsed={elapsed:.2f}s"
        )

    # 4. Return Dictionary matching the new Controller's expectations
    return {
        "worker_id": worker_id,
        "pid": pid,
        "block_nodes": block_nodes,
        "block_ranks": block_ranks,
        "fw_block_before": fw_block_before,
        "fw_block_after": fw_block_after,
        "local_gain": local_gain,  # <--- REQUIRED by the new Controller
        "elapsed_worker": elapsed,
    }


def _scc_net_out_in_score(subG, nodes):
    # score(u) = sum_w(u->*) - sum_w(*->u) within subG
    score = {u: 0.0 for u in nodes}
    for u in nodes:
        for v in subG.successors(u):
            if v in score:
                score[u] += subG[u][v].get("weight", 1.0)
        for v in subG.predecessors(u):
            if v in score:
                score[u] -= subG[v][u].get("weight", 1.0)
    return score


def _internal_fw_of_order(subG, order):
    pos = {u: i for i, u in enumerate(order)}
    fw = 0.0
    for u, v, data in subG.edges(data=True):
        if u in pos and v in pos and pos[u] < pos[v]:
            fw += data.get("weight", 1.0)
    return fw


def _local_search_scc(subG, order, iters=300, rng=None):
    """
    Hill-climb on internal FW using cheap insert/swap moves.
    Works even for big SCCs. Keeps only improvements (monotone).
    """
    import random
    if rng is None:
        rng = random.Random()

    best = list(order)
    best_fw = _internal_fw_of_order(subG, best)
    n = len(best)

    if n < 3 or iters <= 0:
        return best, best_fw

    for _ in range(iters):
        if n <= 2:
            break

        move_type = rng.random()
        if move_type < 0.65:
            # INSERT move: remove i and insert at j
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            cand = best[:]
            x = cand.pop(i)
            cand.insert(j, x)
        else:
            # SWAP move
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            cand = best[:]
            cand[i], cand[j] = cand[j], cand[i]

        fw = _internal_fw_of_order(subG, cand)
        if fw > best_fw + 1e-12:
            best, best_fw = cand, fw

    return best, best_fw


def _max_fw_order_dp(subG, nodes):
    """
    Exact max forward-weight linear ordering on 'nodes' using DP over subsets.
    Complexity: O(k^2 * 2^k). Practical up to ~18 (maybe 20 if sparse).
    Returns: best order (list), best internal FW.
    """
    k = len(nodes)
    idx = {nodes[i]: i for i in range(k)}

    # weight[i][j] = weight of edge nodes[i] -> nodes[j] within subG
    w = [[0.0]*k for _ in range(k)]
    for u, v, data in subG.edges(data=True):
        if u in idx and v in idx:
            w[idx[u]][idx[v]] += data.get("weight", 1.0)

    # dp[mask] = best FW for an ordering of subset mask
    # parent[mask] = (prev_mask, last_added_index)
    dp = {0: 0.0}
    parent = {0: (-1, -1)}

    for mask in range(1 << k):
        if mask not in dp:
            continue
        cur = dp[mask]
        # try adding a new last node j not in mask
        for j in range(k):
            if mask & (1 << j):
                continue
            nmask = mask | (1 << j)
            # when j is last, it contributes edges from existing nodes i -> j
            add = 0.0
            for i in range(k):
                if mask & (1 << i):
                    add += w[i][j]
            cand = cur + add
            if nmask not in dp or cand > dp[nmask] + 1e-12:
                dp[nmask] = cand
                parent[nmask] = (mask, j)

    full = (1 << k) - 1
    best_fw = dp.get(full, 0.0)

    # reconstruct order
    order_idx = []
    mask = full
    while mask:
        pmask, j = parent[mask]
        order_idx.append(j)
        mask = pmask
    order_idx.reverse()
    order = [nodes[j] for j in order_idx]
    return order, best_fw


def refine_block_scc_interval(
    G,
    edges,
    node_order,
    block_nodes,
    brute_force_min_size=2,
    brute_force_max_size=7,
    debug=False,
    # ---------------- NEW knobs (safe defaults) ----------------
    dp_exact_max_size=18,          # use subset-DP exact solver for SCC sizes up to this (if available)
    large_scc_ls_iters=200,        # local-search budget for large SCCs
    eps_improve=1e-12,             # numerical tolerance for “improved”
    # ---------------- NEW logging knobs ----------------
    log_path=None,                # REQUIRED if debug=True (so we can log)
    log_solvers_always=False,      # if True, print solver summary even when debug=False
    log_only_when_no_gain=True,    # if True, solver summary prints only when no global improvement
):
    """
    Refine a given interval (block) of nodes using SCC decomposition + topo sort.

    For SCCs:
      - size == 1: keep stable order
      - brute_force_min_size..brute_force_max_size: brute force permutations (exact)
      - (brute_force_max_size+1)..dp_exact_max_size: exact subset-DP (if _max_fw_order_dp exists)
      - larger: heuristic net-out-in + local search (if helpers exist), otherwise stable order

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

    # --- small helper to enforce "ONLY log_message" ---
    def _log(msg):
        # We assume your log_message(msg, log_path) exists globally.
        # If log_path is None, we still call log_message to keep behavior consistent with your rule.
        log_message(msg, log_path)

    block_nodes = list(block_nodes)

    if debug:
        _log(f"[FUNC2][refine_block] start | block_size={len(block_nodes)}")

    # Sort block nodes by current rank for stability
    block_nodes = sorted(block_nodes, key=lambda n: node_order[n])

    # Copy ranking
    prev_order = node_order.copy()
    new_node_order = node_order.copy()

    # Subgraph induced by the block
    subG = G.subgraph(block_nodes).copy()
    if debug:
        _log(f"[FUNC2][refine_block] subgraph | V={subG.number_of_nodes()} E={subG.number_of_edges()}")

    # SCC decomposition inside block
    sub_sccs = list(nx.strongly_connected_components(subG))
    if debug:
        _log(f"[FUNC2][refine_block] SCCs_in_block={len(sub_sccs)}")

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
    if debug:
        _log(f"[FUNC2][refine_block] topo_SCC_order_len={len(scc_order)}")

    # ---- NEW: solver usage counters (high-signal, low-volume) ----
    brute_cnt = 0
    dp_cnt = 0
    heur_ls_cnt = 0
    heur_only_cnt = 0
    fallback_cnt = 0
    single_cnt = 0
    max_scc_size_seen = 0
    dp_failed_cnt = 0
    ls_missing_cnt = 0
    helpers_missing_cnt = 0

    # Construct new order for the block
    new_block_order = []
    for scc_id in scc_order:
        scc_nodes = list(sub_sccs[scc_id])
        scc_size = len(scc_nodes)
        if scc_size > max_scc_size_seen:
            max_scc_size_seen = scc_size

        if scc_size == 1:
            single_cnt += 1
            new_block_order.extend(scc_nodes)

        elif brute_force_min_size <= scc_size <= brute_force_max_size:
            brute_cnt += 1
            if debug:
                _log(f"[FUNC2][block_scc] id={scc_id} size={scc_size} solver=bruteforce")
            best_perm = None
            max_weight = float("-inf")

            for perm in permutations(scc_nodes):
                w_sum = 0.0
                for i, u in enumerate(perm):
                    for j in range(i + 1, scc_size):
                        v = perm[j]
                        if subG.has_edge(u, v):
                            w_sum += subG[u][v].get("weight", 1.0)
                if w_sum > max_weight + eps_improve:
                    max_weight = w_sum
                    best_perm = perm

            if best_perm is None:
                # fallback (shouldn't happen)
                best_perm = sorted(scc_nodes, key=lambda n: prev_order[n])
                max_weight = 0.0

            if debug:
                _log(f"[FUNC2][block_scc] id={scc_id} size={scc_size} bruteforce_best_internalFW={max_weight:.2f}")
            new_block_order.extend(best_perm)

        else:
            # Mid SCC: try exact subset-DP if available and enabled
            used_solver = False
            if dp_exact_max_size is not None and scc_size <= dp_exact_max_size:
                try:
                    best_order, best_fw = _max_fw_order_dp(subG, scc_nodes)
                    dp_cnt += 1
                    used_solver = True
                    if debug:
                        _log(f"[FUNC2][block_scc] id={scc_id} size={scc_size} solver=dp_exact internalFW={best_fw:.2f}")
                    new_block_order.extend(best_order)
                except NameError:
                    dp_failed_cnt += 1
                    used_solver = False
                except Exception as e:
                    dp_failed_cnt += 1
                    used_solver = False
                    if debug:
                        _log(f"[FUNC2][block_scc] id={scc_id} size={scc_size} dp_exact_failed err={type(e).__name__}:{e}")

            if not used_solver:
                # Large SCC: heuristic net-out-in + local search if helpers exist, else stable order
                try:
                    net = _scc_net_out_in_score(subG, scc_nodes)
                    base = sorted(
                        scc_nodes,
                        key=lambda n: (-net.get(n, 0.0), prev_order[n])
                    )

                    # Try local search if available
                    try:
                        base2, base_fw = _local_search_scc(subG, base, iters=int(large_scc_ls_iters))
                        heur_ls_cnt += 1
                        if debug:
                            _log(f"[FUNC2][block_scc] id={scc_id} size={scc_size} solver=heuristic+LS internalFW={base_fw:.2f} iters={int(large_scc_ls_iters)}")
                        new_block_order.extend(base2)
                    except NameError:
                        ls_missing_cnt += 1
                        heur_only_cnt += 1
                        if debug:
                            _log(f"[FUNC2][block_scc] id={scc_id} size={scc_size} solver=heuristic_only ls_helper_missing")
                        new_block_order.extend(base)

                except NameError:
                    helpers_missing_cnt += 1
                    fallback_cnt += 1
                    if debug:
                        _log(f"[FUNC2][block_scc] id={scc_id} size={scc_size} solver=fallback_original helpers_missing")
                    sorted_original = sorted(scc_nodes, key=lambda n: prev_order[n])
                    new_block_order.extend(sorted_original)

    # Sanity: permutation
    if set(new_block_order) != set(block_nodes) or len(new_block_order) != len(block_nodes):
        if debug:
            missing = set(block_nodes) - set(new_block_order)
            extra = set(new_block_order) - set(block_nodes)
            _log(f"[FUNC2][refine_block] ERROR mismatch new_block_order | missing={len(missing)} extra={len(extra)}")
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
            _log(f"[FUNC2][refine_block] ERROR duplicate ranks | num_dup_ranks={len(duplicates)}")
        raise ValueError("Duplicate ranks detected in new_node_order!")

    # Compute global FW before/after
    existing_fw = compute_forward_weight(edges, prev_order)
    new_fw = compute_forward_weight(edges, new_node_order)
    improved = new_fw > existing_fw + eps_improve
    delta = new_fw - existing_fw

    # ---- NEW: single summary line about solver usage ----
    # We print it:
    #   - if debug, OR
    #   - if log_solvers_always, OR
    #   - if (not improved) and log_only_when_no_gain
    should_log_solver_summary = (
        debug
        or log_solvers_always
        or (log_only_when_no_gain and (not improved))
    )

    if should_log_solver_summary:
        _log(
            "[FUNC2][block_solver_summary] "
            f"block={len(block_nodes)} sccs={len(sub_sccs)} max_scc={max_scc_size_seen} "
            f"single={single_cnt} brute={brute_cnt} dp={dp_cnt} heurLS={heur_ls_cnt} "
            f"heurOnly={heur_only_cnt} fallback={fallback_cnt} "
            f"dpFailed={dp_failed_cnt} lsMissing={ls_missing_cnt} helpersMissing={helpers_missing_cnt} "
            f"ΔFW={delta:.2f} improved={int(improved)}"
        )

    # Existing final debug line (but using log_message only)
    if debug:
        _log(f"[FUNC2][refine_block] FW_before={existing_fw:.2f} FW_after={new_fw:.2f} ΔFW={delta:.2f} improved={int(improved)}")

    return new_node_order, existing_fw, new_fw, improved


def parallel_refine_largest_scc_intervals(
    block_size=530,                 # kept for API compatibility
    brute_force_min_size=2,
    brute_force_max_size=7,
    max_backward_flips=100,
    verify_every=None,              # AUTO
    csv_path="/content/drive/MyDrive/connectome_graph.csv",
    initial_ranking_path="/content/drive/MyDrive/bader.csv",
    output_path=None,
    save_every=None,                # AUTO (best-only checkpoint cadence)
    debug=False,                    # if True -> a bit more logs, still "important only"
    max_no_improvement_batches=30,  # patience on non-improving *accepted* batches
    max_empty_batch_builds=15,      # patience on failing to construct a batch
    log_path=None,                 # REQUIRED
    # ---------- adaptive sizing knobs ----------
    block_frac_min=0.01,            # min fraction of SCC size
    block_frac_max=0.08,            # max fraction of SCC size
    blocks_per_core_target=2.0,     # try to generate ~num_procs*blocks_per_core_target intervals, pick non-overlapping
    max_block_abs=5000,             # absolute cap on block size
    min_block_abs=50,               # absolute floor
    # ---------- meaningful "don't retry bad regions" ----------
    cooldown_batches=25,            # avoid "cold" intervals for this many batches
    cold_if_delta_fw_le=-1e-9,      # only strictly negative ΔFW marks interval cold
    good_interval_factor=2.0,       # good interval memory size ~ factor*num_buckets
    exploit_prob=0.45,
    ucb_prob=0.35,
    random_prob=0.20,
    eps_improve_fw=1e-9,            # improvement threshold
    best_save_eps_bw=1e-9,          # only save when best improves by this
):
    """
    Fused "best of both" func2:

    ✅ Considers ALL nontrivial SCCs (not only largest)
    ✅ Adaptive block sizing (fraction of SCC size + abs min/max)
    ✅ Cached SCC sorting + O(k) slicing (no O(n) scans per attempt)
    ✅ Stops cleanly (max_backward_flips, patience on no-improve, patience on empty-batch-build)
    ✅ Meaningful "don't retry bad regions" using cooldown of cold intervals (strictly negative ΔFW)
    ✅ Uses log_message(log_path) and logs only important events
    ✅ Saves BEST ranking only, with Node ID / Order column names (loader-compatible)
    """
    import time, os, math, random
    import multiprocessing as mp
    import networkx as nx
    import pandas as pd
    from bisect import bisect_left, bisect_right
    from collections import deque

    if log_path is None:
        raise TypeError("parallel_refine_largest_scc_intervals: log_path is required.")

    # ----------------- logging helpers (important only) -----------------
    def _imp(msg: str):
        log_message(f"[func2] {msg}", log_path)

    def _dbg(msg: str):
        if debug:
            log_message(f"[func2] {msg}", log_path)

    # ----------------- parameter guards -----------------
    if exploit_prob + ucb_prob + random_prob <= 0:
        raise ValueError("exploit_prob + ucb_prob + random_prob must be > 0.")
    s = exploit_prob + ucb_prob + random_prob
    exploit_p = exploit_prob / s
    ucb_p = ucb_prob / s
    # random_p = random_prob / s  # implicit

    if not (0.0 < float(block_frac_min) <= float(block_frac_max) <= 1.0):
        raise ValueError(f"block_frac_min/max invalid: {block_frac_min}, {block_frac_max}")
    if min_block_abs <= 0 or max_block_abs is not None and max_block_abs < min_block_abs:
        raise ValueError(f"min_block_abs/max_block_abs invalid: {min_block_abs}, {max_block_abs}")

    # ----------------- processors -----------------
    try:
        num_procs = len(os.sched_getaffinity(0))
    except Exception:
        num_procs = os.cpu_count() or 1
    num_procs = max(1, int(num_procs))

    # ----------------- read graph + scores -----------------
    edges_indexed, node_to_index, index_to_node = read_graph(csv_path)
    scores = load_initial_scores(initial_ranking_path, node_to_index)

    n_nodes = len(scores)
    m_edges = len(edges_indexed)
    total_weight = sum(float(w) for (_, _, w) in edges_indexed)

    # -------- cadence autos --------
    if verify_every is None or verify_every <= 0:
        verify_every = int(200 * max(1.0, math.sqrt(m_edges / 2e5)))
        verify_every = max(100, min(5000, verify_every))

    if save_every is None or save_every <= 0:
        save_every = int(3 * max(1.0, math.sqrt(n_nodes / 2e4)))
        save_every = max(1, min(50, save_every))

    if output_path is None:
        output_path = initial_ranking_path.replace(".csv", "_scc_opt.csv")

    # ----------------- small helpers -----------------
    def all_scores_unique(d):
        vals = list(d.values())
        return len(vals) == len(set(vals))

    def _reindex_to_unique(scores_dict):
        """Force ranks to be 0..n-1 preserving order."""
        sorted_nodes = sorted(scores_dict.keys(), key=lambda k: scores_dict[k])
        for i, n in enumerate(sorted_nodes):
            scores_dict[n] = float(i)

    def _interval_overlaps(a_low, a_high, b_low, b_high):
        return not (a_high < b_low or b_high < a_low)

    def _save_best(scores_dict, path):
        rows = [(index_to_node[i], scores_dict[i]) for i in scores_dict]
        pd.DataFrame(sorted(rows, key=lambda x: x[1]), columns=["Node ID", "Order"]).to_csv(path, index=False)

    def compute_f2b_b2f_and_delta_fw(edges, sbef, saft):
        sum_f2b = 0.0
        sum_b2f = 0.0
        f2b_edges = set()
        for (u, v, w) in edges:
            bu, bv = sbef[u], sbef[v]
            au, av = saft[u], saft[v]
            before_fwd = (bu < bv)
            before_bwd = (bu > bv)
            after_fwd = (au < av)
            after_bwd = (au > av)
            if before_fwd and after_bwd:
                sum_f2b += float(w)
                f2b_edges.add((u, v))
            elif before_bwd and after_fwd:
                sum_b2f += float(w)
        delta_fw = sum_b2f - sum_f2b
        return sum_f2b, sum_b2f, delta_fw, f2b_edges

    def _rank_interval_has_backward_edge(edges_in_scc_local, scores_loc, r_low, r_high):
        for (u, v, _w) in edges_in_scc_local:
            ru = scores_loc[u]
            rv = scores_loc[v]
            if r_low <= ru <= r_high and r_low <= rv <= r_high and ru > rv:
                return True
        return False

    # ----------------- ensure uniqueness (important for slicing and disjoint rank-intervals) -----------------
    if not all_scores_unique(scores):
        _imp("⚠️ Initial scores not unique. Re-indexing to force uniqueness.")
        _reindex_to_unique(scores)

    # ----------------- build graph -----------------
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges_indexed)

    _imp(f"Start: refine ALL SCCs | procs={num_procs} | max_backward_flips={max_backward_flips} | verify_every={verify_every} | save_every={save_every}")

    # ----------------- global SCC topo reorder + reindex again -----------------
    scores = global_scc_topo_reorder_scores(G, edges_indexed, scores, log_path=log_path, debug=debug)
    if not all_scores_unique(scores):
        _dbg("Re-indexing after SCC topo reorder (uniqueness).")
        _reindex_to_unique(scores)

    # ----------------- SCC decomposition -----------------
    sccs = list(nx.strongly_connected_components(G))
    scc_list = [set(s) for s in sccs if len(s) >= 2]
    scc_list.sort(key=len, reverse=True)

    if not scc_list:
        _imp("Done: no nontrivial SCCs.")
        _save_best(scores, output_path)
        return scores, [float(total_weight - compute_forward_weight(edges_indexed, scores))], set(), output_path

    _imp(f"SCCs: total={len(sccs)}, nontrivial={len(scc_list)}, largest={len(scc_list[0])}")

    # ----------------- map nodes to SCC id + internal edges -----------------
    node_to_sccid = {}
    for sid, nodeset in enumerate(scc_list):
        for n in nodeset:
            node_to_sccid[n] = sid

    edges_in_scc = [[] for _ in range(len(scc_list))]
    for (u, v, w) in edges_indexed:
        su = node_to_sccid.get(u)
        if su is not None and su == node_to_sccid.get(v):
            edges_in_scc[su].append((u, v, w))

    # ----------------- per-SCC state (bandit + cooldown) -----------------
    # We'll cache nodes_sorted/ranks_sorted/bucket_to_indices per SCC per batch.
    def _init_scc_state(sid, nodeset):
        ranks = [scores[n] for n in nodeset]
        mn, mx = min(ranks), max(ranks)
        span = int(mx - mn + 1)

        # buckets ~ max(8, 4*num_procs, 2*sqrt(span)) but not more than span
        target_buckets = int(max(8, 4 * num_procs, 2 * math.sqrt(max(1, span))))
        target_buckets = min(max(1, span), target_buckets)
        bucket_size = max(1, span // target_buckets)
        num_buckets = (span + bucket_size - 1) // bucket_size

        return {
            "active": True,
            "size": len(nodeset),
            "min_rank": float(mn),
            "max_rank": float(mx),
            "bucket_size": int(bucket_size),
            "num_buckets": int(num_buckets),
            "bucket_fw_gain": [0.0] * int(num_buckets),
            "bucket_counts":  [0] * int(num_buckets),
            "total_blocks_sampled": 0,
            "good_intervals": deque(maxlen=max(10, int(good_interval_factor * num_buckets))),
            "cold_intervals": deque(maxlen=500),   # (r_low, r_high, batch_idx)
            "no_improve_batches": 0,
        }

    scc_state = {sid: _init_scc_state(sid, scc_list[sid]) for sid in range(len(scc_list))}

    # ----------------- cache helper per batch -----------------
    def _get_scc_cache(sid, snapshot_scores, cache):
        if sid in cache:
            return cache[sid]
        nodes_sorted = sorted(list(scc_list[sid]), key=lambda n: snapshot_scores[n])
        ranks_sorted = [snapshot_scores[n] for n in nodes_sorted]
        st = scc_state[sid]

        nb = st["num_buckets"]
        bs = st["bucket_size"]
        mn = st["min_rank"]
        bucket_to_indices = [[] for _ in range(nb)]
        for idx, r in enumerate(ranks_sorted):
            b = int((r - mn) // bs)
            if 0 <= b < nb:
                bucket_to_indices[b].append(idx)

        cache[sid] = {
            "nodes_sorted": nodes_sorted,
            "ranks_sorted": ranks_sorted,
            "bucket_to_indices": bucket_to_indices,
        }
        return cache[sid]

    # ----------------- UCB bucket pick -----------------
    def _pick_bucket_index(st, bucket_to_indices):
        st["total_blocks_sampled"] += 1
        t = st["total_blocks_sampled"]

        non_empty = [b for b, idxs in enumerate(bucket_to_indices) if idxs]
        if not non_empty:
            return None

        if all(st["bucket_counts"][b] == 0 for b in non_empty):
            return random.choice(non_empty)

        explore_c = 1.0
        best_b = None
        best_score = float("-inf")
        for b in non_empty:
            n_b = st["bucket_counts"][b]
            if n_b == 0:
                mean = 0.0
                bonus = math.sqrt(2.0 * math.log(t + 1.0))
            else:
                mean = st["bucket_fw_gain"][b] / n_b
                bonus = explore_c * math.sqrt(math.log(t + 1.0) / n_b)
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_b = b
        return best_b

    # ----------------- cooldown check -----------------
    def _in_cooldown(st, r_low, r_high, current_batch):
        # If it overlaps any cold interval whose age <= cooldown_batches, skip.
        for (cl, ch, bidx) in st["cold_intervals"]:
            if current_batch - bidx <= cooldown_batches:
                if _interval_overlaps(r_low, r_high, cl, ch):
                    return True
        return False

    # ----------------- init objective tracking -----------------
    current_scores = scores.copy()
    fw_current = float(compute_forward_weight(edges_indexed, current_scores))
    bw_current = float(total_weight - fw_current)
    bw_history = [bw_current]

    best_bw_global = bw_current
    best_scores_global = current_scores.copy()
    last_saved_best_bw = best_bw_global

    f2b_edges_global = set()
    batch_idx = 0
    no_improve_batches_global = 0
    empty_build_streak = 0

    _imp(f"Initial BW={bw_current:.2f}")

    # ----------------- MAIN LOOP -----------------
    while True:
        if len(f2b_edges_global) >= max_backward_flips:
            _imp(f"Stop: reached max_backward_flips={max_backward_flips}.")
            break

        active_sids = [sid for sid, st in scc_state.items() if st["active"]]
        if not active_sids:
            _imp("Stop: all SCCs inactive.")
            break

        if max_no_improvement_batches is not None and no_improve_batches_global >= max_no_improvement_batches:
            _imp(f"Stop: no FW improvement in {no_improve_batches_global} consecutive accepted batches (patience={max_no_improvement_batches}).")
            break

        if max_empty_batch_builds is not None and empty_build_streak >= max_empty_batch_builds:
            _imp(f"Stop: failed to build any valid batch for {empty_build_streak} consecutive attempts (patience={max_empty_batch_builds}).")
            break

        # Fresh per-batch cache built from snapshot
        snapshot_scores = current_scores
        cache = {}

        # We try to build up to K blocks (non-overlapping rank ranges globally)
        # K aims for ~num_procs*blocks_per_core_target, but we cap at num_procs to keep merging simple.
        target_blocks = int(max(1, min(num_procs, round(num_procs * blocks_per_core_target / 2.0))))
        # (Explanation: blocks_per_core_target is aspirational; in practice we keep <=num_procs.)

        batch_blocks = []
        used_rank_ranges = []

        # Weighted SCC sampling by size (more focus on big SCCs)
        weights = [scc_state[sid]["size"] for sid in active_sids]

        max_attempts = num_procs * 80
        attempts = 0

        while len(batch_blocks) < target_blocks and attempts < max_attempts:
            attempts += 1
            sid = random.choices(active_sids, weights=weights, k=1)[0]
            st = scc_state[sid]

            # If SCC has been cold too many times, deactivate (soft)
            if st["no_improve_batches"] >= max(5, max_no_improvement_batches or 30):
                st["active"] = False
                continue

            cached = _get_scc_cache(sid, snapshot_scores, cache)
            nodes_sorted = cached["nodes_sorted"]
            ranks_sorted = cached["ranks_sorted"]
            bucket_to_indices = cached["bucket_to_indices"]
            n_scc = st["size"]
            if n_scc <= 1:
                st["active"] = False
                continue

            # -------- choose block size adaptively --------
            frac = random.uniform(block_frac_min, block_frac_max)
            size_scc = int(round(frac * n_scc))
            size_scc = max(min_block_abs, size_scc)
            if max_block_abs is not None:
                size_scc = min(max_block_abs, size_scc)
            size_scc = min(size_scc, n_scc)

            if size_scc < 2:
                continue

            # -------- choose interval strategy --------
            r = random.random()
            candidate = None  # (r_low, r_high, start_idx, end_idx)

            # (1) Exploit good intervals in this SCC
            if st["good_intervals"] and r < exploit_p:
                base_r_low, base_r_high, _gain, _bidx = random.choice(st["good_intervals"])
                width = max(1.0, base_r_high - base_r_low)

                new_width = max(1.0, width * random.uniform(0.7, 1.6))
                r_center = (base_r_low + base_r_high) / 2.0 + random.uniform(-0.25, 0.25) * width

                r_low = float(max(st["min_rank"], r_center - new_width / 2.0))
                r_high = float(min(st["max_rank"], r_center + new_width / 2.0))
                if r_high > r_low:
                    start_idx = bisect_left(ranks_sorted, r_low)
                    end_idx = bisect_right(ranks_sorted, r_high)
                    if end_idx - start_idx >= 2:
                        candidate = (r_low, r_high, start_idx, end_idx)

            # (2) UCB bucket
            elif r < exploit_p + ucb_p:
                b = _pick_bucket_index(st, bucket_to_indices)
                if b is None:
                    candidate = None
                else:
                    idxs = bucket_to_indices[b]
                    if idxs:
                        center_idx = random.choice(idxs)
                        start_idx = max(0, center_idx - size_scc // 2)
                        end_idx = min(n_scc, start_idx + size_scc)
                        start_idx = max(0, end_idx - size_scc)
                        if end_idx - start_idx >= 2:
                            candidate = (ranks_sorted[start_idx], ranks_sorted[end_idx - 1], start_idx, end_idx)

            # (3) random
            else:
                start_idx = random.randint(0, max(0, n_scc - size_scc))
                end_idx = start_idx + size_scc
                if end_idx - start_idx >= 2:
                    candidate = (ranks_sorted[start_idx], ranks_sorted[end_idx - 1], start_idx, end_idx)

            if not candidate:
                continue

            r_low, r_high, start_idx, end_idx = candidate

            # Flat-rank protection (should not happen if uniqueness is enforced, but safe)
            if not (r_high > r_low):
                continue

            # Cooldown skip (meaningful "don't retry bad regions")
            if _in_cooldown(st, r_low, r_high, batch_idx + 1):
                continue

            # Global overlap skip (preserves cross-boundary direction)
            if any(_interval_overlaps(r_low, r_high, rl, rh) for (rl, rh) in used_rank_ranges):
                continue

            # Require there is at least one backward edge inside the interval (avoid useless work)
            if not _rank_interval_has_backward_edge(edges_in_scc[sid], snapshot_scores, r_low, r_high):
                continue

            block_nodes = nodes_sorted[start_idx:end_idx]
            if len(block_nodes) < 2:
                continue

            batch_blocks.append({"sid": sid, "r_low": r_low, "r_high": r_high, "block_nodes": block_nodes})
            used_rank_ranges.append((r_low, r_high))

        if not batch_blocks:
            empty_build_streak += 1
            continue
        else:
            empty_build_streak = 0

        # -------------- EXECUTE --------------
        batch_idx += 1
        scores_before_batch = current_scores.copy()

        worker_args = []
        for i, blk in enumerate(batch_blocks):
            worker_args.append((f"b{batch_idx}_{i}", blk["block_nodes"], brute_force_min_size, brute_force_max_size, False))

        # Use pool size = number of blocks
        with mp.Pool(
            processes=min(num_procs, len(batch_blocks)),
            initializer=_init_refine_worker,
            initargs=(G, edges_indexed, scores_before_batch),
        ) as pool:
            results = pool.map(_worker_refine_block, worker_args)

        # -------------- MERGE --------------
        for res in results:
            for n, rnk in res["block_ranks"].items():
                current_scores[n] = rnk

        # Safety: ensure uniqueness (should hold; if not, repair and keep going)
        if not all_scores_unique(current_scores):
            _imp(f"⚠️ Duplicate scores detected after batch {batch_idx}. Re-indexing to restore uniqueness.")
            _reindex_to_unique(current_scores)

        # -------------- GLOBAL CREDIT --------------
        sum_f2b, sum_b2f, delta_fw, f2b_batch = compute_f2b_b2f_and_delta_fw(edges_indexed, scores_before_batch, current_scores)
        fw_current = fw_current + float(delta_fw)
        bw_current = float(total_weight - fw_current)
        bw_history.append(bw_current)
        f2b_edges_global.update(f2b_batch)

        # -------------- patience counters --------------
        if delta_fw > eps_improve_fw:
            no_improve_batches_global = 0
        else:
            no_improve_batches_global += 1

        # per-SCC credit assignment (bucket stats + good/cold intervals)
        # We split delta_fw proportionally across SCCs touched in this batch (simple but effective)
        touched = {}
        for blk in batch_blocks:
            touched.setdefault(blk["sid"], []).append(blk)

        for sid, blks in touched.items():
            st = scc_state[sid]

            # If SCC had positive global delta, reset its no_improve; else increment (soft)
            if delta_fw > eps_improve_fw:
                st["no_improve_batches"] = 0
            else:
                st["no_improve_batches"] += 1

            # reward and bucket update
            for blk in blks:
                mn = st["min_rank"]
                bs = st["bucket_size"]
                b_start = int(max(0, (blk["r_low"] - mn) // bs))
                b_end = int(min(st["num_buckets"] - 1, (blk["r_high"] - mn) // bs))
                affected = list(range(b_start, b_end + 1))
                if affected:
                    rew = float(delta_fw) / len(affected)
                    for b in affected:
                        st["bucket_fw_gain"][b] += rew
                        st["bucket_counts"][b] += 1

                # good interval memory
                if delta_fw > eps_improve_fw:
                    st["good_intervals"].append((blk["r_low"], blk["r_high"], float(delta_fw), batch_idx))

                # cold interval cooldown (strictly negative batches)
                if delta_fw <= cold_if_delta_fw_le:
                    st["cold_intervals"].append((blk["r_low"], blk["r_high"], batch_idx))

        # -------------- update best --------------
        improved_best = False
        if bw_current < best_bw_global - best_save_eps_bw:
            best_bw_global = bw_current
            best_scores_global = current_scores.copy()
            improved_best = True

        # -------------- important logging (sparse) --------------
        # Log every 10 batches OR on best improvement OR on big improvements
        if improved_best or (batch_idx % 10 == 0) or (delta_fw > 1e-3):
            _imp(
                f"Batch {batch_idx}: BW={bw_current:.2f} (ΔFW={delta_fw:+.2f}, F2B_wt={sum_f2b:.2f}, B2F_wt={sum_b2f:.2f}) | "
                f"BestBW={best_bw_global:.2f} | blocks={len(batch_blocks)} | noImp={no_improve_batches_global}/{max_no_improvement_batches}"
            )

        # -------------- save best only (rate limited by save_every) --------------
        if save_every and (batch_idx % save_every == 0) and (best_bw_global < last_saved_best_bw - best_save_eps_bw):
            _save_best(best_scores_global, output_path)
            last_saved_best_bw = best_bw_global
            _imp(f"Checkpoint: saved BEST ranking (BestBW={best_bw_global:.2f}) -> {output_path}")

        # -------------- verification (rare) --------------
        if verify_every and (batch_idx % verify_every == 0):
            fw_manual = float(compute_forward_weight(edges_indexed, current_scores))
            bw_manual = float(total_weight - fw_manual)
            if abs(bw_manual - bw_current) > 1e-6:
                _imp(f"❌ Verification failed at batch {batch_idx}: tracked_BW={bw_current:.6f}, manual_BW={bw_manual:.6f}")
                raise RuntimeError(
                    f"BW mismatch at batch {batch_idx}: tracked={bw_current:.6f}, manual={bw_manual:.6f}"
                )
            _imp(f"Verify batch {batch_idx}: BW OK (tracked={bw_current:.2f})")

    # ----------------- final save (best) -----------------
    _save_best(best_scores_global, output_path)
    _imp(f"Done. Final Current BW={bw_current:.2f}, Best BW={best_bw_global:.2f}, batches={batch_idx}, f2b_edges={len(f2b_edges_global)}")

    return best_scores_global, bw_history, f2b_edges_global, output_path

def compute_direction_change_stats(
    edges,
    before_scores,
    after_scores,
    restrict_to_edges=None,
):
    """
    Compute how many edges changed direction between two rankings and
    the total weight of those changes.

    IMPORTANT (consistency with BW tracking):
      - wt_B2F is the amount of weight REMOVED from backward edges (BW decreases by wt_B2F)
      - wt_F2B is the amount of weight ADDED to backward edges (BW increases by wt_F2B)
      - Therefore, net BW change = (+wt_F2B) - (wt_B2F)

    Parameters
    ----------
    edges : iterable of (u, v, w)
        All directed edges with weights.
    before_scores : dict
        node -> rank BEFORE the phase.
    after_scores : dict
        node -> rank AFTER the phase.
    restrict_to_edges : iterable of (u, v) or None
        If not None, only consider edges whose (u, v) is in this set.

    Returns
    -------
    num_B2F : int
        Number of edges that went from backward to forward.
    wt_B2F : float
        Total weight of edges that went from backward to forward. (BW decrease)
    num_F2B : int
        Number of edges that went from forward to backward.
    wt_F2B : float
        Total weight of edges that went from forward to backward. (BW increase)
    """
    restrict_set = set(restrict_to_edges) if restrict_to_edges is not None else None

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

        # Orientation before/after (ties ignored)
        before_forward = (bu < bv)
        before_backward = (bu > bv)
        after_forward = (au < av)
        after_backward = (au > av)

        if before_backward and after_forward:
            num_B2F += 1
            wt_B2F += float(w)
        elif before_forward and after_backward:
            num_F2B += 1
            wt_F2B += float(w)

    return num_B2F, wt_B2F, num_F2B, wt_F2B


# In[13]:
def hybrid_refine_func1_func2(
    csv_path,
    initial_ranking_path,
    output_ranking_path,        # single CSV that is always overwritten
    log_path,                   # text log file (explicit)
    h_hours: float = 72.0,       # kept for API compatibility; ignored if NEVER_STOP=True
    func2_f2b_limit: int = 200,
    func1_full_every: int = 50,
    func1_log_every_round: int = 10000,
    func1_bw_check_every_round: int = 500,   # BW not FW
    func2_verify_every: int = 5000,
    c: float = 1.0,             # kept for compatibility, IGNORED
    EPS_BW_IMPROVE: float = 1e-9,  # improvement threshold

    # ---------------- NEW: “never stop + quiet logs” knobs ----------------
    NEVER_STOP: bool = True,
    HEARTBEAT_SEC: float = 900.0,
    IDLE_SLEEP_SEC: float = 0.5,

    # ---------------- NEW: WMSF seed + escape ----------------
    WMSF_AT_START: bool = True,
    WMSF_START_ORDERING: str = "L2",           # try "L2" first (usually better than L1)
    WMSF_ESCAPE_SEC: float = 2 * 3600.0,       # if no NEW_BEST for this long, try WMSF escape
    WMSF_ESCAPE_MIN_GAP_SEC: float = 15 * 60.0,# don't spam WMSF retries
    WMSF_ORDERINGS=("L2", "L1"),               # try these orderings for escape, keep best if improves
    WMSF_MAX_SEC: float = 1800.0,              # soft budget for WMSF (seconds)
    WMSF_MAX_NODES: int = 20000,               # safety: skip WMSF if graph too big (set None to disable)
    WMSF_POLISH_GLOBAL_DP_BATCHES: int = 8,    # after accepting WMSF escape, polish with a short global func1
):
    """
    Option A (your requested behavior) + WMSF:

      - Hybrid loop: func2 ↔ func1 forever.
      - func1 is called in TARGETED mode and does ONLY ONE DP batch per call.
      - We maintain a persistent cycle pool: cycle_remaining.
      - We DO NOT refill cycle_remaining unless DP has nothing left to choose:
            i.e., backward_now_in_cycle == 0  (no backward edges remain inside cycle_remaining).
      - When we refill: cycle_remaining := ALL CURRENT backward edges in the WHOLE graph (under current scores).
      - After each func1 call, we remove the DP-selected batch edges from cycle_remaining,
        so every edge in a cycle is selected by DP at least once before refill.

    WMSF integration:
      - Seed at the beginning: WMSF produces a ranking; we accept it only if BW improves.
      - Escape local minima: if no NEW_BEST for WMSF_ESCAPE_SEC, try WMSF again (rate-limited).
        Accept only if BW improves; then run a short GLOBAL func1 polish and reset the cycle.

    REQUIREMENTS:
      - refine_ranking_parallel_dynamic must accept MAX_DP_BATCHES_PER_CALL.
      - you must have these functions available in your codebase:
          * maybe_run_wmsf_reseed(...)
          * save_scores_to_csv(...)
          * read_graph, load_initial_scores, compute_forward_weight
          * parallel_refine_largest_scc_intervals(...)
          * log_message(...)
    """
    import os
    import time
    import shutil

    # ------------------------------
    # Load graph once
    # ------------------------------
    edges, node_to_index, index_to_node = read_graph(csv_path)
    total_weight = sum(float(w) for (_, _, w) in edges)
    edges_dict = {(u, v): float(w) for (u, v, w) in edges}  # existence filter (kept)

    # ------------------------------
    # Ensure output directory exists
    # ------------------------------
    out_dir = os.path.dirname(output_ranking_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------
    # Processor counts (for log)
    # ------------------------------
    try:
        func2_procs = len(os.sched_getaffinity(0))
    except Exception:
        func2_procs = os.cpu_count() or 1
    func1_procs = os.cpu_count() or 1

    # ------------------------------
    # Time limit (disabled if NEVER_STOP)
    # ------------------------------
    start_ts = time.time()
    if NEVER_STOP:
        deadline = float("inf")
    else:
        deadline = start_ts + h_hours * 3600.0

    def _hours_left():
        if deadline == float("inf"):
            return 1e9
        return max(0.0, (deadline - time.time()) / 3600.0)

    def _now_str():
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # ------------------------------
    # State tracking
    # ------------------------------
    phase_idx = 0
    func1_runs = 0
    func2_runs = 0
    current_input_path = initial_ranking_path
    best_bw_seen = float("inf")
    last_best_ts = start_ts
    last_escape_attempt_ts = 0.0

    # ------------------------------
    # QUIET logging helpers
    # ------------------------------
    last_heartbeat_ts = start_ts

    def heartbeat(msg: str):
        nonlocal last_heartbeat_ts
        now = time.time()
        if (now - last_heartbeat_ts) >= HEARTBEAT_SEC:
            last_heartbeat_ts = now
            log_message(msg, log_path)

    # ------------------------------
    # Helpers
    # ------------------------------
    def bw_from_ranking_path(ranking_path: str):
        scores = load_initial_scores(ranking_path, node_to_index)
        fw = float(compute_forward_weight(edges, scores))
        bw = float(total_weight - fw)
        return scores, fw, bw

    def _phase_log_if_signal(phase_name, t0, bw_before, bw_after, extra=""):
        """Quiet: log only on improvement / NEW_BEST, otherwise heartbeat."""
        nonlocal best_bw_seen, last_best_ts
        dt = time.time() - t0
        dBW = bw_after - bw_before

        is_improve = (bw_before - bw_after) > EPS_BW_IMPROVE
        is_new_best = bw_after + EPS_BW_IMPROVE < best_bw_seen

        if is_new_best:
            best_bw_seen = bw_after
            last_best_ts = time.time()
            log_message(
                f"[HYBRID] phase={phase_idx:04d} {phase_name} | dt={dt:.2f}s | "
                f"BW {bw_before:.2f} -> {bw_after:.2f} (dBW={dBW:+.2f}) | "
                f"bestBW={best_bw_seen:.2f} [NEW_BEST] | {extra}",
                log_path
            )
        elif is_improve:
            log_message(
                f"[HYBRID] phase={phase_idx:04d} {phase_name} | dt={dt:.2f}s | "
                f"BW {bw_before:.2f} -> {bw_after:.2f} (dBW={dBW:+.2f}) | "
                f"bestBW={best_bw_seen:.2f} | {extra}",
                log_path
            )
        else:
            heartbeat(
                f"[HYBRID] heartbeat | phase={phase_idx:04d} | func1_runs={func1_runs} func2_runs={func2_runs} | "
                f"BW={bw_after:.2f} bestBW={best_bw_seen:.2f} | (quiet)"
            )

    def _count_backward_in_pairs(scores, pairs_iterable):
        b = 0
        for (u, v) in pairs_iterable:
            if scores[u] > scores[v]:
                b += 1
        return b

    def _all_current_backward_pairs(scores):
        out = set()
        for (u, v, _) in edges:
            if scores[u] > scores[v]:
                out.add((u, v))
        return out

    def _maybe_wmsf_escape():
        """
        Try WMSF escape (rate-limited). Accept only if BW improves.
        If accepted: overwrite output_ranking_path, reset cycle later, and polish with short global func1.
        """
        nonlocal current_input_path, last_escape_attempt_ts, phase_idx

        now = time.time()
        if (now - last_best_ts) < float(WMSF_ESCAPE_SEC):
            return False
        if (now - last_escape_attempt_ts) < float(WMSF_ESCAPE_MIN_GAP_SEC):
            return False

        last_escape_attempt_ts = now

        # BW before escape
        _, _, bw_before = bw_from_ranking_path(current_input_path)

        log_message(
            f"[HYBRID] WMSF_ESCAPE start | since_best={now-last_best_ts:.1f}s "
            f"(thr={WMSF_ESCAPE_SEC:.1f}s) | BW_before={bw_before:.2f}",
            log_path
        )

        best_path = None
        best_bw = bw_before
        tried = 0

        for ordn in list(WMSF_ORDERINGS):
            tried += 1
            tmp_path = output_ranking_path + f".wmsf_escape_{ordn}.csv"

            ok = maybe_run_wmsf_reseed(
                csv_path=csv_path,
                current_input_path=current_input_path,
                output_ranking_path=tmp_path,
                log_path=log_path,
                ordering=ordn,
                MAX_SEC=WMSF_MAX_SEC,
                MAX_NODES=WMSF_MAX_NODES,
            )

            if not ok:
                log_message(f"[HYBRID] WMSF_ESCAPE ordering={ordn} -> failed/skip", log_path)
                continue

            _, _, bw_tmp = bw_from_ranking_path(tmp_path)
            log_message(f"[HYBRID] WMSF_ESCAPE ordering={ordn} | BW={bw_tmp:.2f}", log_path)

            if bw_tmp + EPS_BW_IMPROVE < best_bw:
                best_bw = bw_tmp
                best_path = tmp_path

        if best_path is None:
            log_message(f"[HYBRID] WMSF_ESCAPE end | tried={tried} | no candidate produced", log_path)
            return False

        if best_bw + EPS_BW_IMPROVE < bw_before:
            # accept
            shutil.copyfile(best_path, output_ranking_path)
            current_input_path = output_ranking_path
            log_message(
                f"[HYBRID] WMSF_ESCAPE accepted | BW {bw_before:.2f} -> {best_bw:.2f} | picked={os.path.basename(best_path)}",
                log_path
            )

            # polish with a SHORT global func1
            phase_idx += 1
            t0 = time.time()
            _, _, bw_before_p = bw_from_ranking_path(current_input_path)

            log_message(
                f"[HYBRID] func1 POLISH after WMSF_ESCAPE | phase={phase_idx:04d} | "
                f"MAX_DP_BATCHES_PER_CALL={WMSF_POLISH_GLOBAL_DP_BATCHES}",
                log_path
            )

            refine_ranking_parallel_dynamic(
                csv_path=csv_path,
                initial_ranking_path=current_input_path,
                output_excel=output_ranking_path,
                log_path=log_path,
                MAX_HOURS=_hours_left(),
                LOG_EVERY_ROUND=func1_log_every_round,
                BW_CHECK_EVERY_ROUND=func1_bw_check_every_round,
                edge_subset=None,
                MAX_DP_BATCHES_PER_CALL=WMSF_POLISH_GLOBAL_DP_BATCHES,
            )

            _, _, bw_after_p = bw_from_ranking_path(output_ranking_path)
            _phase_log_if_signal("func1_polish_after_wmsf", t0, bw_before_p, bw_after_p, extra="mode=GLOBAL_POLISH")
            current_input_path = output_ranking_path
            return True

        # reject
        log_message(
            f"[HYBRID] WMSF_ESCAPE rejected | BW_before={bw_before:.2f} best_candidate={best_bw:.2f}",
            log_path
        )
        return False

    # ------------------------------
    # Header
    # ------------------------------
    log_message(
        f"[HYBRID] START {_now_str()} | NEVER_STOP={NEVER_STOP} | output={output_ranking_path} | "
        f"func1_procs={func1_procs} func2_procs={func2_procs} | EPS_BW_IMPROVE={EPS_BW_IMPROVE} | "
        f"HEARTBEAT_SEC={HEARTBEAT_SEC} IDLE_SLEEP_SEC={IDLE_SLEEP_SEC} | NOTE: c ignored | OPTION_A_CYCLE=ON | "
        f"WMSF_AT_START={WMSF_AT_START} WMSF_ESCAPE_SEC={WMSF_ESCAPE_SEC} WMSF_MAX_SEC={WMSF_MAX_SEC}",
        log_path
    )

    # =====================================================
    # (0) WMSF seed at start (accept only if improves BW)
    # =====================================================
    if WMSF_AT_START:
        try:
            _, _, bw_init = bw_from_ranking_path(current_input_path)
            seed_path = output_ranking_path + ".wmsf_seed.csv"

            log_message(f"[HYBRID] WMSF_START start | ordering={WMSF_START_ORDERING} | BW_init={bw_init:.2f}", log_path)

            ok = maybe_run_wmsf_reseed(
                csv_path=csv_path,
                current_input_path=current_input_path,
                output_ranking_path=seed_path,
                log_path=log_path,
                ordering=WMSF_START_ORDERING,
                MAX_SEC=WMSF_MAX_SEC,
                MAX_NODES=WMSF_MAX_NODES,
            )

            if ok:
                _, _, bw_seed = bw_from_ranking_path(seed_path)
                if bw_seed + EPS_BW_IMPROVE < bw_init:
                    shutil.copyfile(seed_path, output_ranking_path)
                    current_input_path = output_ranking_path
                    log_message(f"[HYBRID] WMSF_START accepted | BW {bw_init:.2f} -> {bw_seed:.2f}", log_path)
                else:
                    log_message(f"[HYBRID] WMSF_START rejected | BW_init={bw_init:.2f} BW_wmsf={bw_seed:.2f}", log_path)
            else:
                log_message("[HYBRID] WMSF_START failed/skip", log_path)
        except Exception as e:
            log_message(f"[HYBRID] WMSF_START exception: {e}", log_path)

    # =====================================================
    # PHASE 1: func1 GLOBAL (initial)
    # =====================================================
    phase_idx += 1
    func1_runs += 1
    t0 = time.time()

    _, _, bw_before = bw_from_ranking_path(current_input_path)
    log_message("[HYBRID] Running Initial Global func1...", log_path)

    refine_ranking_parallel_dynamic(
        csv_path=csv_path,
        initial_ranking_path=current_input_path,
        output_excel=output_ranking_path,
        log_path=log_path,
        MAX_HOURS=_hours_left(),
        LOG_EVERY_ROUND=func1_log_every_round,
        BW_CHECK_EVERY_ROUND=func1_bw_check_every_round,
        edge_subset=None,  # GLOBAL
    )

    _, _, bw_after = bw_from_ranking_path(output_ranking_path)
    _phase_log_if_signal("func1_initial_global", t0, bw_before, bw_after, extra="mode=GLOBAL")
    current_input_path = output_ranking_path

    # =====================================================
    # Option A cycle state (persistent)
    # =====================================================
    cycle_id = 0
    cycle_remaining = set()  # pairs (u,v) not yet DP-selected in THIS cycle

    # =====================================================
    # MAIN LOOP: func2 ↔ func1 forever (+ WMSF escape)
    # =====================================================
    try:
        while True:
            # ------------------------------------
            # (PRE) local-minima escape check (rate-limited)
            # ------------------------------------
            escaped = _maybe_wmsf_escape()
            if escaped:
                # ranking changed significantly; reset cycle so we start a clean cycle on new ordering
                cycle_remaining.clear()

            # ------------------------------------
            # (A) func2
            # ------------------------------------
            phase_idx += 1
            func2_runs += 1
            t0 = time.time()

            _, _, bw_before = bw_from_ranking_path(current_input_path)

            scores_after_func2, bw_history_func2, f2b_edges_global, out_path_used = parallel_refine_largest_scc_intervals(
                max_backward_flips=func2_f2b_limit,
                verify_every=func2_verify_every,
                csv_path=csv_path,
                initial_ranking_path=current_input_path,
                output_path=output_ranking_path,
                debug=False,
                log_path=log_path,
            )

            _, _, bw_after = bw_from_ranking_path(output_ranking_path)
            flips_raw = len(f2b_edges_global or [])
            if flips_raw > 0:
                log_message(f"[HYBRID] func2_signal | phase={phase_idx:04d} | flips_raw={flips_raw}", log_path)

            _phase_log_if_signal(f"func2_run{func2_runs}", t0, bw_before, bw_after, extra=f"flips_raw={flips_raw}")
            current_input_path = output_ranking_path

            # ------------------------------------
            # (B) Option A: func1 TARGETED, EXACTLY ONE DP batch per call
            # ------------------------------------
            scores_now, _, bw_before_func1 = bw_from_ranking_path(current_input_path)

            backward_now_in_cycle = _count_backward_in_pairs(scores_now, cycle_remaining) if cycle_remaining else 0
            if (not cycle_remaining) or (backward_now_in_cycle == 0):
                cycle_id += 1
                t_refill0 = time.time()
                cycle_remaining = _all_current_backward_pairs(scores_now)
                t_refill = time.time() - t_refill0
                log_message(
                    f"[HYBRID] cycle_refill | cycle_id={cycle_id} | backward_pairs={len(cycle_remaining)} | refill_dt={t_refill:.2f}s",
                    log_path
                )
                backward_now_in_cycle = len(cycle_remaining)

            if backward_now_in_cycle == 0:
                heartbeat(
                    f"[HYBRID] heartbeat | phase={phase_idx:04d} | cycle_id={cycle_id} | no_backward_edges_global | BW={bw_before_func1:.2f}"
                )
                if IDLE_SLEEP_SEC > 0:
                    time.sleep(IDLE_SLEEP_SEC)
                continue

            # run one DP batch
            phase_idx += 1
            func1_runs += 1
            t0 = time.time()

            subset_list = list(cycle_remaining)

            log_message(
                f"[HYBRID] Running func1 TARGETED | phase={phase_idx:04d} | cycle_id={cycle_id} | "
                f"cycle_remaining={len(cycle_remaining)} backward_now_in_cycle={backward_now_in_cycle} | one_dp_batch=YES",
                log_path
            )

            _, _, remaining_pairs_unassessed = refine_ranking_parallel_dynamic(
                csv_path=csv_path,
                initial_ranking_path=current_input_path,
                output_excel=output_ranking_path,
                log_path=log_path,
                MAX_HOURS=_hours_left(),
                LOG_EVERY_ROUND=func1_log_every_round,
                BW_CHECK_EVERY_ROUND=func1_bw_check_every_round,
                edge_subset=subset_list,
                MAX_DP_BATCHES_PER_CALL=10,  # <-- FIXED: EXACTLY ONE batch per call
            )

            remaining_pairs_set = set(remaining_pairs_unassessed or [])
            assessed_pairs = set(subset_list) - remaining_pairs_set
            cycle_remaining.difference_update(assessed_pairs)

            _, _, bw_after_func1 = bw_from_ranking_path(output_ranking_path)
            improve_amt = bw_before_func1 - bw_after_func1

            _phase_log_if_signal(
                f"func1_targeted_batch{func1_runs}",
                t0,
                bw_before_func1,
                bw_after_func1,
                extra=(
                    f"cycle_id={cycle_id} assessed_batch={len(assessed_pairs)} "
                    f"cycle_remaining_after={len(cycle_remaining)} improve={improve_amt:.6g}"
                )
            )

            current_input_path = output_ranking_path

            # ------------------------------------
            # (C) periodic GLOBAL refresh (forced)
            # ------------------------------------
            if func1_full_every > 0 and (func1_runs % func1_full_every == 0):
                phase_idx += 1
                t0 = time.time()

                _, _, bw_before_p = bw_from_ranking_path(current_input_path)
                log_message(
                    f"[HYBRID] Periodic Global func1 (forced) | every={func1_full_every} | phase={phase_idx:04d}",
                    log_path
                )

                refine_ranking_parallel_dynamic(
                    csv_path=csv_path,
                    initial_ranking_path=current_input_path,
                    output_excel=output_ranking_path,
                    log_path=log_path,
                    MAX_HOURS=_hours_left(),
                    LOG_EVERY_ROUND=func1_log_every_round,
                    BW_CHECK_EVERY_ROUND=func1_bw_check_every_round,
                    edge_subset=None,
                )

                _, _, bw_after_p = bw_from_ranking_path(output_ranking_path)

                _phase_log_if_signal(
                    "func1_periodic_global",
                    t0,
                    bw_before_p,
                    bw_after_p,
                    extra="mode=GLOBAL (forced)"
                )

                current_input_path = output_ranking_path
                cycle_remaining.clear()

    except KeyboardInterrupt:
        log_message(
            f"[HYBRID] INTERRUPTED {_now_str()} | phases={phase_idx} func1_runs={func1_runs} func2_runs={func2_runs} "
            f"cycle_id={cycle_id} bestBW={best_bw_seen:.2f}",
            log_path
        )
        return output_ranking_path


# In[14]:


def compute_backward_weight(edges, scores):
    """
    Backward Weight (BW): total weight of edges (u->v) that go backward w.r.t. scores,
    i.e., score[u] > score[v]. (Scores are unique in your pipeline.)
    """
    bw = 0.0
    for (u, v, w) in edges:
        if scores[u] > scores[v]:
            bw += float(w)
    return bw


# In[15]:


def generate_scc_ranking(edges, index_to_node, log_path):
    """
    Generates a ranking based on the topological order of Strongly Connected Components (SCCs).

    Input
    -----
    edges : list of (u_idx, v_idx, weight)
        Directed weighted edges (weight is ignored for SCC structure).
    index_to_node : dict
        Maps internal index -> original node ID/name.
    log_path : str
        Path for log_message.

    Output
    ------
    dict {original_node_id/name: rank}
        A total order produced by ordering SCCs topologically, then listing members within each SCC.
        (Order inside an SCC is arbitrary but deterministic here.)
    """
    import networkx as nx

    log_message("generate_scc_ranking: building DiGraph from edges...", log_path)

    G = nx.DiGraph()
    for u, v, _w in edges:
        G.add_edge(u, v)

    # Ensure all nodes exist (including isolated nodes)
    for idx in index_to_node.keys():
        if idx not in G:
            G.add_node(idx)

    log_message("generate_scc_ranking: condensing graph into SCC DAG...", log_path)
    scc_graph = nx.condensation(G)

    log_message("generate_scc_ranking: topological sorting SCC DAG...", log_path)
    scc_topo_order = list(nx.topological_sort(scc_graph))

    log_message("generate_scc_ranking: assigning ranks to nodes...", log_path)
    node_ranks = {}
    current_rank = 0

    # Determinism: sort members so output is stable across runs
    for scc_id in scc_topo_order:
        members = scc_graph.nodes[scc_id].get("members", [])
        for member_idx in sorted(members):
            original_node_name = index_to_node[member_idx]
            node_ranks[original_node_name] = current_rank
            current_rank += 1

    log_message(
        f"generate_scc_ranking: done. n_nodes_ranked={len(node_ranks)}, n_scc={len(scc_topo_order)}",
        log_path
    )
    return node_ranks

def compare_forward_weights(edges, scores_before, scores_after):
    """
    Compare edge orientations between two rankings.

    Returns:
      sum_sbef_only:    total weight of edges that are forward in BEFORE but NOT forward in AFTER
                        (i.e., forward->backward OR forward->tie)
      sum_saft_only:    total weight of edges that are forward in AFTER but NOT forward in BEFORE
                        (i.e., backward->forward OR tie->forward)
      sum_both_forward: total weight of edges forward in BOTH
      sum_both_backward:total weight of edges backward in BOTH
      only_before_global: set of (u, v) edges that went forward->backward (strictly)
    """
    sum_sbef_only = 0.0
    sum_saft_only = 0.0
    sum_both_forward = 0.0
    sum_both_backward = 0.0
    only_before_global = set()

    for (u, v, w) in edges:
        bu = scores_before[u]
        bv = scores_before[v]
        au = scores_after[u]
        av = scores_after[v]

        before_forward = (bu < bv)
        before_backward = (bu > bv)

        after_forward = (au < av)
        after_backward = (au > av)

        # forward in BOTH
        if before_forward and after_forward:
            sum_both_forward += float(w)

        # backward in BOTH
        if before_backward and after_backward:
            sum_both_backward += float(w)

        # forward only in BEFORE (not forward after)
        if before_forward and (not after_forward):
            sum_sbef_only += float(w)
            # specifically forward->backward (strict), collect as F2B edge
            if after_backward:
                only_before_global.add((u, v))

        # forward only in AFTER (not forward before)
        if after_forward and (not before_forward):
            sum_saft_only += float(w)

    return sum_sbef_only, sum_saft_only, sum_both_forward, sum_both_backward, only_before_global


# In[16]:


# 1) Define Base Directory and Input File
import os
import datetime
import pandas as pd

base_dir = "/mmfs1/home/sv96/Feedback-arc-set-paper/datasets/"
graph_file_path = os.path.join(base_dir, "ecc.d")

# IMPORTANT: set an explicit log path for ALL logs in this pipeline
# 1) Extract graph name (remove directory and ".d" extension)
graph_basename = os.path.basename(graph_file_path).replace(".d", "")

# Define log path dynamically based on graph name
log_path = os.path.join(base_dir, f"{graph_basename}_dimacs_hybrid_phases_log-48cpu.txt")

# 0) Clear log file before starting
# Opening in 'w' mode truncates the file to 0 bytes
open(log_path, 'w').close()

# 2) Read Graph (log via log_message only)
log_message(f"Reading graph from {graph_file_path} ...", log_path)

# Note: read_graph returns (edges_indexed, node_to_index, index_to_node)
edges, node_to_index, index_to_node = read_graph(graph_file_path)

log_message(
    f"Graph loaded: n_nodes={len(node_to_index)}, n_edges={len(edges)}",
    log_path
)


# 3) Generate SCC-based Ranking (log via log_message only)
scc_ranking_dict = generate_scc_ranking(edges, index_to_node, log_path)


# 4) Save SCC Ranking to CSV (no prints; log via log_message only)
# We match the columns expected by load_initial_scores: "Node ID" and "Order"
initial_scc_csv_path = os.path.join(base_dir, f"{graph_basename}_dimacs_scc_init.csv")

data_rows = [{"Node ID": node_name, "Order": int(score)} for node_name, score in scc_ranking_dict.items()]
pd.DataFrame(data_rows).to_csv(initial_scc_csv_path, index=False)

log_message(f"Generated SCC-based initial ranking at: {initial_scc_csv_path}", log_path)


# 5) Call the Hybrid Function (single output CSV always overwritten)
final_ranking_path = hybrid_refine_func1_func2(
    csv_path=graph_file_path,
    initial_ranking_path=initial_scc_csv_path,  # SCC init ranking
    output_ranking_path=os.path.join(base_dir, f"{graph_basename}_dimacs_hybrid_ranking-48cpu.csv"),
    log_path=log_path,
    h_hours=72.0,
    c=1.0,  # ignored (kept for compatibility)
)

log_message(f"Hybrid finished. Final ranking path: {final_ranking_path}", log_path)


# In[ ]:




