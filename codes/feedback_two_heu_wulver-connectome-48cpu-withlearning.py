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

from collections import defaultdict

def read_graph(file_path):
    """
    Reads a DIMACS-like file with lines:
      a <source> <target> <weight> <transit_time>

    Paper/WMSF-friendly behavior:
      - Aggregates parallel arcs: multiple (u,v) become one (u,v) with summed weight.
      - Builds a deterministic node mapping.
      - Skips malformed lines safely.
      - Accepts extra fields after weight (e.g., transit_time) and ignores them.

    Returns:
      edges_indexed: list[(u_idx, v_idx, w_sum)]
      node_to_index: dict[node_id_str -> int]
      index_to_node: dict[int -> node_id_str]
    """
    from collections import defaultdict

    agg = defaultdict(float)
    node_ids = set()

    with open(file_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # DIMACS-like comments / headers
            if line.startswith(("c", "p")):
                continue
            if not line.startswith("a"):
                continue

            parts = line.split()
            # Expect at least: a u v w
            if len(parts) < 4:
                continue

            u = parts[1]
            v = parts[2]
            try:
                w = float(parts[3])
            except ValueError:
                continue

            # Track nodes and aggregate weight
            node_ids.add(u)
            node_ids.add(v)
            agg[(u, v)] += w

    # Deterministic mapping
    node_list = sorted(node_ids)
    node_to_index = {node: i for i, node in enumerate(node_list)}
    index_to_node = {i: node for node, i in node_to_index.items()}

    # Deterministic edge list (sort by (u_idx, v_idx))
    edges_indexed = []
    for (u, v), w_sum in agg.items():
        edges_indexed.append((node_to_index[u], node_to_index[v], float(w_sum)))
    edges_indexed.sort(key=lambda e: (e[0], e[1]))

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

import os
import csv
import time

def ml_append_row(ml_csv_path: str, row: dict, log_path: str | None = None) -> None:
    """
    Append one training example for ML.

    - row must be a flat dict (no nested dicts/lists).
    - creates the CSV if it doesn't exist (writes header once).
    - never prints; optionally logs failures via log_message.
    """
    try:
        # Make sure directory exists
        d = os.path.dirname(ml_csv_path)
        if d:
            os.makedirs(d, exist_ok=True)

        # Ensure stable column order: existing header if file exists, else row keys
        write_header = not os.path.exists(ml_csv_path) or os.path.getsize(ml_csv_path) == 0

        # Add a timestamp if not present
        if "ts" not in row:
            row = dict(row)
            row["ts"] = time.time()

        if write_header:
            fieldnames = list(row.keys())
        else:
            # Read header to keep consistent columns
            with open(ml_csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                fieldnames = next(reader)

            # If new keys appear, extend schema (simple approach: rewrite not done here).
            # So enforce keys subset to avoid breaking rows:
            for k in row.keys():
                if k not in fieldnames:
                    # drop unseen keys to keep file consistent
                    row.pop(k, None)

        with open(ml_csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row)

    except Exception as e:
        if log_path is not None:
            log_message(f"[ML] Failed to append row to {ml_csv_path}: {e}", log_path)


def log_message(message, log_file_path):
    """
    Extremely-low-volume logger for long (72h) runs.

    - Keeps the same signature (so you don't need to change call sites).
    - Drops non-important lines by default (to avoid filling disk).
    - Rate-limits repeated messages and can append "(suppressed N repeats)".
    - If log file grows too large, it will stop writing non-fatal lines.

    Tuning (edit if you want):
      IMPORTANT_SUBSTRINGS: only messages containing one of these are logged (unless FATAL/ERROR).
      MIN_REPEAT_INTERVAL_SEC: minimum seconds between identical messages.
      MAX_LOG_BYTES: after this, only FATAL/ERROR messages are logged.
    """
    import os
    import time
    import datetime

    if not log_file_path:
        return

    # ----------------- knobs -----------------
    IMPORTANT_SUBSTRINGS = (
        "❌ FATAL", "FATAL", "ERROR", "RuntimeError", "Traceback",
        "NEW_BEST", "Checkpoint", "Verify", "VERIFY",
        "Start", "Done", "Stop", "STOP",
        # keep your phase tags (but still filtered by the keywords above if you want even quieter)
        "[HYBRID]", "[FUNC1]", "[FUNC2]",
    )

    # If you want even quieter: comment out the phase tags above and rely only on NEW_BEST/Stop/etc.

    MIN_REPEAT_INTERVAL_SEC = 60.0          # suppress identical messages within 60s
    MAX_LOG_BYTES = 250 * 1024 * 1024       # 250MB; after this -> only FATAL/ERROR
    SIZE_CHECK_EVERY_SEC = 30.0             # don't stat the file every call

    # ----------------- init static state -----------------
    st = getattr(log_message, "_state", None)
    if st is None:
        st = {
            "last_write_ts": {},     # key -> ts
            "suppressed": {},        # key -> count
            "last_size_check_ts": 0.0,
            "too_big": False,
        }
        setattr(log_message, "_state", st)

    now = time.time()

    # ----------------- hard allow / deny -----------------
    msg = str(message)

    is_fatal = ("❌" in msg) or ("FATAL" in msg) or ("ERROR" in msg) or ("Traceback" in msg) or ("RuntimeError" in msg)

    # size guard (once in a while)
    if (not st["too_big"]) and (now - st["last_size_check_ts"] >= SIZE_CHECK_EVERY_SEC):
        st["last_size_check_ts"] = now
        try:
            if os.path.exists(log_file_path) and os.path.getsize(log_file_path) >= MAX_LOG_BYTES:
                st["too_big"] = True
        except Exception:
            # if stat fails, ignore
            pass

    # If log is too big, only allow fatal messages
    if st["too_big"] and (not is_fatal):
        return

    # importance filter (fatal always passes)
    if not is_fatal:
        if not any(s in msg for s in IMPORTANT_SUBSTRINGS):
            return

    # ----------------- rate-limit duplicates -----------------
    # Key = the raw message text (you can normalize if you prefer)
    key = msg

    last_ts = st["last_write_ts"].get(key)
    if last_ts is not None and (now - last_ts) < MIN_REPEAT_INTERVAL_SEC and (not is_fatal):
        st["suppressed"][key] = st["suppressed"].get(key, 0) + 1
        return

    # If we previously suppressed repeats of this message, append a compact note once
    sup = st["suppressed"].pop(key, 0)
    if sup and (not is_fatal):
        msg = f"{msg} (suppressed {sup} repeats)"

    st["last_write_ts"][key] = now

    # ----------------- write -----------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {msg}\n"

    # 'a' mode appends; keep it simple/fast
    try:
        with open(log_file_path, "a") as f:
            f.write(formatted_message)
    except Exception:
        # Never crash the algorithm because logging failed
        return

    
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
    UPDATED (more general + correct):

    We optimize the ordering inside the block:
        [v] + between + [u]
    where (u,v) is a backward edge in the current ranking (so rank(v) < rank(u)).

    Previous version forced v to be placed immediately after u (… [u, v] …).
    This updated version enforces ONLY the necessary constraint:
        rank(u) < rank(v)   (i.e., v goes after u)
    while allowing an arbitrary (stable) number of nodes from `between` to sit between u and v.

    We keep the relative order of nodes in `between` unchanged (stable), and only choose
    two cut indices a,b (0 <= a <= b <= n):

        between[0:a] + [u] + between[a:b] + [v] + between[b:n]

    Objective:
      maximize BW reduction (equivalently FW increase) considering only edges involving u or v
      (and the edge between u and v). Other edges among `between` nodes keep the same relative order.

    Complexity:
      O(|between|) edge lookups + O(|between|) preprocessing + O(|between|) scan.
    """
    import time
    import heapq  # (not used, but keep import pattern stable if you later add heaps)

    PROFILE_THRESHOLD = 0.08  # seconds
    t_total0 = time.time()

    # Snapshot ranks of endpoints (do NOT copy full scores)
    ru0 = scores[u]
    rv0 = scores[v]
    ranks_before = (ru0, rv0)

    # Defensive: ensure u,v are not inside between
    if between:
        if u in between or v in between:
            between = [x for x in between if x != u and x != v]

    # ---------- 0) Ensure `between` is rank-sorted ----------
    # We need between ordered by increasing rank in the current permutation.
    if len(between) >= 2:
        is_sorted = True
        prev = scores[between[0]]
        for node in between[1:]:
            r = scores[node]
            if r < prev:
                is_sorted = False
                break
            prev = r
        if not is_sorted:
            if debug:
                log_message(
                    f"[apply_new_strategy][WARN] between not sorted; fixing. "
                    f"u={u} v={v} ru={ru0} rv={rv0} |between|={len(between)}",
                    log_path
                )
            between = sorted(between, key=lambda n: scores[n])

    nB = len(between)

    # If block is trivial, only possible improvement is swapping u/v order.
    # But in this routine, we still allow the general logic to handle it.
    # ---------- 1) Precompute per-node contributions ----------
    # du[i] = FW gain from flipping u vs between[i] (since u moves BEFORE between[i])
    #       = w(u->x) - w(x->u)
    # dv[i] = FW gain from flipping v vs between[i] (since v moves AFTER between[i])
    #       = w(x->v) - w(v->x)
    #
    # Original block order is: v before all between, and all between before u.
    # New order changes u relative to nodes in between[a:], and changes v relative to nodes in between[:b].
    du = [0.0] * nB
    dv = [0.0] * nB

    # Loop once over between to compute both arrays (O(nB)).
    for i, x in enumerate(between):
        # u vs x
        du[i] = edges_dict.get((u, x), 0.0) - edges_dict.get((x, u), 0.0)
        # v vs x
        dv[i] = edges_dict.get((x, v), 0.0) - edges_dict.get((v, x), 0.0)

    # Edge between u and v (originally v before u, new u before v)
    w_uv = edges_dict.get((u, v), 0.0)
    w_vu = edges_dict.get((v, u), 0.0)
    base_bw_reduction = w_uv - w_vu  # == FW increase from flipping u/v

    # ---------- 2) Build suffix for u part: U_suffix[a] = sum_{i=a..nB-1} du[i] ----------
    U_suffix = [0.0] * (nB + 1)
    for i in range(nB - 1, -1, -1):
        U_suffix[i] = U_suffix[i + 1] + du[i]
    # U_suffix[nB] = 0 already

    # ---------- 3) Build prefix for v part: V_prefix[b] = sum_{i=0..b-1} dv[i] ----------
    # Here b is "how many between nodes are placed before v" (0..nB).
    V_prefix = [0.0] * (nB + 1)
    for i in range(nB):
        V_prefix[i + 1] = V_prefix[i] + dv[i]

    # ---------- 4) Suffix max of V_prefix to answer max_{b>=a} V_prefix[b] in O(1) ----------
    bestV_from = [0.0] * (nB + 1)     # best value among b in [a..nB]
    bestB_from = [0] * (nB + 1)       # argmax b for that best value

    bestV_from[nB] = V_prefix[nB]
    bestB_from[nB] = nB
    for a in range(nB - 1, -1, -1):
        # prefer earlier b if equal (deterministic), or later? Either is fine.
        # Earlier b keeps v more left (usually cheaper move); we'll pick earlier for stability.
        v_here = V_prefix[a]
        v_next = bestV_from[a + 1]
        if v_here >= v_next:
            bestV_from[a] = v_here
            bestB_from[a] = a
        else:
            bestV_from[a] = v_next
            bestB_from[a] = bestB_from[a + 1]

    # ---------- 5) Choose best cuts (a,b) with constraint b >= a ----------
    best_bw_reduction = float("-inf")
    best_a = -1
    best_b = -1
    best_u_part = 0.0
    best_v_part = 0.0

    for a in range(nB + 1):
        b = bestB_from[a]
        # total BW reduction (FW increase)
        bw_reduction = base_bw_reduction + U_suffix[a] + bestV_from[a]
        if bw_reduction > best_bw_reduction:
            best_bw_reduction = bw_reduction
            best_a = a
            best_b = b
            best_u_part = U_suffix[a]
            best_v_part = bestV_from[a]

    # ---------- 6) Early exit if no improvement ----------
    if best_bw_reduction <= 0 or best_a < 0 or best_b < 0:
        ranks_after = (scores[u], scores[v])
        if debug:
            log_message(
                f"[apply_new_strategy] NO-IMPROVE u={u} v={v} best={best_bw_reduction:.6g} "
                f"base={base_bw_reduction:.6g}",
                log_path
            )
        return False, 0.0, ranks_before, ranks_after, {}

    # ---------- 7) Apply reordering (success) ----------
    # New order inside the block:
    #   between[0:best_a] + [u] + between[best_a:best_b] + [v] + between[best_b:]
    left = between[:best_a]
    mid = between[best_a:best_b]
    right = between[best_b:]
    new_order = left + [u] + mid + [v] + right

    # Optional (debug) exact objective sanity: expensive but very informative.
    bw_before = None
    if debug:
        try:
            bw_before = compute_backward_weight(edges_dict_to_edge_list(edges_dict), scores)
        except Exception:
            bw_before = None

    base_score = rv0  # preserve the block's rank interval start
    for k, node in enumerate(new_order):
        scores[node] = base_score + k

    ranks_after = (scores[u], scores[v])

    info = {
        "a": best_a,
        "b": best_b,
        "base_uv": base_bw_reduction,
        "u_part": best_u_part,
        "v_part": best_v_part,
        "between_len": nB,
    }

    if debug:
        log_message(
            f"[apply_new_strategy] SUCCESS u={u} v={v} a={best_a} b={best_b} "
            f"best_bw_reduction={best_bw_reduction:.6f} (base={base_bw_reduction:.6f} "
            f"uPart={best_u_part:.6f} vPart={best_v_part:.6f}) base_score={base_score}",
            log_path
        )

    if debug and bw_before is not None:
        try:
            bw_after = compute_backward_weight(edges_dict_to_edge_list(edges_dict), scores)
            if bw_after > bw_before + 1e-9:
                log_message(
                    f"[apply_new_strategy][ERROR] BW increased! before={bw_before:.3f} after={bw_after:.3f} "
                    f"u={u} v={v} a={best_a} b={best_b} best={best_bw_reduction:.6f}",
                    log_path
                )
        except Exception:
            pass

    # ---------- 8) Profiling log ----------
    t_total = time.time() - t_total0
    if t_total > PROFILE_THRESHOLD:
        log_message(
            f"[apply_new_strategy][SLOW] u={u} v={v} |between|={nB} a={best_a} b={best_b} "
            f"best_bw_reduction={best_bw_reduction:.3f} t={t_total:.4f}s",
            log_path
        )

    return True, best_bw_reduction, ranks_before, ranks_after, info

# --- helper: only needed if you keep the debug BW sanity above ---
def edges_dict_to_edge_list(edges_dict):
    """
    Convert edges_dict {(u,v): w} into an edge list [(u,v,w), ...].
    NOTE: This assumes edges_dict already aggregates parallel edges.
    """
    return [(u, v, w) for (u, v), w in edges_dict.items()]

# In[4]:


def compute_all_gains_local(scores, u, v, out_edges, in_edges, log_path=None):
    """
    Compute BW reduction gains for three local reorder strategies around backward edge (u->v):
      1) swap u and v
      2) move v after u
      3) move u before v

    Returns: (swap_gain, mvvafu_gain, mvubfv_gain)
    where each gain is "BW reduction" (positive is good).

    Updates vs your version:
      - safe .get(...) for adjacency lists (avoids KeyError)
      - no set() of edges; uses a lightweight 'seen' set on (a,b) only
      - optional log_path parameter for worker compatibility (no TypeError overhead)
    """
    idx_u = scores[u]
    idx_v = scores[v]
    if idx_u <= idx_v:
        raise ValueError(f"Invalid edge ({u}->{v}) is already forward!")

    lo, hi = idx_v, idx_u

    scores_local = scores
    idx_u_local = idx_u
    idx_v_local = idx_v

    def new_rank(node, rank_node, strategy):
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

        return rank_node  # defensive

    swap_gain = 0.0
    mvvafu_gain = 0.0
    mvubfv_gain = 0.0

    # Safe adjacency fetch: nodes may have no in/out list
    out_u = out_edges.get(u, [])
    in_u  = in_edges.get(u, [])
    out_v = out_edges.get(v, [])
    in_v  = in_edges.get(v, [])

    # We'll only evaluate edges that touch [lo, hi].
    # To avoid double counting without hashing floats, de-dup by directed pair (a,b).
    seen = set()

    def process_edge(a, b, w):
        nonlocal swap_gain, mvvafu_gain, mvubfv_gain

        sa = scores_local[a]
        sb = scores_local[b]

        # Only edges touching [lo, hi] matter
        if not (lo <= sa <= hi or lo <= sb <= hi):
            return

        old_forward = (sa < sb)

        # ---- swap ----
        ra2 = new_rank(a, sa, 'swap')
        rb2 = new_rank(b, sb, 'swap')
        new_forward = (ra2 < rb2)
        if old_forward and not new_forward:
            swap_gain -= w
        elif (not old_forward) and new_forward:
            swap_gain += w

        # ---- mvvafu ----
        ra2 = new_rank(a, sa, 'mvvafu')
        rb2 = new_rank(b, sb, 'mvvafu')
        new_forward = (ra2 < rb2)
        if old_forward and not new_forward:
            mvvafu_gain -= w
        elif (not old_forward) and new_forward:
            mvvafu_gain += w

        # ---- mvubfv ----
        ra2 = new_rank(a, sa, 'mvubfv')
        rb2 = new_rank(b, sb, 'mvubfv')
        new_forward = (ra2 < rb2)
        if old_forward and not new_forward:
            mvubfv_gain -= w
        elif (not old_forward) and new_forward:
            mvubfv_gain += w

    # Helper: add an edge once
    def add_edge(a, b, w):
        key = (a, b)
        if key in seen:
            return
        seen.add(key)
        process_edge(a, b, float(w))

    # Only neighbors of u or v can change status under these interval reorders
    scores_vals = scores_local

    for x, w in out_u:
        rx = scores_vals[x]
        if lo <= rx <= hi:
            add_edge(u, x, w)

    for x, w in in_u:
        rx = scores_vals[x]
        if lo <= rx <= hi:
            add_edge(x, u, w)

    for x, w in out_v:
        rx = scores_vals[x]
        if lo <= rx <= hi:
            add_edge(v, x, w)

    for x, w in in_v:
        rx = scores_vals[x]
        if lo <= rx <= hi:
            add_edge(x, v, w)

    return swap_gain, mvvafu_gain, mvubfv_gain



def select_nonconflicting_edge_indices_dp(
    backward_edges,
    *,
    edge_scores=None,                 # optional: list/tuple/dict giving a score per LOCAL edge index
    maximize: str = "count",          # "count" (old behavior) or "score" (weighted by edge_scores)
    allow_touching: bool = False,     # if True: allow hi == lo between chosen intervals (non-overlap allows touching)
    prefer_higher_score_on_tie: bool = True,  # tie-break when dp values equal
):
    """
    Interval scheduling DP on edges with intervals [lo, hi].

    Input:
      backward_edges: list of tuples (u, v, w, lo, hi) OR (u, v, w, lo, hi, ...extra)
                      - we only read positions 3 and 4 as lo,hi.
      edge_scores:
          - None: every edge has score 1.0 (same as maximize="count").
          - list/tuple: edge_scores[idx] used for LOCAL edge idx
          - dict: edge_scores[idx] used (missing -> 0.0)
      maximize:
          - "count": maximize number of chosen edges (classic maximum-cardinality, old behavior)
          - "score": maximize sum(edge_scores) (weighted interval scheduling)
      allow_touching:
          - False (default): require strictly non-overlapping, i.e., previous hi < current lo
          - True: allow boundary touch, i.e., previous hi <= current lo
      prefer_higher_score_on_tie:
          - if dp "take" and "skip" are equal, optionally choose "take" when it has >= edge score.

    Returns:
      selected_indices: set of LOCAL indices into backward_edges.

    Complexity:
      O(E log E) due to sorting + binary searches.
    """
    if not backward_edges:
        return set()

    # ----------------------------
    # Score accessor
    # ----------------------------
    def _score(local_idx: int) -> float:
        if maximize == "count":
            return 1.0
        # maximize == "score"
        if edge_scores is None:
            return 1.0
        if isinstance(edge_scores, dict):
            try:
                return float(edge_scores.get(local_idx, 0.0))
            except Exception:
                return 0.0
        # list/tuple/sequence
        try:
            return float(edge_scores[local_idx])
        except Exception:
            return 0.0

    # ----------------------------
    # 1) Collect all distinct endpoints
    # ----------------------------
    endpoints = set()
    for e in backward_edges:
        # expect lo,hi at positions 3,4
        lo = e[3]
        hi = e[4]
        endpoints.add(lo)
        endpoints.add(hi)

    sorted_endpoints = sorted(endpoints)
    coord_index = {rank: i for i, rank in enumerate(sorted_endpoints)}

    # ----------------------------
    # 2) Convert edges to compressed coordinates and sort by hi_pos
    #    Keep local index and score.
    # ----------------------------
    edges = []  # (hi_pos, lo_pos, local_idx, score)
    for idx, e in enumerate(backward_edges):
        lo = e[3]
        hi = e[4]
        lo_pos = coord_index[lo]
        hi_pos = coord_index[hi]
        s = _score(idx)
        edges.append((hi_pos, lo_pos, idx, float(s)))

    edges.sort(key=lambda x: x[0])  # sort by hi_pos
    hi_list = [t[0] for t in edges]

    # ----------------------------
    # helper: p(i) via binary search on hi_list
    # ----------------------------
    import bisect

    def _rightmost_compatible_edge_index(lo_pos: int) -> int:
        """
        Return largest j such that:
          - if allow_touching: hi_list[j] <= lo_pos
          - else:              hi_list[j] <  lo_pos
        """
        if allow_touching:
            # rightmost hi <= lo_pos
            j = bisect.bisect_right(hi_list, lo_pos) - 1
        else:
            # rightmost hi < lo_pos
            j = bisect.bisect_left(hi_list, lo_pos) - 1
        return j

    n = len(edges)

    # p[i] = last compatible edge index (0-based into `edges`) for edge i
    p = [-1] * n
    for i in range(n):
        hi_pos, lo_pos, _local_idx, _s = edges[i]
        p[i] = _rightmost_compatible_edge_index(lo_pos)

    # ----------------------------
    # 3) DP
    # dp[i] = best value using first i edges (edges[0..i-1])
    # take[i] = whether we take edge i-1
    # ----------------------------
    dp = [0.0] * (n + 1)
    take = [False] * (n + 1)

    for i in range(1, n + 1):
        # skip edge i-1
        opt1 = dp[i - 1]

        # take edge i-1
        j = p[i - 1]
        edge_val = edges[i - 1][3]  # score
        opt2 = edge_val + (dp[j + 1] if j >= 0 else 0.0)

        if opt2 > opt1:
            dp[i] = opt2
            take[i] = True
        elif opt2 < opt1:
            dp[i] = opt1
            take[i] = False
        else:
            # tie: choose based on preference
            if prefer_higher_score_on_tie:
                # if "take" has non-negative score, prefer take (gives more coverage / more info)
                dp[i] = opt2
                take[i] = True
            else:
                dp[i] = opt1
                take[i] = False

    # ----------------------------
    # 4) Reconstruct chosen edges (LOCAL indices)
    # ----------------------------
    selected_indices = set()
    i = n
    while i > 0:
        if take[i]:
            _hi_pos, _lo_pos, local_idx, _s = edges[i - 1]
            selected_indices.add(int(local_idx))
            j = p[i - 1]
            i = j + 1
        else:
            i -= 1

    return selected_indices


# In[6]:


def worker_loop(
    worker_idx, num_procs,
    backward_edges,
    scores_snapshot,
    out_edges, in_edges, edges_dict,
    active_intervals,   # unused (kept for API compatibility; may be None)
    used_intervals,     # unused (kept for API compatibility; may be None)
    lock,               # unused (kept for API compatibility; may be None)
    edge_queue,
    result_queue,
    log_path,
    debug: bool = False,
    DEBUG_SAMPLE_EVERY: int = 200,     # log ~1 per N "none" edges (per worker)
    DEBUG_MAX_EDGE_LOGS: int = 40,     # hard cap per worker
    DEBUG_LOG_TOP_GAINS: bool = True,  # log any unusually large gains
    DEBUG_BIG_GAIN: float = 1e-3,      # "big" gain threshold (BW reduction)

    # ---------------- queue behavior knobs ----------------
    QUEUE_GET_TIMEOUT_SEC: float = 1.0,   # periodic wakeup to allow robust exit

    # ---------------- deadline-aware worker exit ----------------
    deadline_ts: float | None = None,     # if set, worker stops ASAP when time>=deadline_ts

    # ---------------- ML instrumentation knobs ----------------
    ML_ENABLE_LOGGING: bool = True,       # whether to emit ML training rows in result_queue
    ML_SAMPLE_LOG_EVERY: int = 4000,      # worker logs one ML sample line every N processed edges (per worker)
    ML_MAX_SAMPLE_LOGS: int = 50,         # hard cap per worker for ML sample lines
    ML_LOG_PREFIX: str = "[FUNC1][ML]",   # prefix for ML instrumentation logs
):
    """
    Deadline-aware worker with robust queue get + rank-sorted "between" via rank_to_node slicing.

    IMPORTANT FIXES (vs many older versions):
      - Always casts snapshot ranks to int before using them as list indices.
      - Always emits exactly one payload per popped idx (including skips/errors).
      - Uses QUEUE_GET_TIMEOUT_SEC to periodically wake up and check deadline.
    """
    import time
    import queue as _pyqueue

    # -------------------- helpers --------------------
    def _deadline_hit() -> bool:
        if deadline_ts is None:
            return False
        try:
            return time.time() >= float(deadline_ts)
        except Exception:
            return False

    # Thresholds for profiling logs (seconds)
    BETWEEN_WARN = 0.05
    APPLY_WARN = 0.10
    BETWEEN_SAFETY_CHECK_LIMIT = 5
    between_checks_done = 0

    # -------------------- DEBUG helpers --------------------
    dbg_edge_logs = 0
    none_seen = 0

    def dbg_log(msg: str):
        nonlocal dbg_edge_logs
        if debug and dbg_edge_logs < int(DEBUG_MAX_EDGE_LOGS):
            dbg_edge_logs += 1
            log_message(f"[W{worker_idx:02d}][DBG] {msg}", log_path)

    # -------------------- ML helpers --------------------
    ml_sample_logs = 0

    # Cache per-node totals for cheap features
    _node_cache = {}  # node -> (out_w_sum, in_w_sum, out_deg, in_deg)

    def _node_stats(node):
        """Return (out_w_sum, in_w_sum, out_deg, in_deg) from out_edges/in_edges. Cheap, cached."""
        if node in _node_cache:
            return _node_cache[node]
        out_list = out_edges.get(node, []) if isinstance(out_edges, dict) else []
        in_list  = in_edges.get(node, []) if isinstance(in_edges, dict) else []
        out_w = 0.0
        in_w = 0.0
        try:
            for _nbr, _w in out_list:
                out_w += float(_w)
        except Exception:
            out_w = 0.0
        try:
            for _nbr, _w in in_list:
                in_w += float(_w)
        except Exception:
            in_w = 0.0
        out_deg = int(len(out_list)) if hasattr(out_list, "__len__") else 0
        in_deg  = int(len(in_list)) if hasattr(in_list, "__len__") else 0
        _node_cache[node] = (float(out_w), float(in_w), out_deg, in_deg)
        return _node_cache[node]

    def _ml_log_sample(line: str):
        nonlocal ml_sample_logs
        if not ML_ENABLE_LOGGING:
            return
        if int(ML_SAMPLE_LOG_EVERY) <= 0:
            return
        if ml_sample_logs >= int(ML_MAX_SAMPLE_LOGS):
            return
        ml_sample_logs += 1
        log_message(f"{ML_LOG_PREFIX}[W{worker_idx:02d}] {line}", log_path)

    # -------------------- SNAPSHOT / RANK ARRAY --------------------
    if not scores_snapshot:
        log_message("[FUNC1][W] RUNTIME ERROR: scores_snapshot is empty.", log_path)
        raise RuntimeError("Empty scores_snapshot.")

    # Enforce permutation ranks 0..n-1
    n_nodes = len(scores_snapshot)
    rank_to_node = [None] * n_nodes

    for node, r in scores_snapshot.items():
        try:
            rr = int(r)
        except Exception:
            log_message(f"[FUNC1][W] RUNTIME ERROR: non-integer rank: node={node}, rank={r}", log_path)
            raise RuntimeError("Non-integer rank in scores_snapshot.")

        if rr < 0 or rr >= n_nodes:
            log_message(f"[FUNC1][W] RUNTIME ERROR: rank out of range: node={node}, rank={rr}, n={n_nodes}", log_path)
            raise RuntimeError("Rank out of range in scores_snapshot.")

        if rank_to_node[rr] is not None:
            log_message(
                f"[FUNC1][W] RUNTIME ERROR: duplicate rank={rr}: nodes={rank_to_node[rr]} and {node}",
                log_path
            )
            raise RuntimeError("Non-injective ranks in scores_snapshot.")

        rank_to_node[rr] = node

    for rr, node in enumerate(rank_to_node):
        if node is None:
            log_message(f"[FUNC1][W] RUNTIME ERROR: missing node at rank={rr} (non-permutation ranks)", log_path)
            raise RuntimeError("scores_snapshot ranks are not a permutation 0..n-1.")

    worker_start = time.time()

    # Timers
    total_apply_strategy_time = 0.0
    total_greedy_time = 0.0
    total_queue_wait_time = 0.0
    total_edge_processing_time = 0.0

    # Basic counts
    edges_processed = 0
    successes = 0
    failures = 0
    extended_successes = 0
    greedy_successes = 0

    # -------------------- ML aggregate stats (per worker) --------------------
    ml_attempts = 0
    ml_successes = 0
    ml_sum_delta = 0.0
    ml_sum_gap = 0.0
    ml_sum_w = 0.0
    ml_sum_eval_ms = 0.0
    ml_mode_counts = {"extended": 0, "greedy": 0, "none": 0, "error": 0, "skip": 0}
    ml_top = []  # list of (delta, edge_idx, u, v, gap, w, mode)

    def _ml_update_top(delta, edge_idx, u, v, gap, w, mode, k=8):
        if delta <= 0:
            return
        ml_top.append((float(delta), int(edge_idx), u, v, int(gap), float(w), str(mode)))
        ml_top.sort(reverse=True, key=lambda x: x[0])
        if len(ml_top) > k:
            ml_top[:] = ml_top[:k]

    # Aliases (read-only)
    scores_base = scores_snapshot
    out_edges_base = out_edges
    in_edges_base = in_edges
    edges_array = backward_edges

    # ---- Signature detection (compute_all_gains_local) ----
    _gains_accepts_log_path = None

    def _compute_gains(scores_base_local, u0, v0, out_e, in_e, log_file_path):
        nonlocal _gains_accepts_log_path
        if _gains_accepts_log_path is None:
            try:
                compute_all_gains_local(scores_base_local, u0, v0, out_e, in_e, log_file_path)
                _gains_accepts_log_path = True
            except TypeError:
                _gains_accepts_log_path = False
            except Exception:
                _gains_accepts_log_path = True

        if _gains_accepts_log_path:
            return compute_all_gains_local(scores_base_local, u0, v0, out_e, in_e, log_file_path)
        else:
            return compute_all_gains_local(scores_base_local, u0, v0, out_e, in_e)

    # -------------------- DEBUG counters (per worker) --------------------
    ext_called = 0
    ext_exception = 0
    ext_success_pos = 0
    ext_success_nonpos = 0
    ext_fail = 0

    greedy_called = 0
    greedy_pos = 0
    greedy_nonpos = 0
    greedy_choice_swap = 0
    greedy_choice_mvvafu = 0
    greedy_choice_mvubfv = 0

    reject_changed_empty = 0
    skip_not_backward = 0
    unhandled_errors = 0
    bad_edge_tuple = 0
    bad_edge_index = 0
    rank_invariant_viol = 0
    interval_mismatch_warn = 0
    greedy_error = 0
    gains_error = 0
    ext_nonpos_reject = 0

    out_mode_none = 0
    out_mode_ext = 0
    out_mode_greedy = 0
    out_mode_error = 0

    sentinels_seen = 0
    get_timeouts = 0
    first_item_ts = None
    last_item_ts = None

    stopped_for_deadline = False

    if debug:
        dbg_log(
            f"START worker_idx={worker_idx} num_procs={num_procs} "
            f"QUEUE_GET_TIMEOUT_SEC={QUEUE_GET_TIMEOUT_SEC} deadline_ts={deadline_ts} "
            f"ML_ENABLE_LOGGING={ML_ENABLE_LOGGING}"
        )

    # Helper: (debug only) ensure ranks in a block are a permutation of the range
    def _check_block_is_permutation(scores_dict, lo_r, hi_r, where):
        seen = set()
        for rr in range(lo_r, hi_r + 1):
            n = rank_to_node[rr]
            if n is None:
                dbg_log(f"[{where}] block_perm FAIL: rank_to_node[{rr}] is None")
                return False
            new_r = scores_dict.get(n, rr)
            if new_r < lo_r or new_r > hi_r:
                dbg_log(f"[{where}] block_perm FAIL: node={n} new_r={new_r} outside [{lo_r},{hi_r}]")
                return False
            if new_r in seen:
                dbg_log(f"[{where}] block_perm FAIL: duplicate new_r={new_r} in [{lo_r},{hi_r}]")
                return False
            seen.add(new_r)
        return True

    while True:
        # -----------------------------------------------
        # HARD STOP: deadline
        # -----------------------------------------------
        if _deadline_hit():
            stopped_for_deadline = True
            dbg_log("DEADLINE_HIT -> exit loop")
            break

        # -----------------------------------------------
        # 1) POP EDGE INDEX FROM QUEUE (deadline-aware wakeups)
        # -----------------------------------------------
        t0 = time.time()
        try:
            tout = float(QUEUE_GET_TIMEOUT_SEC)
            if tout <= 0:
                tout = 1.0
            idx = edge_queue.get(timeout=tout)
        except _pyqueue.Empty:
            total_queue_wait_time += (time.time() - t0)
            get_timeouts += 1
            continue
        except Exception as e:
            total_queue_wait_time += (time.time() - t0)
            dbg_log(f"QUEUE_EXC err={repr(e)} -> break")
            break

        total_queue_wait_time += (time.time() - t0)

        if first_item_ts is None:
            first_item_ts = time.time()
        last_item_ts = time.time()

        # Sentinel
        if idx is None:
            sentinels_seen += 1
            break

        # If deadline hits after we popped work, stop *before* heavy processing
        if _deadline_hit():
            stopped_for_deadline = True
            dbg_log("DEADLINE_HIT_AFTER_GET -> stop before processing")
            break

        edge_start = time.time()
        edges_processed += 1

        # Defaults (included in payload even for early exits)
        u = v = None
        lo = hi = None
        w = 0.0
        mode = "none"
        success = False
        delta_bw_reduction = 0.0
        changed_scores = {}
        between = []
        t_between = 0.0
        t_apply_dur = 0.0
        greedy_dur = 0.0
        idx_u = None
        idx_v = None

        ml_row = None

        def _emit_payload(*, force_mode=None, force_success=None, force_delta=None, force_changed=None, err=None):
            nonlocal ml_row, ml_attempts, ml_successes, ml_sum_delta, ml_sum_gap, ml_sum_w, ml_sum_eval_ms
            nonlocal out_mode_none, out_mode_ext, out_mode_greedy, out_mode_error
            nonlocal failures, successes, extended_successes, greedy_successes

            now_mode = str(force_mode) if force_mode is not None else str(mode)
            is_real_success = bool(force_success) if force_success is not None else bool(success and delta_bw_reduction > 0.0 and bool(changed_scores))
            used_delta = float(force_delta) if force_delta is not None else float(delta_bw_reduction)
            used_changed = force_changed if force_changed is not None else (changed_scores if is_real_success else {})

            edge_total_time = time.time() - edge_start

            # Count success/failure exactly once per popped idx
            if is_real_success:
                successes += 1
                if now_mode.startswith("extended"):
                    extended_successes += 1
                elif now_mode.startswith("greedy"):
                    greedy_successes += 1
            else:
                failures += 1

            if now_mode.startswith("extended"):
                out_mode_ext += 1
            elif now_mode.startswith("greedy"):
                out_mode_greedy += 1
            elif now_mode.startswith("error"):
                out_mode_error += 1
            else:
                out_mode_none += 1

            # ML row (always produce if enabled)
            if ML_ENABLE_LOGGING:
                try:
                    ru = int(idx_u) if idx_u is not None else (int(scores_base.get(u, -1)) if u is not None else -1)
                    rv = int(idx_v) if idx_v is not None else (int(scores_base.get(v, -1)) if v is not None else -1)
                except Exception:
                    ru, rv = -1, -1

                gap = int(ru - rv) if (ru >= 0 and rv >= 0) else 0
                eval_ms = float(edge_total_time * 1000.0)

                out_u_w = in_u_w = out_v_w = in_v_w = 0.0
                out_u_deg = in_u_deg = out_v_deg = in_v_deg = 0
                if u is not None:
                    out_u_w, in_u_w, out_u_deg, in_u_deg = _node_stats(u)
                if v is not None:
                    out_v_w, in_v_w, out_v_deg, in_v_deg = _node_stats(v)

                ml_attempts += 1
                ml_sum_gap += float(gap)
                ml_sum_w += float(w)
                ml_sum_eval_ms += float(eval_ms)

                if now_mode.startswith("extended"):
                    ml_mode_counts["extended"] += 1
                elif now_mode.startswith("greedy"):
                    ml_mode_counts["greedy"] += 1
                elif now_mode.startswith("error"):
                    ml_mode_counts["error"] += 1
                elif now_mode.startswith("skip"):
                    ml_mode_counts["skip"] += 1
                else:
                    ml_mode_counts["none"] += 1

                if is_real_success:
                    ml_successes += 1
                    ml_sum_delta += float(max(0.0, used_delta))
                    _ml_update_top(max(0.0, used_delta), idx, u, v, gap, w, now_mode)

                ml_row = {
                    "edge_idx": int(idx) if idx is not None else -1,
                    "worker_idx": int(worker_idx),
                    "u": u,
                    "v": v,
                    "w_uv": float(w),
                    "rank_u": int(ru),
                    "rank_v": int(rv),
                    "gap": int(gap),
                    "interval_len": int(gap + 1) if gap > 0 else 0,
                    "between_len": int(len(between)) if between is not None else 0,
                    "out_u_w": float(out_u_w), "in_u_w": float(in_u_w),
                    "out_v_w": float(out_v_w), "in_v_w": float(in_v_w),
                    "out_u_deg": int(out_u_deg), "in_u_deg": int(in_u_deg),
                    "out_v_deg": int(out_v_deg), "in_v_deg": int(in_v_deg),
                    "mode": str(now_mode),
                    "success": (1 if is_real_success else 0),
                    "delta": float(used_delta if is_real_success else 0.0),
                    "eval_ms": float(eval_ms),
                    "t_apply": float(t_apply_dur),
                    "t_greedy": float(greedy_dur),
                    "t_between": float(t_between),
                    "stored_lo": lo,
                    "stored_hi": hi,
                    "ts": float(time.time()),
                }
                if err is not None:
                    ml_row["error"] = str(err)

                if ML_SAMPLE_LOG_EVERY and (edges_processed % max(1, int(ML_SAMPLE_LOG_EVERY)) == 0):
                    _ml_log_sample(
                        f"sample edge_idx={idx} u={u} v={v} w={float(w):.6g} "
                        f"ru={ru} rv={rv} gap={gap} between={len(between)} "
                        f"mode={now_mode} success={1 if is_real_success else 0} "
                        f"delta={float(used_delta if is_real_success else 0.0):.6g} eval_ms={eval_ms:.2f}"
                    )

            payload = {
                "worker_idx": worker_idx,
                "edge_idx": int(idx),
                "u": u,
                "v": v,
                "success": bool(is_real_success),
                "delta": float(used_delta),
                "changed_scores": used_changed if is_real_success else {},
                "mode": str(now_mode),
                "interval": (lo, hi),
                "timing": {
                    "apply_new_strategy": float(t_apply_dur),
                    "greedy_time": float(greedy_dur),
                    "between_list_time": float(t_between),
                    "edge_total_time": float(edge_total_time),
                },
                "debug_flags": {
                    "interval_mismatch_warn": bool(interval_mismatch_warn > 0),
                }
            }
            if err is not None:
                payload["error"] = str(err)
            if ML_ENABLE_LOGGING and (ml_row is not None):
                payload["ml_row"] = ml_row

            result_queue.put(payload)

        try:
            # Defensive: idx range
            if not (0 <= idx < len(edges_array)):
                bad_edge_index += 1
                dbg_log(f"BAD_IDX idx={idx} len(edges_array)={len(edges_array)} -> emit+skip")
                _emit_payload(force_mode="skip/bad_idx", force_success=False, force_delta=0.0, force_changed={}, err="bad_edge_index")
                continue

            tup = edges_array[idx]
            if not (isinstance(tup, (tuple, list)) and len(tup) >= 5):
                bad_edge_tuple += 1
                dbg_log(f"BAD_TUPLE idx={idx} tup={tup!r} -> emit+skip")
                _emit_payload(force_mode="skip/bad_tuple", force_success=False, force_delta=0.0, force_changed={}, err="bad_edge_tuple")
                continue

            # expected: u,v,w,lo,hi (lo/hi may be stale)
            u, v, w, lo, hi = tup[:5]
            w = float(w)

            # CRITICAL: ranks must be ints for slicing
            idx_u = int(scores_base[u])
            idx_v = int(scores_base[v])

            # Not backward anymore (snapshot mismatch / already fixed)
            if not (idx_u > idx_v):
                skip_not_backward += 1
                _emit_payload(force_mode="skip/not_backward", force_success=False, force_delta=0.0, force_changed={})
                continue

            # Optional warn: stored lo/hi mismatch snapshot ranks (does NOT break)
            if debug:
                try:
                    lo_i = int(lo)
                    hi_i = int(hi)
                    if not (lo_i == idx_v and hi_i == idx_u):
                        interval_mismatch_warn += 1
                        if interval_mismatch_warn <= 5:
                            dbg_log(
                                f"INTERVAL_MISMATCH u={u} v={v} stored(lo,hi)=({lo},{hi}) "
                                f"curr(idx_v,idx_u)=({idx_v},{idx_u})"
                            )
                except Exception:
                    pass

            # -----------------------------------------------
            # 2) Build "between" (rank-sorted)
            # -----------------------------------------------
            t_btw = time.time()
            if idx_u - idx_v > 1:
                between = rank_to_node[idx_v + 1: idx_u]  # already sorted by rank
            else:
                between = []
            t_between = time.time() - t_btw

            if t_between > BETWEEN_WARN:
                dbg_log(f"BETWEEN_SLOW u={u} v={v} gap={idx_u-idx_v} len={len(between)} dt={t_between:.6f}s")

            if debug and between_checks_done < BETWEEN_SAFETY_CHECK_LIMIT:
                between_checks_done += 1
                slow_between = [node for node, r in scores_base.items() if idx_v < int(r) < idx_u]
                if set(between) != set(slow_between):
                    log_message("[FUNC1][W] RUNTIME ERROR: Fast 'between' != slow method.", log_path)
                    raise RuntimeError("Fast 'between' implementation mismatch detected.")

            # Local working copy for this edge attempt
            scores = scores_base.copy()

            # -----------------------------------------------
            # 3) Extended strategy
            # -----------------------------------------------
            if _deadline_hit():
                stopped_for_deadline = True
                dbg_log("DEADLINE_HIT_BEFORE_EXT -> break")
                break

            ext_called += 1
            t_apply = time.time()
            try:
                ext_success, ext_bw_reduction, _rB, _rA, _dbg = apply_new_strategy(
                    scores, u, v, between,
                    out_edges_base, in_edges_base, edges_dict,
                    log_path,
                    debug=debug
                )
                success = bool(ext_success)
                delta_bw_reduction = float(ext_bw_reduction)
                mode = "extended"
            except Exception as e:
                ext_exception += 1
                dbg_log(f"EXT_EXC idx={idx} u={u} v={v} err={repr(e)}")
                success = False
                delta_bw_reduction = 0.0
                mode = "extended_error"

            t_apply_dur = time.time() - t_apply
            total_apply_strategy_time += t_apply_dur

            if t_apply_dur > APPLY_WARN:
                dbg_log(f"EXT_SLOW idx={idx} u={u} v={v} dt={t_apply_dur:.6f}s")

            if success:
                if delta_bw_reduction > 0.0:
                    ext_success_pos += 1
                else:
                    ext_success_nonpos += 1
                    ext_nonpos_reject += 1
                    dbg_log(f"EXT_SUCC_NONPOS idx={idx} u={u} v={v} delta={delta_bw_reduction:.6g} -> reject")
                    success = False
                    delta_bw_reduction = 0.0
                    mode = "extended_nonpositive_delta"
            else:
                ext_fail += 1

            # -----------------------------------------------
            # 4) Greedy fallback
            # -----------------------------------------------
            if not (success and delta_bw_reduction > 0.0):
                if _deadline_hit():
                    stopped_for_deadline = True
                    dbg_log("DEADLINE_HIT_BEFORE_GREEDY -> break")
                    break

                greedy_called += 1
                try:
                    t_g = time.time()
                    g1, g2, g3 = _compute_gains(scores_base, u, v, out_edges_base, in_edges_base, log_path)
                    greedy_dur = time.time() - t_g
                    total_greedy_time += greedy_dur
                except Exception as e:
                    gains_error += 1
                    greedy_error += 1
                    dbg_log(f"GAINS_EXC idx={idx} u={u} v={v} err={repr(e)}")
                    g1 = g2 = g3 = 0.0
                    greedy_dur = 0.0

                best_gain = max(float(g1), float(g2), float(g3))
                if best_gain > 0.0:
                    greedy_pos += 1
                    scores = scores_base.copy()
                    delta_bw_reduction = float(best_gain)
                    success = True

                    if best_gain == float(g1):
                        scores[u], scores[v] = idx_v, idx_u
                        mode = "greedy/swap"
                        greedy_choice_swap += 1
                    elif best_gain == float(g2):
                        for k, r in list(scores.items()):
                            rr = int(r)
                            if idx_v < rr <= idx_u:
                                scores[k] = rr - 1
                        scores[v] = idx_u
                        mode = "greedy/move_v_after_u"
                        greedy_choice_mvvafu += 1
                    else:
                        for k, r in list(scores.items()):
                            rr = int(r)
                            if idx_v <= rr < idx_u:
                                scores[k] = rr + 1
                        scores[u] = idx_v
                        mode = "greedy/move_u_before_v"
                        greedy_choice_mvubfv += 1

                    if debug and DEBUG_LOG_TOP_GAINS and delta_bw_reduction >= float(DEBUG_BIG_GAIN):
                        dbg_log(
                            f"GREEDY_BIG idx={idx} u={u} v={v} delta={delta_bw_reduction:.6g} "
                            f"(g1={float(g1):.3g},g2={float(g2):.3g},g3={float(g3):.3g}) mode={mode}"
                        )
                else:
                    greedy_nonpos += 1
                    success = False
                    delta_bw_reduction = 0.0
                    mode = "none" if mode != "extended_error" else "none_after_error"

                    if debug:
                        none_seen += 1
                        if (none_seen % max(1, int(DEBUG_SAMPLE_EVERY)) == 0) and (dbg_edge_logs < int(DEBUG_MAX_EDGE_LOGS)):
                            dbg_log(f"NONE_SAMPLE idx={idx} u={u} v={v} g1={float(g1):.6g} g2={float(g2):.6g} g3={float(g3):.6g} last_mode={mode}")

            # -----------------------------------------------
            # 5) Collect changed_scores on touched block [idx_v .. idx_u]
            # -----------------------------------------------
            if success and delta_bw_reduction > 0.0:
                lo_blk = int(idx_v)
                hi_blk = int(idx_u)

                lo_blk = max(0, lo_blk)
                hi_blk = min(n_nodes - 1, hi_blk)

                if hi_blk < lo_blk:
                    reject_changed_empty += 1
                    dbg_log(f"CHANGED_EMPTY_BAD_BLOCK idx={idx} u={u} v={v} lo_blk={lo_blk} hi_blk={hi_blk} -> reject")
                    success = False
                    delta_bw_reduction = 0.0
                    mode = f"{mode}_no_changes"
                else:
                    interval_nodes = rank_to_node[lo_blk:hi_blk + 1]
                    for node in interval_nodes:
                        r_old = int(scores_base[node])
                        r_new = int(scores.get(node, r_old))
                        if r_new != r_old:
                            changed_scores[node] = int(r_new)

                    if not changed_scores:
                        reject_changed_empty += 1
                        dbg_log(
                            f"CHANGED_EMPTY idx={idx} u={u} v={v} delta={delta_bw_reduction:.6g} mode={mode} "
                            f"block=[{lo_blk},{hi_blk}] gap={idx_u-idx_v} -> reject"
                        )
                        success = False
                        delta_bw_reduction = 0.0
                        mode = f"{mode}_no_changes"

            # -----------------------------------------------
            # 5b) Optional invariant check (debug, rare)
            # -----------------------------------------------
            if debug and success and delta_bw_reduction > 0.0:
                if (edges_processed % 5000) == 0 and (dbg_edge_logs < int(DEBUG_MAX_EDGE_LOGS)):
                    ok = _check_block_is_permutation(scores, int(idx_v), int(idx_u), where="AFTER_MOVE")
                    if not ok:
                        rank_invariant_viol += 1
                        dbg_log(f"RANK_INVARIANT_VIOL idx={idx} u={u} v={v} mode={mode} -> flagged")

            # -----------------------------------------------
            # 6) Final emit
            # -----------------------------------------------
            _emit_payload()

        except Exception as e:
            unhandled_errors += 1
            dbg_log(f"UNHANDLED idx={idx} u={u} v={v} err={repr(e)}")
            _emit_payload(force_mode="error", force_success=False, force_delta=0.0, force_changed={}, err=repr(e))
            continue

        finally:
            total_edge_processing_time += (time.time() - edge_start)

    # -----------------------------------------------
    # Worker summary (always send; main decides logging)
    # -----------------------------------------------
    runtime = time.time() - worker_start
    active_span = 0.0
    if first_item_ts is not None and last_item_ts is not None:
        active_span = max(0.0, last_item_ts - first_item_ts)

    # Always emit compact ML summary line (even if debug=False)
    if ML_ENABLE_LOGGING:
        try:
            att = max(1, int(ml_attempts))
            succ = int(ml_successes)
            succ_rate = float(succ) / float(att)
            avg_delta = (float(ml_sum_delta) / float(max(1, succ))) if succ > 0 else 0.0
            avg_gap = float(ml_sum_gap) / float(att)
            avg_w = float(ml_sum_w) / float(att)
            avg_eval_ms = float(ml_sum_eval_ms) / float(att)

            top_str = ""
            if ml_top:
                top3 = ml_top[:3]
                top_str = " top=" + "; ".join(
                    [f"(d={d:.3g},idx={ei},gap={g},w={ww:.3g},m={m})" for (d, ei, uu, vv, g, ww, m) in top3]
                )

            log_message(
                f"{ML_LOG_PREFIX}[W{worker_idx:02d}][SUM] attempts={ml_attempts} succ={ml_successes} "
                f"succ_rate={succ_rate:.4f} avg_delta_given_succ={avg_delta:.6g} "
                f"avg_gap={avg_gap:.2f} avg_w={avg_w:.3g} avg_eval_ms={avg_eval_ms:.2f} "
                f"modes(ext/greedy/none/skip/error)="
                f"{ml_mode_counts.get('extended',0)}/{ml_mode_counts.get('greedy',0)}/"
                f"{ml_mode_counts.get('none',0)}/{ml_mode_counts.get('skip',0)}/{ml_mode_counts.get('error',0)}"
                f"{top_str}",
                log_path
            )
        except Exception:
            pass

    if debug:
        log_message(
            f"[W{worker_idx:02d}][SUM] edges={edges_processed} succ={successes} fail={failures} "
            f"sentinels={sentinels_seen} timeouts={get_timeouts} active_span={active_span:.3f}s runtime={runtime:.3f}s "
            f"stopped_for_deadline={stopped_for_deadline} | "
            f"ext(called/exc/spos/snonpos/fail/nonposrej)={ext_called}/{ext_exception}/{ext_success_pos}/{ext_success_nonpos}/{ext_fail}/{ext_nonpos_reject} | "
            f"greedy(called/pos/nonpos swap/mvvafu/mvubfv)={greedy_called}/{greedy_pos}/{greedy_nonpos} "
            f"{greedy_choice_swap}/{greedy_choice_mvvafu}/{greedy_choice_mvubfv} | "
            f"rej_changed_empty={reject_changed_empty} skip_not_backward={skip_not_backward} "
            f"bad_idx={bad_edge_index} bad_tuple={bad_edge_tuple} inv_viol={rank_invariant_viol} interval_warn={interval_mismatch_warn} "
            f"gains_err={gains_error} unhandled={unhandled_errors} | "
            f"t_queue={total_queue_wait_time:.3f}s t_apply={total_apply_strategy_time:.3f}s "
            f"t_greedy={total_greedy_time:.3f}s t_edges={total_edge_processing_time:.3f}s dbg_edge_logs={dbg_edge_logs}",
            log_path
        )

    # Always send done + ML aggregates to parent so it can compute global metrics
    result_queue.put({
        "worker_idx": worker_idx,
        "done": True,
        "stopped_for_deadline": bool(stopped_for_deadline),
        "deadline_ts": (None if deadline_ts is None else float(deadline_ts)),
        "edges_processed": edges_processed,
        "successes": successes,
        "extended_successes": extended_successes,
        "greedy_successes": greedy_successes,
        "failures": failures,
        "runtime": float(runtime),
        "timing_totals": {
            "queue_wait": float(total_queue_wait_time),
            "apply_total": float(total_apply_strategy_time),
            "greedy_total": float(total_greedy_time),
            "edge_processing_total": float(total_edge_processing_time),
        },
        "ml_stats": {
            "attempts": int(ml_attempts),
            "successes": int(ml_successes),
            "sum_delta": float(ml_sum_delta),
            "sum_gap": float(ml_sum_gap),
            "sum_w": float(ml_sum_w),
            "sum_eval_ms": float(ml_sum_eval_ms),
            "mode_counts": dict(ml_mode_counts),
            "top": list(ml_top),
            "ml_sample_logs": int(ml_sample_logs),
        },
        "debug_stats": {
            "ext_called": ext_called,
            "ext_exception": ext_exception,
            "ext_success_pos": ext_success_pos,
            "ext_success_nonpos": ext_success_nonpos,
            "ext_fail": ext_fail,
            "ext_nonpos_reject": ext_nonpos_reject,

            "greedy_called": greedy_called,
            "greedy_pos": greedy_pos,
            "greedy_nonpos": greedy_nonpos,
            "greedy_choice_swap": greedy_choice_swap,
            "greedy_choice_mvvafu": greedy_choice_mvvafu,
            "greedy_choice_mvubfv": greedy_choice_mvubfv,

            "reject_changed_empty": reject_changed_empty,
            "skip_not_backward": skip_not_backward,
            "bad_edge_index": bad_edge_index,
            "bad_edge_tuple": bad_edge_tuple,
            "rank_invariant_viol": rank_invariant_viol,
            "interval_mismatch_warn": interval_mismatch_warn,
            "gains_error": gains_error,
            "greedy_error": greedy_error,
            "unhandled_errors": unhandled_errors,

            "out_mode_none": out_mode_none,
            "out_mode_ext": out_mode_ext,
            "out_mode_greedy": out_mode_greedy,
            "out_mode_error": out_mode_error,

            "dbg_edge_logs": dbg_edge_logs,
            "queue_timeouts": get_timeouts,
            "sentinels_seen": sentinels_seen,
            "active_span": float(active_span),
        },
    })


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
    LOG_EVERY_ROUND: int = 0,            # 0 => no periodic progress logs (quiet)
    BW_CHECK_EVERY_ROUND: int = 400,
    edge_subset=None,

    # -------- control --------
    MAX_DP_BATCHES_PER_CALL: int | None = None,
    NO_IMPROVEMENT_TIME_LIMIT_SEC: float = 600.0,
    EPS_IMPROVE: float = 1e-12,

    # -------- safety / debug (computational checks, not logging) --------
    debug: bool = True,
    BW_INCREASE_TOL: float = 1e-9,
    DEBUG_RECOMP_ACCEPTS_PER_BATCH: int = 1,

    # -------- hard global deadline (wall clock) --------
    deadline_ts: float | None = None,

    # -------- robust result collection --------
    RESULT_GET_TIMEOUT_SEC: float = 5.0,

    # -------- keep workers quiet --------
    WORKER_DEBUG_LOGS: bool = False,     # True only if you want per-worker START/SUM logs
    WORKER_DEBUG_SAMPLE_EVERY: int = 200,
    WORKER_DEBUG_MAX_EDGE_LOGS: int = 40,
    WORKER_DEBUG_BIG_GAIN: float = 1e-3,

    # -------- optional heartbeat --------
    HEARTBEAT_SEC: float | None = None,  # None => off

    # -------- performance knobs (safe defaults) --------
    MAX_PROCS_PER_BATCH: int = 12,       # cap processes per batch (prevents overhead explosion)
    SERIAL_BELOW_EDGES: int = 6,         # run single-process when batch is tiny
    JOIN_TIMEOUT_SEC: float = 3.0,       # join time before terminate
    TERMINATE_TIMEOUT_SEC: float = 3.0,  # after terminate, join this long

    # -------- collect behavior when deadline hits --------
    TERMINATE_ON_DEADLINE_DURING_COLLECT: bool = True,

    # ==========================================================
    # ML instrumentation + prioritization (NEW)
    # ==========================================================
    ML_WRITE_DATASET: bool = True,                 # write per-edge attempt rows to CSV (single-writer in parent)
    ML_DATASET_PATH: str | None = None,            # if None, defaults next to output_excel
    ML_FLUSH_EVERY_ROWS: int = 5000,               # flush training CSV every N rows
    ML_RUN_TAG: str = "baseline",                  # "baseline" or "ml" (useful for paper plots)

    # Prioritize which backward edges go into DP each batch (cheap heuristic now; later you can plug a model)
    ML_PRIORITIZE_FOR_DP: bool = True,
    ML_SCORER: str = "heuristic",                  # "heuristic" for now
    ML_MAX_EDGES_FOR_DP: int | None = 20000,       # cap edges_for_dp before DP selection (speed + focus)
    ML_EXPLORATION_FRAC: float = 0.15,             # keep some random edges to avoid blind spots
    ML_LOG_EVERY_BATCH: int = 50,                  # log ML stats every N batches (in addition to IMPROVE lines)

    # Pass-through to workers (so you can toggle without editing worker_loop)
    WORKER_QUEUE_GET_TIMEOUT_SEC: float = 1.0,
    WORKER_ML_SAMPLE_LOG_EVERY: int = 4000,
    WORKER_ML_MAX_SAMPLE_LOGS: int = 50,
    WORKER_ML_LOG_PREFIX: str = "[FUNC1][ML]",
):
    """
    UPDATED refine_ranking_parallel_dynamic with ML-ready logging + candidate prioritization:

    A) ML dataset writing:
       - Collects `ml_row` from workers (parallel path) and writes to a single CSV from the parent.
       - Also emits analogous rows in SERIAL path so your dataset is consistent.
       - Adds metadata columns: run_tag, graph, epoch, round, batch, path(serial/parallel), ts_parent.

    B) ML prioritization for DP (first practical step):
       - Before DP, optionally scores each backward edge candidate using a cheap heuristic,
         keeps top-K + exploration, then runs DP on that reduced set.

    C) Evidence logs for “ML helped”:
       - Per-batch: attempts, applies, success_rate, sum_delta, delta_per_sec.
       - Aggregates worker ml_stats (attempts/successes/sum_delta/sum_eval_ms/mode_counts).

    Returns:
      (best_scores, final_bw, remaining_pairs_unassessed)
    """
    import time
    import os
    import csv as _csv
    import random
    import multiprocessing as mp
    import queue as _pyqueue

    # ------------------------------
    # CPU count
    # ------------------------------
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

    # ------------------------------
    # Deadlines: MAX_HOURS AND deadline_ts (slice)
    # ------------------------------
    start_ts = time.time()
    deadline = start_ts + float(MAX_HOURS) * 3600.0
    if deadline_ts is not None:
        try:
            deadline = min(deadline, float(deadline_ts))
        except Exception:
            pass

    def _time_left():
        return max(0.0, deadline - time.time())

    def _deadline_hit():
        return time.time() >= deadline

    # Clamp NO_IMPROVEMENT_TIME_LIMIT_SEC to the slice size
    slice_sec = max(0.0, deadline - start_ts)
    eff_no_improve_sec = float(NO_IMPROVEMENT_TIME_LIMIT_SEC)
    if slice_sec > 0:
        eff_no_improve_sec = max(1.0, min(float(NO_IMPROVEMENT_TIME_LIMIT_SEC), float(slice_sec)))

    # ------------------------------
    # Build adjacency & edges_dict
    # ------------------------------
    out_edges = {}
    in_edges = {}
    edges_dict = {}
    parallel_overwrites = 0

    for (u, v, w) in edges:
        ww = float(w)
        out_edges.setdefault(u, []).append((v, ww))
        in_edges.setdefault(v, []).append((u, ww))
        if (u, v) in edges_dict:
            parallel_overwrites += 1
        edges_dict[(u, v)] = ww

    if parallel_overwrites > 0:
        log_message(f"[FUNC1][WARN] edges_dict overwrote {parallel_overwrites} parallel arcs (u,v). Consider aggregating.", log_path)

    # ------------------------------
    # ML dataset writer (single-writer in parent)
    # ------------------------------
    graph_name = os.path.basename(str(csv_path)).replace(".d", "").replace(".csv", "")
    if ML_DATASET_PATH is None:
        try:
            base_dir = os.path.dirname(str(output_excel))
            ML_DATASET_PATH = os.path.join(base_dir, f"{graph_name}_func1_ml_dataset.csv")
        except Exception:
            ML_DATASET_PATH = f"{graph_name}_func1_ml_dataset.csv"

    ml_fh = None
    ml_writer = None
    ml_header_written = False
    ml_rows_written = 0

    _node_cache = {}  # node -> (out_w_sum, in_w_sum, out_deg, in_deg)

    def _node_stats(node):
        if node in _node_cache:
            return _node_cache[node]
        out_list = out_edges.get(node, [])
        in_list = in_edges.get(node, [])
        out_w = 0.0
        in_w = 0.0
        try:
            for _nbr, _w in out_list:
                out_w += float(_w)
        except Exception:
            out_w = 0.0
        try:
            for _nbr, _w in in_list:
                in_w += float(_w)
        except Exception:
            in_w = 0.0
        out_deg = int(len(out_list)) if hasattr(out_list, "__len__") else 0
        in_deg = int(len(in_list)) if hasattr(in_list, "__len__") else 0
        _node_cache[node] = (float(out_w), float(in_w), out_deg, in_deg)
        return _node_cache[node]

    def _ml_open_writer():
        nonlocal ml_fh, ml_writer, ml_header_written
        if (not ML_WRITE_DATASET) or (ml_fh is not None):
            return
        try:
            d = os.path.dirname(str(ML_DATASET_PATH))
            if d:
                os.makedirs(d, exist_ok=True)
        except Exception:
            pass
        ml_fh = open(ML_DATASET_PATH, "a", newline="")
        ml_writer = None
        ml_header_written = False

    def _ml_write_row(row_dict):
        nonlocal ml_writer, ml_header_written, ml_rows_written
        if not ML_WRITE_DATASET:
            return
        if row_dict is None or ml_fh is None:
            return

        if ml_writer is None:
            fieldnames = list(row_dict.keys())
            ml_writer = _csv.DictWriter(ml_fh, fieldnames=fieldnames, extrasaction="ignore")
            try:
                if ml_fh.tell() == 0:
                    ml_writer.writeheader()
                    ml_header_written = True
            except Exception:
                if not ml_header_written:
                    ml_writer.writeheader()
                    ml_header_written = True

        ml_writer.writerow(row_dict)
        ml_rows_written += 1
        if ML_FLUSH_EVERY_ROWS and (ml_rows_written % max(1, int(ML_FLUSH_EVERY_ROWS)) == 0):
            try:
                ml_fh.flush()
            except Exception:
                pass

    # ------------------------------
    # Normalize edge_subset → subset_pairs (u,v)
    # ------------------------------
    subset_pairs = None
    if edge_subset is not None:
        norm_set = set()
        for e in edge_subset:
            if isinstance(e, tuple) and len(e) >= 2:
                norm_set.add((e[0], e[1]))
        subset_pairs = norm_set

    assessed_pairs_in_this_call = set()

    # ------------------------------
    # Fixed candidate pool for whole call
    # ------------------------------
    candidate_edges = []
    for (u, v, w) in edges:
        if subset_pairs is not None and (u, v) not in subset_pairs:
            continue
        candidate_edges.append((u, v, float(w)))

    if subset_pairs is not None and len(candidate_edges) == 0:
        log_message("[FUNC1][ERROR] edge_subset matches 0 edges in graph.", log_path)
        raise RuntimeError("edge_subset does not match any edges in this graph.")

    # ------------------------------
    # Helpers for S
    # ------------------------------
    def rebuild_S_from_scores(curr_scores):
        return [i for i, (u, v, _w) in enumerate(candidate_edges) if curr_scores[u] > curr_scores[v]]

    def filter_S_keep_backward(curr_scores, S_indices):
        out = []
        for gi in S_indices:
            u, v, _w = candidate_edges[gi]
            if curr_scores[u] > curr_scores[v]:
                out.append(gi)
        return out

    # ------------------------------
    # Worker cleanup helper
    # ------------------------------
    def _cleanup_workers(workers, reason_tag: str):
        terminated = 0
        for p in workers:
            try:
                p.join(timeout=JOIN_TIMEOUT_SEC)
            except Exception:
                pass
        for p in workers:
            try:
                if p.is_alive():
                    p.terminate()
                    terminated += 1
            except Exception:
                pass
        for p in workers:
            try:
                p.join(timeout=TERMINATE_TIMEOUT_SEC)
            except Exception:
                pass
        if terminated > 0:
            log_message(f"[FUNC1][WARN] terminated_workers={terminated} reason={reason_tag}", log_path)
        return terminated

    # ------------------------------
    # Crash packet helper
    # ------------------------------
    def _summarize_changed_scores(changed_scores, base_scores, max_items=12):
        if not changed_scores:
            return "changed_scores=EMPTY"
        items = list(changed_scores.items())
        items.sort(key=lambda kv: base_scores.get(kv[0], 10**18))
        n = len(items)
        old_ranks = [base_scores.get(node, None) for node, _ in items]
        new_ranks = [new_r for _, new_r in items]

        def _minmax(arr):
            arr2 = [a for a in arr if a is not None]
            if not arr2:
                return ("?", "?")
            return (min(arr2), max(arr2))

        old_mm = _minmax(old_ranks)
        new_mm = (min(new_ranks), max(new_ranks)) if new_ranks else ("?", "?")
        head = items[:max_items]
        head_str = ",".join([f"{node}:{base_scores.get(node,'?')}→{nr}" for node, nr in head])
        if n > max_items:
            head_str += ",..."
        return f"changed_scores(n={n}) old[min,max]={old_mm} new[min,max]={new_mm} head=[{head_str}]"

    def _log_crash_packet(tag, *, batch, round_idx, epoch_id, BW_tracked, best_BW,
                          s_len, num_edges_batch, extra_lines):
        log_message(
            f"[FUNC1][CRASH] {tag} batch={batch} round={round_idx} epoch={epoch_id} "
            f"BW_tracked={BW_tracked:.6f} best={best_BW:.6f} S={s_len} batch_edges={num_edges_batch}",
            log_path
        )
        for line in extra_lines:
            log_message(f"[FUNC1][CRASH] {line}", log_path)

    # ------------------------------
    # ML scorer for candidate prioritization (cheap heuristic now)
    # ------------------------------
    def _ml_score_edge(u, v, w, scores_snapshot):
        try:
            ru = int(scores_snapshot[u])
            rv = int(scores_snapshot[v])
        except Exception:
            return 0.0
        gap = ru - rv
        if gap <= 0:
            return -1e18

        out_u_w, in_u_w, out_u_deg, in_u_deg = _node_stats(u)
        out_v_w, in_v_w, out_v_deg, in_v_deg = _node_stats(v)

        score = 0.0
        score += float(w) * (1.0 + 0.01 * float(gap))
        score += 0.0005 * (float(out_u_w) - float(in_u_w))
        score += 0.0005 * (float(in_v_w) - float(out_v_w))
        score += 0.0001 * (float(out_u_deg) - float(in_u_deg))
        score += 0.0001 * (float(in_v_deg) - float(out_v_deg))
        return float(score)

    # ------------------------------
    # State
    # ------------------------------
    best_scores = scores.copy()
    best_BW = float(bw0)
    BW_tracked = float(bw0)

    last_improvement_time = start_ts
    last_heartbeat_time = start_ts

    BW_MISMATCH_TOL = 1e-6
    round_idx = 0
    dp_batches_done = 0
    S_indices = []
    epoch_id = 0

    try:
        n_nodes = len(scores)
    except Exception:
        n_nodes = len(node_to_index)

    # ------------------------------
    # START log
    # ------------------------------
    log_message(
        f"[FUNC1] START pid={pid_main} procs={num_procs} | BW={bw0:.2f}/{total_weight:.2f} "
        f"| MAX_HOURS={MAX_HOURS} NO_IMPROVE_SEC={NO_IMPROVEMENT_TIME_LIMIT_SEC} eff_no_improve={eff_no_improve_sec:.1f} "
        f"| deadline_ts={'None' if deadline_ts is None else int(deadline_ts)} "
        f"| MAX_PROCS_PER_BATCH={MAX_PROCS_PER_BATCH} SERIAL_BELOW_EDGES={SERIAL_BELOW_EDGES} "
        f"| ML(run_tag={ML_RUN_TAG} write={ML_WRITE_DATASET} path={ML_DATASET_PATH} "
        f"prioritize={ML_PRIORITIZE_FOR_DP} scorer={ML_SCORER} max_for_dp={ML_MAX_EDGES_FOR_DP} explore={ML_EXPLORATION_FRAC})",
        log_path
    )

    if ML_WRITE_DATASET:
        _ml_open_writer()

    # ===================================================
    # MAIN LOOP
    # ===================================================
    while True:
        now = time.time()

        if now >= deadline:
            log_message("[FUNC1] STOP deadline/time_limit", log_path)
            break

        if (MAX_DP_BATCHES_PER_CALL is not None) and (dp_batches_done >= MAX_DP_BATCHES_PER_CALL):
            log_message(f"[FUNC1] STOP max_dp_batches={MAX_DP_BATCHES_PER_CALL}", log_path)
            break

        if HEARTBEAT_SEC is not None and (now - last_heartbeat_time) >= float(HEARTBEAT_SEC):
            since_imp = now - last_improvement_time
            log_message(
                f"[FUNC1] HEARTBEAT BW={BW_tracked:.2f} best={best_BW:.2f} since_improve={since_imp:.1f}s "
                f"S={len(S_indices)} left={_time_left():.1f}s ml_rows={ml_rows_written}",
                log_path
            )
            last_heartbeat_time = now

        eff_no_improve_now = max(1.0, min(eff_no_improve_sec, max(1.0, (deadline - time.time()))))

        # Refresh epoch if S is empty
        if not S_indices:
            epoch_id += 1
            S_indices = rebuild_S_from_scores(scores)
            if not S_indices:
                log_message(f"[FUNC1] STOP no_backward_edges epoch={epoch_id}", log_path)
                break

        round_idx += 1

        if LOG_EVERY_ROUND and (round_idx % LOG_EVERY_ROUND == 0):
            elapsed_h = (now - start_ts) / 3600.0
            since_improvement = now - last_improvement_time
            log_message(
                f"[FUNC1] progress epoch={epoch_id} round={round_idx} batches={dp_batches_done} "
                f"| BW={BW_tracked:.2f} best={best_BW:.2f} | S={len(S_indices)} "
                f"| elapsed={elapsed_h:.2f}h since_improve={since_improvement:.1f}s left={_time_left():.1f}s ml_rows={ml_rows_written}",
                log_path
            )

        # Snapshot for this batch
        scores_snapshot = scores.copy()

        # Keep only backward edges under snapshot
        S_indices = filter_S_keep_backward(scores_snapshot, S_indices)
        if not S_indices:
            continue

        # Build DP list from current S (store gi for bookkeeping)
        edges_for_dp = []
        for gi in S_indices:
            u, v, w = candidate_edges[gi]
            ru = scores_snapshot[u]
            rv = scores_snapshot[v]
            if ru > rv:
                edges_for_dp.append((u, v, w, rv, ru, gi))  # (u,v,w,lo,hi,gi)

        if not edges_for_dp:
            S_indices = []
            continue

        # -------------------------------------------------------
        # ML prioritization BEFORE DP (reduce edges_for_dp)
        # -------------------------------------------------------
        if ML_PRIORITIZE_FOR_DP and (ML_MAX_EDGES_FOR_DP is not None) and (len(edges_for_dp) > int(ML_MAX_EDGES_FOR_DP)):
            K = int(ML_MAX_EDGES_FOR_DP)
            explore = max(0, min(K, int(float(K) * float(ML_EXPLORATION_FRAC))))
            topk = max(0, K - explore)

            scored = []
            for (u, v, w, lo, hi, gi) in edges_for_dp:
                s = _ml_score_edge(u, v, w, scores_snapshot) if ML_SCORER == "heuristic" else _ml_score_edge(u, v, w, scores_snapshot)
                scored.append((s, u, v, w, lo, hi, gi))

            scored.sort(reverse=True, key=lambda t: t[0])
            keep = scored[:topk]
            if explore > 0 and len(scored) > topk:
                rest = scored[topk:]
                keep.extend(random.sample(rest, k=min(explore, len(rest))))

            edges_for_dp = [(u, v, w, lo, hi, gi) for (_s, u, v, w, lo, hi, gi) in keep]

            if ML_LOG_EVERY_BATCH and ((dp_batches_done + 1) % max(1, int(ML_LOG_EVERY_BATCH)) == 0):
                log_message(
                    f"[FUNC1][ML] preDP_prune batch={dp_batches_done+1} kept={len(edges_for_dp)} from={len(scored)} "
                    f"topk={topk} explore={explore} run_tag={ML_RUN_TAG}",
                    log_path
                )

        # DP selector expects (u,v,w,lo,hi)
        edges_for_dp_plain = [(u, v, w, lo, hi) for (u, v, w, lo, hi, _gi) in edges_for_dp]
        local_selected = select_nonconflicting_edge_indices_dp(edges_for_dp_plain)
        if not local_selected:
            S_indices = []
            continue

        selected_global_indices = []
        batch_edges = []
        for li in local_selected:
            u, v, w, lo, hi, gi = edges_for_dp[li]
            selected_global_indices.append(int(gi))
            batch_edges.append((u, v, w, lo, hi))
        num_edges_batch = len(batch_edges)

        # Mark these pairs as assessed for hybrid accounting
        for gi in selected_global_indices:
            uu, vv, _ww = candidate_edges[gi]
            assessed_pairs_in_this_call.add((uu, vv))

        dp_batches_done += 1

        # ------------------------------------------
        # SERIAL fast path
        # ------------------------------------------
        if num_edges_batch <= int(SERIAL_BELOW_EDGES):
            batch_start = time.time()
            applied_edges_round = 0
            total_bw_reduct_round = 0.0
            recompute_left = int(DEBUG_RECOMP_ACCEPTS_PER_BATCH) if debug else 0

            rank_to_node = [None] * n_nodes
            for node, r in scores_snapshot.items():
                rr = int(r)
                if 0 <= rr < n_nodes:
                    rank_to_node[rr] = node

            serial_attempts = 0
            serial_succ = 0
            serial_sum_delta = 0.0

            for i in range(num_edges_batch):
                if _deadline_hit():
                    log_message(f"[FUNC1] STOP deadline hit inside SERIAL batch={dp_batches_done}", log_path)
                    break

                u, v, w, lo, hi = batch_edges[i]
                if scores_snapshot[u] <= scores_snapshot[v]:
                    if ML_WRITE_DATASET:
                        ru = int(scores_snapshot.get(u, -1))
                        rv = int(scores_snapshot.get(v, -1))
                        out_u_w, in_u_w, out_u_deg, in_u_deg = _node_stats(u)
                        out_v_w, in_v_w, out_v_deg, in_v_deg = _node_stats(v)
                        _ml_write_row({
                            "run_tag": ML_RUN_TAG, "graph": graph_name, "path": "serial",
                            "epoch": epoch_id, "round": round_idx, "batch": dp_batches_done,
                            "edge_local_i": i, "edge_idx_global": -1,
                            "u": u, "v": v, "w_uv": float(w),
                            "rank_u": ru, "rank_v": rv, "gap": int(ru - rv),
                            "between_len": 0,
                            "out_u_w": out_u_w, "in_u_w": in_u_w, "out_u_deg": out_u_deg, "in_u_deg": in_u_deg,
                            "out_v_w": out_v_w, "in_v_w": in_v_w, "out_v_deg": out_v_deg, "in_v_deg": in_v_deg,
                            "mode": "skip_not_backward", "success": 0, "delta": 0.0, "eval_ms": 0.0,
                            "ts_parent": float(time.time()),
                        })
                    continue

                serial_attempts += 1
                t_edge0 = time.time()

                scores_local = scores_snapshot.copy()
                idx_u = int(scores_snapshot[u])
                idx_v = int(scores_snapshot[v])

                between = [node for node in rank_to_node[idx_v + 1: idx_u] if node is not None] if (idx_v + 1 < idx_u) else []

                success = False
                delta = 0.0
                mode = "none"
                changed_scores = {}

                # extended
                try:
                    ext_success, ext_bw_reduction, _rB, _rA, _dbg = apply_new_strategy(
                        scores_local, u, v, between,
                        out_edges, in_edges, edges_dict,
                        log_path,
                        debug=False
                    )
                    if ext_success and float(ext_bw_reduction) > 0.0:
                        success = True
                        delta = float(ext_bw_reduction)
                        mode = "extended"
                except Exception:
                    success = False
                    delta = 0.0
                    mode = "extended_error"

                # greedy fallback
                if not success:
                    try:
                        try:
                            g1, g2, g3 = compute_all_gains_local(scores_snapshot, u, v, out_edges, in_edges, log_path)
                        except TypeError:
                            g1, g2, g3 = compute_all_gains_local(scores_snapshot, u, v, out_edges, in_edges)
                        best_gain = max(float(g1), float(g2), float(g3))
                        if best_gain > 0.0:
                            success = True
                            delta = best_gain
                            scores_local = scores_snapshot.copy()
                            if best_gain == float(g1):
                                scores_local[u], scores_local[v] = idx_v, idx_u
                                mode = "greedy/swap"
                            elif best_gain == float(g2):
                                for k, r in list(scores_local.items()):
                                    if idx_v < r <= idx_u:
                                        scores_local[k] = r - 1
                                scores_local[v] = idx_u
                                mode = "greedy/move_v_after_u"
                            else:
                                for k, r in list(scores_local.items()):
                                    if idx_v <= r < idx_u:
                                        scores_local[k] = r + 1
                                scores_local[u] = idx_v
                                mode = "greedy/move_u_before_v"
                    except Exception:
                        success = False
                        delta = 0.0
                        mode = "greedy_error"

                if not (success and delta > EPS_IMPROVE):
                    if ML_WRITE_DATASET:
                        out_u_w, in_u_w, out_u_deg, in_u_deg = _node_stats(u)
                        out_v_w, in_v_w, out_v_deg, in_v_deg = _node_stats(v)
                        eval_ms = float((time.time() - t_edge0) * 1000.0)
                        _ml_write_row({
                            "run_tag": ML_RUN_TAG, "graph": graph_name, "path": "serial",
                            "epoch": epoch_id, "round": round_idx, "batch": dp_batches_done,
                            "edge_local_i": i, "edge_idx_global": -1,
                            "u": u, "v": v, "w_uv": float(w),
                            "rank_u": idx_u, "rank_v": idx_v, "gap": int(idx_u - idx_v),
                            "between_len": int(len(between)),
                            "out_u_w": out_u_w, "in_u_w": in_u_w, "out_u_deg": out_u_deg, "in_u_deg": in_u_deg,
                            "out_v_w": out_v_w, "in_v_w": in_v_w, "out_v_deg": out_v_deg, "in_v_deg": in_v_deg,
                            "mode": str(mode), "success": 0, "delta": 0.0, "eval_ms": eval_ms,
                            "ts_parent": float(time.time()),
                        })
                    continue

                # collect changed_scores within [lo,hi]
                lo_i = int(lo) if lo is not None else 0
                hi_i = int(hi) if hi is not None else (n_nodes - 1)
                lo_i = max(0, lo_i)
                hi_i = min(n_nodes - 1, hi_i)
                if lo_i > hi_i:
                    continue

                for rr in range(lo_i, hi_i + 1):
                    node = rank_to_node[rr]
                    if node is None:
                        continue
                    r_new = scores_local.get(node, rr)
                    if r_new != rr:
                        changed_scores[node] = int(r_new)

                if not changed_scores:
                    if ML_WRITE_DATASET:
                        out_u_w, in_u_w, out_u_deg, in_u_deg = _node_stats(u)
                        out_v_w, in_v_w, out_v_deg, in_v_deg = _node_stats(v)
                        eval_ms = float((time.time() - t_edge0) * 1000.0)
                        _ml_write_row({
                            "run_tag": ML_RUN_TAG, "graph": graph_name, "path": "serial",
                            "epoch": epoch_id, "round": round_idx, "batch": dp_batches_done,
                            "edge_local_i": i, "edge_idx_global": -1,
                            "u": u, "v": v, "w_uv": float(w),
                            "rank_u": idx_u, "rank_v": idx_v, "gap": int(idx_u - idx_v),
                            "between_len": int(len(between)),
                            "out_u_w": out_u_w, "in_u_w": in_u_w, "out_u_deg": out_u_deg, "in_u_deg": in_u_deg,
                            "out_v_w": out_v_w, "in_v_w": in_v_w, "out_v_deg": out_v_deg, "in_v_deg": in_v_deg,
                            "mode": f"{mode}_no_changes", "success": 0, "delta": 0.0, "eval_ms": eval_ms,
                            "ts_parent": float(time.time()),
                        })
                    continue

                # optional recompute safety
                if recompute_left > 0:
                    bw_before = total_weight - float(compute_forward_weight(edges, scores))
                    scores_test = scores.copy()
                    for node, new_r in changed_scores.items():
                        scores_test[node] = new_r
                    bw_after = total_weight - float(compute_forward_weight(edges, scores_test))
                    if bw_after > bw_before + BW_INCREASE_TOL:
                        _log_crash_packet(
                            "BW_INCREASE_RECOMP_SERIAL",
                            batch=dp_batches_done,
                            round_idx=round_idx,
                            epoch_id=epoch_id,
                            BW_tracked=BW_tracked,
                            best_BW=best_BW,
                            s_len=len(S_indices),
                            num_edges_batch=num_edges_batch,
                            extra_lines=[
                                f"path=SERIAL i={i}/{num_edges_batch} mode={mode} u={u} v={v} w={w} idx_u={idx_u} idx_v={idx_v} stored(lo,hi)=({lo},{hi}) clamp(lo,hi)=({lo_i},{hi_i})",
                                f"bw_before={bw_before:.6f} bw_after={bw_after:.6f} delta_reported={delta:.6g} bw_delta_recomp={bw_after-bw_before:.6g}",
                                _summarize_changed_scores(changed_scores, scores),
                                f"between_len={len(between)} between_sorted=YES (rank_to_node slice)",
                            ]
                        )
                        raise RuntimeError("Backward weight increased after applying an accepted change (SERIAL, recomputed).")
                    recompute_left -= 1

                bw_before_tracked = BW_tracked
                for node, new_r in changed_scores.items():
                    scores[node] = new_r
                BW_tracked -= delta

                if BW_tracked > bw_before_tracked + BW_INCREASE_TOL:
                    _log_crash_packet(
                        "BW_TRACKED_INCREASE_SERIAL",
                        batch=dp_batches_done,
                        round_idx=round_idx,
                        epoch_id=epoch_id,
                        BW_tracked=BW_tracked,
                        best_BW=best_BW,
                        s_len=len(S_indices),
                        num_edges_batch=num_edges_batch,
                        extra_lines=[
                            f"path=SERIAL i={i}/{num_edges_batch} mode={mode} u={u} v={v} w={w} idx_u={idx_u} idx_v={idx_v} delta_reported={delta:.6g}",
                            f"BW_tracked_before={bw_before_tracked:.6f} BW_tracked_after={BW_tracked:.6f} increase={BW_tracked-bw_before_tracked:.6g}",
                            _summarize_changed_scores(changed_scores, scores_snapshot),
                        ]
                    )
                    raise RuntimeError("BW_tracked increased after applying an accepted change (SERIAL, forbidden).")

                applied_edges_round += 1
                total_bw_reduct_round += delta
                last_improvement_time = time.time()

                serial_succ += 1
                serial_sum_delta += float(delta)

                if ML_WRITE_DATASET:
                    out_u_w, in_u_w, out_u_deg, in_u_deg = _node_stats(u)
                    out_v_w, in_v_w, out_v_deg, in_v_deg = _node_stats(v)
                    eval_ms = float((time.time() - t_edge0) * 1000.0)
                    _ml_write_row({
                        "run_tag": ML_RUN_TAG, "graph": graph_name, "path": "serial",
                        "epoch": epoch_id, "round": round_idx, "batch": dp_batches_done,
                        "edge_local_i": i, "edge_idx_global": -1,
                        "u": u, "v": v, "w_uv": float(w),
                        "rank_u": idx_u, "rank_v": idx_v, "gap": int(idx_u - idx_v),
                        "between_len": int(len(between)),
                        "out_u_w": out_u_w, "in_u_w": in_u_w, "out_u_deg": out_u_deg, "in_u_deg": in_u_deg,
                        "out_v_w": out_v_w, "in_v_w": in_v_w, "out_v_deg": out_v_deg, "in_v_deg": in_v_deg,
                        "mode": str(mode), "success": 1, "delta": float(delta), "eval_ms": eval_ms,
                        "ts_parent": float(time.time()),
                    })

            assessed_set = set(selected_global_indices)
            S_indices = [gi for gi in S_indices if gi not in assessed_set]
            S_indices = filter_S_keep_backward(scores, S_indices)

            batch_dt = max(1e-9, time.time() - batch_start)
            if ML_LOG_EVERY_BATCH and (dp_batches_done % max(1, int(ML_LOG_EVERY_BATCH)) == 0) and serial_attempts > 0:
                log_message(
                    f"[FUNC1][ML] batch_stats path=SERIAL batch={dp_batches_done} attempts={serial_attempts} succ={serial_succ} "
                    f"succ_rate={(serial_succ/serial_attempts):.4f} sum_delta={serial_sum_delta:.6g} "
                    f"delta_per_sec={(serial_sum_delta/batch_dt):.6g} dt={batch_dt:.3f}s run_tag={ML_RUN_TAG}",
                    log_path
                )

            if applied_edges_round > 0:
                log_message(
                    f"[FUNC1] IMPROVE batch={dp_batches_done} applied={applied_edges_round}/{num_edges_batch} "
                    f"| dBW={-total_bw_reduct_round:.6f} | BW_now={BW_tracked:.2f} best={best_BW:.2f} | S={len(S_indices)}",
                    log_path
                )

            if BW_tracked + EPS_IMPROVE < best_BW:
                best_BW = BW_tracked
                best_scores = scores.copy()
                save_scores_to_csv(best_scores, output_excel, index_to_node)
                log_message(f"[FUNC1] NEW_BEST BW={best_BW:.2f} saved={output_excel}", log_path)

            if BW_CHECK_EVERY_ROUND > 0 and (round_idx % BW_CHECK_EVERY_ROUND == 1):
                fw_recompute = compute_forward_weight(edges, scores)
                bw_recompute = total_weight - float(fw_recompute)
                if abs(bw_recompute - BW_tracked) > BW_MISMATCH_TOL:
                    _log_crash_packet(
                        "SANITY_MISMATCH_SERIAL",
                        batch=dp_batches_done,
                        round_idx=round_idx,
                        epoch_id=epoch_id,
                        BW_tracked=BW_tracked,
                        best_BW=best_BW,
                        s_len=len(S_indices),
                        num_edges_batch=num_edges_batch,
                        extra_lines=[
                            f"BW_recompute={bw_recompute:.6f} BW_tracked={BW_tracked:.6f} diff={bw_recompute-BW_tracked:.6g} tol={BW_MISMATCH_TOL}",
                        ]
                    )
                    raise RuntimeError("Backward-weight tracking mismatch (SERIAL).")

            if (time.time() - last_improvement_time) >= eff_no_improve_now:
                log_message(
                    f"[FUNC1] STOP no_improve_for={time.time() - last_improvement_time:.1f}s "
                    f"limit={eff_no_improve_now:.1f}s (clamped)",
                    log_path
                )
                break

            continue  # next batch

        # ------------------------------------------
        # PARALLEL path
        # ------------------------------------------
        if _deadline_hit():
            log_message(f"[FUNC1] STOP deadline hit before spawning PARALLEL batch={dp_batches_done}", log_path)
            break

        num_procs_effective = min(num_procs, num_edges_batch, int(MAX_PROCS_PER_BATCH))

        edge_queue = mp.Queue()
        result_queue = mp.Queue()

        for i in range(num_edges_batch):
            edge_queue.put(i)
        for _ in range(num_procs_effective):
            edge_queue.put(None)

        active_intervals = None
        used_intervals = None
        lock = None

        workers = []
        for wi in range(num_procs_effective):
            if _deadline_hit():
                log_message(f"[FUNC1][WARN] deadline hit while spawning workers batch={dp_batches_done}", log_path)
                break
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
                ),
                kwargs={
                    "debug": bool(WORKER_DEBUG_LOGS),
                    "DEBUG_SAMPLE_EVERY": int(WORKER_DEBUG_SAMPLE_EVERY),
                    "DEBUG_MAX_EDGE_LOGS": int(WORKER_DEBUG_MAX_EDGE_LOGS),
                    "DEBUG_LOG_TOP_GAINS": bool(WORKER_DEBUG_LOGS),
                    "DEBUG_BIG_GAIN": float(WORKER_DEBUG_BIG_GAIN),

                    # IMPORTANT FIXES:
                    "QUEUE_GET_TIMEOUT_SEC": float(WORKER_QUEUE_GET_TIMEOUT_SEC),
                    "deadline_ts": float(deadline),  # enforce same hard slice in workers
                    "ML_ENABLE_LOGGING": bool(ML_WRITE_DATASET),
                    "ML_SAMPLE_LOG_EVERY": int(WORKER_ML_SAMPLE_LOG_EVERY),
                    "ML_MAX_SAMPLE_LOGS": int(WORKER_ML_MAX_SAMPLE_LOGS),
                    "ML_LOG_PREFIX": str(WORKER_ML_LOG_PREFIX),
                }
            )
            p.daemon = False
            p.start()
            workers.append(p)

        if len(workers) < num_procs_effective:
            _cleanup_workers(workers, reason_tag=f"spawn_incomplete_deadline_batch={dp_batches_done}")
            log_message("[FUNC1] STOP deadline/time_limit", log_path)
            break

        # ------------------------------
        # Collect results (robust + deadline-aware)
        # ------------------------------
        batch_start = time.time()
        alive_workers = num_procs_effective
        results = []
        worker_done = {}
        broke_for_deadline = False

        ml_batch_rows = 0
        ml_batch_attempts = 0
        ml_batch_succ = 0
        ml_batch_sum_delta = 0.0
        ml_batch_sum_eval_ms = 0.0
        ml_batch_mode_counts = {"extended": 0, "greedy": 0, "none": 0, "error": 0}

        while alive_workers > 0:
            if _deadline_hit():
                broke_for_deadline = True
                log_message(f"[FUNC1][WARN] STOP during_collect deadline batch={dp_batches_done}", log_path)
                break

            try:
                res = result_queue.get(timeout=min(float(RESULT_GET_TIMEOUT_SEC), max(0.1, _time_left())))
            except _pyqueue.Empty:
                if all((not p.is_alive()) for p in workers):
                    _log_crash_packet(
                        "ALL_WORKERS_DEAD_BEFORE_DONE",
                        batch=dp_batches_done,
                        round_idx=round_idx,
                        epoch_id=epoch_id,
                        BW_tracked=BW_tracked,
                        best_BW=best_BW,
                        s_len=len(S_indices),
                        num_edges_batch=num_edges_batch,
                        extra_lines=[
                            f"num_procs_effective={num_procs_effective} results_so_far={len(results)} done_so_far={len(worker_done)}",
                        ]
                    )
                    log_message(f"[FUNC1][WARN] all_workers_dead_before_done batch={dp_batches_done}", log_path)
                    break
                continue

            if res.get("done"):
                alive_workers -= 1
                widx = int(res.get("worker_idx", -1))
                worker_done[widx] = res

                ms = res.get("ml_stats", {}) or {}
                try:
                    ml_batch_attempts += int(ms.get("attempts", 0))
                    ml_batch_succ += int(ms.get("successes", 0))
                    ml_batch_sum_delta += float(ms.get("sum_delta", 0.0))
                    ml_batch_sum_eval_ms += float(ms.get("sum_eval_ms", 0.0))
                    mc = ms.get("mode_counts", {}) or {}
                    for k in ml_batch_mode_counts.keys():
                        ml_batch_mode_counts[k] += int(mc.get(k, 0))
                except Exception:
                    pass
            else:
                if ML_WRITE_DATASET and ("ml_row" in res):
                    mlr = res.get("ml_row", None)
                    if isinstance(mlr, dict):
                        out_row = dict(mlr)
                        out_row["run_tag"] = ML_RUN_TAG
                        out_row["graph"] = graph_name
                        out_row["path"] = "parallel"
                        out_row["epoch"] = int(epoch_id)
                        out_row["round"] = int(round_idx)
                        out_row["batch"] = int(dp_batches_done)
                        out_row["ts_parent"] = float(time.time())
                        _ml_write_row(out_row)
                        ml_batch_rows += 1
                results.append(res)

        if broke_for_deadline and TERMINATE_ON_DEADLINE_DURING_COLLECT:
            _cleanup_workers(workers, reason_tag=f"deadline_during_collect_batch={dp_batches_done}")
        else:
            _cleanup_workers(workers, reason_tag=f"normal_batch={dp_batches_done}")

        if broke_for_deadline:
            log_message("[FUNC1] STOP deadline/time_limit", log_path)
            break

        # ------------------------------
        # Apply ACCEPTED results + STRICT BW checks
        # ------------------------------
        applied_edges_round = 0
        total_bw_reduct_round = 0.0
        recompute_left = int(DEBUG_RECOMP_ACCEPTS_PER_BATCH) if debug else 0

        for r in results:
            if _deadline_hit():
                log_message(f"[FUNC1][WARN] deadline hit during APPLY batch={dp_batches_done}", log_path)
                break

            sflag = bool(r.get("success", False))
            delta = float(r.get("delta", 0.0) or 0.0)
            changed_scores = r.get("changed_scores", {}) or {}
            mode = r.get("mode", "?")
            interval = r.get("interval", None)
            u = r.get("u", None)
            v = r.get("v", None)

            if not (sflag and delta > EPS_IMPROVE and changed_scores):
                continue

            if recompute_left > 0:
                bw_before = total_weight - float(compute_forward_weight(edges, scores))
                scores_test = scores.copy()
                for node, new_r in changed_scores.items():
                    scores_test[node] = new_r
                bw_after = total_weight - float(compute_forward_weight(edges, scores_test))

                if bw_after > bw_before + BW_INCREASE_TOL:
                    _log_crash_packet(
                        "BW_INCREASE_RECOMP_PARALLEL",
                        batch=dp_batches_done,
                        round_idx=round_idx,
                        epoch_id=epoch_id,
                        BW_tracked=BW_tracked,
                        best_BW=best_BW,
                        s_len=len(S_indices),
                        num_edges_batch=num_edges_batch,
                        extra_lines=[
                            f"path=PARALLEL mode={mode} u={u} v={v} interval={interval} delta_reported={delta:.6g}",
                            f"bw_before={bw_before:.6f} bw_after={bw_after:.6f} bw_delta_recomp={bw_after-bw_before:.6g}",
                            _summarize_changed_scores(changed_scores, scores),
                        ]
                    )
                    raise RuntimeError("Backward weight increased after applying an accepted change (PARALLEL, recomputed).")

                recompute_left -= 1

            bw_before_tracked = BW_tracked
            for node, new_r in changed_scores.items():
                scores[node] = new_r
            BW_tracked -= delta

            if BW_tracked > bw_before_tracked + BW_INCREASE_TOL:
                _log_crash_packet(
                    "BW_TRACKED_INCREASE_PARALLEL",
                    batch=dp_batches_done,
                    round_idx=round_idx,
                    epoch_id=epoch_id,
                    BW_tracked=BW_tracked,
                    best_BW=best_BW,
                    s_len=len(S_indices),
                    num_edges_batch=num_edges_batch,
                    extra_lines=[
                        f"path=PARALLEL mode={mode} u={u} v={v} interval={interval} delta_reported={delta:.6g}",
                        f"BW_tracked_before={bw_before_tracked:.6f} BW_tracked_after={BW_tracked:.6f} increase={BW_tracked-bw_before_tracked:.6g}",
                        _summarize_changed_scores(changed_scores, scores_snapshot),
                    ]
                )
                raise RuntimeError("BW_tracked increased after applying an accepted change (PARALLEL, forbidden).")

            applied_edges_round += 1
            total_bw_reduct_round += delta
            last_improvement_time = time.time()

        # ------------------------------
        # Update S
        # ------------------------------
        assessed_set = set(selected_global_indices)
        S_indices = [gi for gi in S_indices if gi not in assessed_set]
        S_indices = filter_S_keep_backward(scores, S_indices)

        # ------------------------------
        # Evidence logs
        # ------------------------------
        batch_dt = max(1e-9, time.time() - batch_start)
        if ML_LOG_EVERY_BATCH and (dp_batches_done % max(1, int(ML_LOG_EVERY_BATCH)) == 0):
            succ_rate = (applied_edges_round / max(1, num_edges_batch))
            log_message(
                f"[FUNC1][ML] batch_stats path=PARALLEL batch={dp_batches_done} dp_edges={num_edges_batch} applied={applied_edges_round} "
                f"succ_rate={succ_rate:.4f} sum_delta={total_bw_reduct_round:.6g} delta_per_sec={(total_bw_reduct_round/batch_dt):.6g} "
                f"dt={batch_dt:.3f}s | worker_ml_attempts={ml_batch_attempts} worker_ml_succ={ml_batch_succ} "
                f"worker_sum_delta={ml_batch_sum_delta:.6g} worker_sum_eval_ms={ml_batch_sum_eval_ms:.2f} "
                f"modes(ext/greedy/none/error)={ml_batch_mode_counts['extended']}/{ml_batch_mode_counts['greedy']}/"
                f"{ml_batch_mode_counts['none']}/{ml_batch_mode_counts['error']} "
                f"| ml_rows_written_total={ml_rows_written} (+{ml_batch_rows}) run_tag={ML_RUN_TAG}",
                log_path
            )

        if applied_edges_round > 0:
            log_message(
                f"[FUNC1] IMPROVE batch={dp_batches_done} applied={applied_edges_round}/{num_edges_batch} "
                f"| dBW={-total_bw_reduct_round:.6f} | BW_now={BW_tracked:.2f} best={best_BW:.2f} | S={len(S_indices)}",
                log_path
            )

        if BW_tracked + EPS_IMPROVE < best_BW:
            best_BW = BW_tracked
            best_scores = scores.copy()
            save_scores_to_csv(best_scores, output_excel, index_to_node)
            log_message(f"[FUNC1] NEW_BEST BW={best_BW:.2f} saved={output_excel}", log_path)

        if BW_CHECK_EVERY_ROUND > 0 and (round_idx % BW_CHECK_EVERY_ROUND == 1):
            fw_recompute = compute_forward_weight(edges, scores)
            bw_recompute = total_weight - float(fw_recompute)
            if abs(bw_recompute - BW_tracked) > BW_MISMATCH_TOL:
                _log_crash_packet(
                    "SANITY_MISMATCH_PARALLEL",
                    batch=dp_batches_done,
                    round_idx=round_idx,
                    epoch_id=epoch_id,
                    BW_tracked=BW_tracked,
                    best_BW=best_BW,
                    s_len=len(S_indices),
                    num_edges_batch=num_edges_batch,
                    extra_lines=[
                        f"BW_recompute={bw_recompute:.6f} BW_tracked={BW_tracked:.6f} diff={bw_recompute-BW_tracked:.6g} tol={BW_MISMATCH_TOL}",
                    ]
                )
                raise RuntimeError("Backward-weight tracking mismatch (PARALLEL).")

        eff_no_improve_now = max(1.0, min(eff_no_improve_sec, max(1.0, _time_left())))
        if (time.time() - last_improvement_time) >= eff_no_improve_now:
            log_message(
                f"[FUNC1] STOP no_improve_for={time.time() - last_improvement_time:.1f}s "
                f"limit={eff_no_improve_now:.1f}s (clamped)",
                log_path
            )
            break

    # Final save best_scores
    save_scores_to_csv(best_scores, output_excel, index_to_node)
    final_fw = compute_forward_weight(edges, best_scores)
    final_bw = total_weight - float(final_fw)

    # Remaining-unassessed pairs (for the hybrid driver)
    remaining_pairs_unassessed = None
    if subset_pairs is not None:
        remaining_pairs_unassessed = list(set(subset_pairs) - set(assessed_pairs_in_this_call))

    # Flush/close ML dataset
    if ML_WRITE_DATASET and (ml_fh is not None):
        try:
            ml_fh.flush()
        except Exception:
            pass
        try:
            ml_fh.close()
        except Exception:
            pass

    log_message(
        f"[FUNC1] END bestBW={final_bw:.2f} wrote={output_excel} | ml_rows={ml_rows_written} ml_dataset={ML_DATASET_PATH}",
        log_path
    )
    return best_scores, final_bw, remaining_pairs_unassessed

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
    Kahn's algorithm topological sort (robust + deterministic).

    Inputs:
      nodes: iterable of nodes (may include isolated nodes)
      out_adj: dict u -> iterable of v (list/set/etc). Nodes not in `nodes`
               are ignored by default.

    Returns:
      (is_acyclic, topo_list)

    Notes / fixes vs old version:
      - De-duplicates parallel neighbor entries (u -> v repeated) so indegrees are correct.
      - Ensures every node in `nodes` appears in output topo_list if acyclic.
      - Deterministic ordering of the initial zero-indegree queue:
          preserves the order of `nodes` as given.
    """
    nodes = list(nodes)
    node_set = set(nodes)

    # indegree over the given node set only
    indeg = {v: 0 for v in nodes}

    # Compute indegrees, de-duplicating out neighbors per u
    for u in nodes:
        nbrs = out_adj.get(u, ())
        if not nbrs:
            continue
        seen = set()
        for v in nbrs:
            if v in node_set and v not in seen:
                indeg[v] += 1
                seen.add(v)

    # Initialize queue with all zero-indegree nodes, in the order they appear in `nodes`
    q = [v for v in nodes if indeg[v] == 0]

    topo = []
    head = 0
    while head < len(q):
        x = q[head]
        head += 1
        topo.append(x)

        nbrs = out_adj.get(x, ())
        if not nbrs:
            continue

        # De-duplicate neighbors for consistent indegree updates
        seen = set()
        for y in nbrs:
            if y in node_set and y not in seen:
                indeg[y] -= 1
                if indeg[y] == 0:
                    q.append(y)
                seen.add(y)

    return (len(topo) == len(nodes)), topo



def wmsf_remove_arcs(nodes, edges, ordering="L2", log_path=None, MAX_SEC=None):
    """
    Paper-faithful WMSF phase-1: remove arcs (not vertices) until the remaining graph is acyclic.

    Implements the paper's removeArcs(G, L) with:
      - Arc ordering L1 or L2:
          L1: increasing weight
          L2: increasing  w(u,v) / (W_in(u,G) + W_out(v,G))
        where W_in/W_out are computed on the ORIGINAL graph G (as in the paper's definition).
      - 2-cycle preprocessing: for each pair (u,v) and (v,u), eliminate the one that appears first in L
      - Safe arcs peeling: repeatedly remove arcs (u,v) where in_degree(u)==0 OR out_degree(v)==0
        (these cannot be in any cycle). Safe arcs are removed from the working graph but NOT added to F.
      - Acyclicity checks every alpha eliminations, alpha = |A|/|V| (at least 1).

    Returns:
      F: set of eliminated arcs (u,v) that form the feedback arc set produced by removeArcs.
    """
    import time
    from collections import deque

    start = time.time()

    if ordering not in ("L1", "L2"):
        ordering = "L2"

    # ---------- aggregate parallel edges ----------
    # Treat parallel edges as a single arc with summed weight (MWFAS-compatible).
    w_map = {}
    for (u, v, w) in edges:
        w = float(w)
        key = (u, v)
        w_map[key] = w_map.get(key, 0.0) + w

    # Ensure all nodes exist in maps
    nodes = list(nodes)

    # ---------- build adjacency for the CURRENT working graph ----------
    out_adj = {u: {} for u in nodes}   # out_adj[u][v] = weight
    in_adj  = {u: {} for u in nodes}   # in_adj[v][u] = weight

    for (u, v), w in w_map.items():
        if u not in out_adj:
            out_adj[u] = {}
            in_adj[u] = {}
            nodes.append(u)
        if v not in out_adj:
            out_adj[v] = {}
            in_adj[v] = {}
            nodes.append(v)
        out_adj[u][v] = out_adj[u].get(v, 0.0) + w
        in_adj[v][u]  = in_adj[v].get(u, 0.0) + w

    # Degree counts for safe-arc logic (counts, not weights)
    in_deg  = {u: len(in_adj[u]) for u in out_adj}
    out_deg = {u: len(out_adj[u]) for u in out_adj}

    # ---------- compute W_in/W_out on ORIGINAL graph (weights) ----------
    W_in  = {u: sum(in_adj[u].values()) for u in out_adj}
    W_out = {u: sum(out_adj[u].values()) for u in out_adj}

    # ---------- build arc ordering L ----------
    arcs = list(w_map.keys())

    if ordering == "L1":
        # increasing weight
        L = sorted(arcs, key=lambda a: (w_map[a], a[0], a[1]))
    else:
        # L2: increasing ratio w(u,v)/(W_in(u)+W_out(v))
        def l2_key(a):
            u, v = a
            denom = (W_in.get(u, 0.0) + W_out.get(v, 0.0))
            # Paper uses this ratio; keep stable tie-breakers.
            return (w_map[a] / (denom + 1e-12), w_map[a], u, v)
        L = sorted(arcs, key=l2_key)

    pos = {a: i for i, a in enumerate(L)}

    # ---------- helpers ----------
    def _time_exceeded():
        return (MAX_SEC is not None) and ((time.time() - start) > MAX_SEC)

    def remove_arc(u, v):
        """Remove arc (u,v) from current graph if present. Returns True if removed."""
        if v not in out_adj.get(u, {}):
            return False
        # delete from adj
        del out_adj[u][v]
        del in_adj[v][u]
        # update degrees
        out_deg[u] -= 1
        in_deg[v]  -= 1
        return True

    def is_dag():
        """Kahn's algorithm on current graph."""
        # Copy indegrees
        indeg = dict(in_deg)
        q = deque([u for u in indeg if indeg[u] == 0])
        popped = 0
        while q:
            u = q.popleft()
            popped += 1
            for v in out_adj[u].keys():
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return popped == len(indeg)

    def peel_safe_arcs():
        """
        Repeatedly remove safe arcs:
          - if in_deg(u)==0 then all outgoing arcs (u,*) are safe
          - if out_deg(v)==0 then all incoming arcs (*,v) are safe
        Safe arcs are NOT added to F.
        """
        removed_safe = 0
        q_in0  = deque([u for u in out_adj if in_deg[u] == 0 and out_deg[u] > 0])
        q_out0 = deque([u for u in out_adj if out_deg[u] == 0 and in_deg[u] > 0])

        while q_in0 or q_out0:
            # process in_deg==0 nodes => remove all outgoing
            while q_in0:
                u = q_in0.popleft()
                if in_deg[u] != 0 or out_deg[u] == 0:
                    continue
                # remove all outgoing arcs (u, v)
                for v in list(out_adj[u].keys()):
                    # Remove (u,v)
                    if remove_arc(u, v):
                        removed_safe += 1
                        # degree-triggered queue updates
                        if in_deg[v] == 0 and out_deg[v] > 0:
                            q_in0.append(v)
                        if out_deg[v] == 0 and in_deg[v] > 0:
                            q_out0.append(v)
                # u now has out_deg[u]==0; if still has incoming, it triggers out0-logic (incoming are safe)
                if out_deg[u] == 0 and in_deg[u] > 0:
                    q_out0.append(u)

            # process out_deg==0 nodes => remove all incoming
            while q_out0:
                v = q_out0.popleft()
                if out_deg[v] != 0 or in_deg[v] == 0:
                    continue
                # remove all incoming arcs (u, v)
                for u in list(in_adj[v].keys()):
                    if remove_arc(u, v):
                        removed_safe += 1
                        # degree-triggered queue updates
                        if out_deg[u] == 0 and in_deg[u] > 0:
                            q_out0.append(u)
                        if in_deg[u] == 0 and out_deg[u] > 0:
                            q_in0.append(u)
                # v now has in_deg[v]==0; if still has outgoing (shouldn't), it triggers in0-logic
                if in_deg[v] == 0 and out_deg[v] > 0:
                    q_in0.append(v)

        return removed_safe

    # ---------- main ----------
    F = set()

    n = len(out_adj)
    m0 = sum(out_deg.values())
    alpha = max(1, m0 // max(1, n))  # paper: alpha = |A|/|V|, at least 1

    # 1) preprocess 2-cycles: eliminate the one that appears first in L
    handled = set()
    removed_2cycle = 0
    for (u, v) in L:
        if _time_exceeded():
            if log_path:
                log_message(f"[WMSF] remove_arcs: hit MAX_SEC during 2-cycle preprocess | F={len(F)}", log_path)
            return F

        if (u, v) in handled:
            continue
        if (v, u) in pos:
            # both directions exist in ORIGINAL arc list
            handled.add((u, v))
            handled.add((v, u))
            a1 = (u, v) if pos[(u, v)] < pos[(v, u)] else (v, u)
            x, y = a1
            # eliminate a1 if still present in current graph
            if y in out_adj.get(x, {}):
                remove_arc(x, y)
                F.add((x, y))
                removed_2cycle += 1

    # 2) peel safe arcs before doing anything else
    removed_safe0 = peel_safe_arcs()

    # If already a DAG, stop.
    if is_dag():
        if log_path:
            m_now = sum(out_deg.values())
            log_message(
                f"[WMSF] remove_arcs done (already DAG after preprocess) | ordering={ordering} "
                f"| alpha={alpha} | F={len(F)} | removed_2cycle={removed_2cycle} | safe_removed={removed_safe0} "
                f"| m0={m0} m_now={m_now}",
                log_path
            )
        return F

    # 3) eliminate arcs in order L until DAG, checking every alpha eliminations
    eliminated = 0
    removed_safe = removed_safe0

    for (u, v) in L:
        if _time_exceeded():
            if log_path:
                m_now = sum(out_deg.values())
                log_message(
                    f"[WMSF] remove_arcs: hit MAX_SEC={MAX_SEC}s | ordering={ordering} "
                    f"| alpha={alpha} | eliminated={eliminated} | F={len(F)} | safe_removed={removed_safe} "
                    f"| m_now={m_now}",
                    log_path
                )
            break

        # arc might have been removed already (2-cycle or safe peeling or earlier elimination)
        if v not in out_adj.get(u, {}):
            continue

        # eliminate it into F
        remove_arc(u, v)
        F.add((u, v))
        eliminated += 1

        # peel newly safe arcs (paper removes safe arcs to speed DAG checking)
        removed_safe += peel_safe_arcs()

        # check DAG every alpha eliminations
        if eliminated % alpha == 0:
            if is_dag():
                break

    # Final check (if we didn't land exactly on multiple of alpha)
    # Not strictly necessary, but harmless and avoids returning a cyclic remainder.
    if not _time_exceeded():
        if not is_dag():
            # continue eliminating remaining active arcs (still respecting MAX_SEC)
            for (u, v) in L:
                if _time_exceeded():
                    break
                if v not in out_adj.get(u, {}):
                    continue
                remove_arc(u, v)
                F.add((u, v))
                eliminated += 1
                removed_safe += peel_safe_arcs()
                if eliminated % alpha == 0 and is_dag():
                    break

    if log_path:
        m_now = sum(out_deg.values())
        log_message(
            f"[WMSF] remove_arcs done | ordering={ordering} | alpha={alpha} "
            f"| eliminated={eliminated} | F={len(F)} | removed_2cycle={removed_2cycle} "
            f"| safe_removed={removed_safe} | m0={m0} m_now={m_now}",
            log_path
        )

    return F



def wmsf_minimize_fas(nodes, edges, F, log_path=None, MAX_CHECKS=None, MAX_SEC=None):
    """
    Paper-faithful WMSF phase-3: MinimizeFas(G, F)

    - Start from the acyclic graph G' = (V, A \ F)
    - Consider arcs in F in DECREASING weight order (paper)
    - Try to reinsert each arc; keep it removed only if reinsertion creates a cycle.

    Budgets:
      MAX_CHECKS: max number of arcs from F to try (in sorted order)
      MAX_SEC: soft wall-time cap; if exceeded, stop early and return current F
    """
    import time
    from collections import deque

    start = time.time()

    # ---------- aggregate parallel edges and build weight map ----------
    w_map = {}
    node_set = set(nodes)
    for (u, v, w) in edges:
        w = float(w)
        w_map[(u, v)] = w_map.get((u, v), 0.0) + w
        node_set.add(u)
        node_set.add(v)

    # ensure nodes covers endpoints
    nodes = list(node_set)

    # ---------- build working edge-set E' = E \ F ----------
    F = set(F)
    Eprime = set((u, v) for (u, v) in w_map.keys() if (u, v) not in F)

    # ---------- adjacency for incremental topo-check ----------
    out_adj = {u: set() for u in nodes}
    indeg = {u: 0 for u in nodes}

    for (u, v) in Eprime:
        if u not in out_adj:
            out_adj[u] = set()
            indeg[u] = indeg.get(u, 0)
        if v not in out_adj:
            out_adj[v] = set()
            indeg[v] = indeg.get(v, 0)
        if v not in out_adj[u]:
            out_adj[u].add(v)
            indeg[v] += 1

    def is_dag_current():
        """Kahn topo on CURRENT out_adj/indeg (copies indeg)."""
        indeg_tmp = dict(indeg)
        q = deque([u for u in out_adj.keys() if indeg_tmp.get(u, 0) == 0])
        seen = 0
        while q:
            u = q.popleft()
            seen += 1
            for v in out_adj[u]:
                indeg_tmp[v] -= 1
                if indeg_tmp[v] == 0:
                    q.append(v)
        return seen == len(out_adj)

    def path_exists(src, dst):
        """
        Cycle test for adding edge (src->dst):
        adding (src->dst) creates a cycle iff dst can reach src in current graph.
        We implement a BFS from dst to see if src is reachable.
        """
        if src == dst:
            return True
        q = deque([dst])
        seen = {dst}
        while q:
            x = q.popleft()
            for y in out_adj.get(x, ()):
                if y == src:
                    return True
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        return False

    # Sanity: Eprime should be acyclic if previous phases were correct;
    # we don't fail hard, but we can log.
    if log_path:
        try:
            if not is_dag_current():
                log_message("[WMSF][WARN] minimize_fas: starting E' is not a DAG (unexpected). Continuing anyway.", log_path)
        except Exception as e:
            log_message(f"[WMSF][WARN] minimize_fas: DAG sanity check failed with {type(e).__name__}: {e}", log_path)

    # ---------- process arcs in F by decreasing weight ----------
    # If an arc in F doesn't appear in w_map (shouldn't), treat weight as 0.
    F_sorted = sorted(F, key=lambda a: (w_map.get(a, 0.0), a[0], a[1]), reverse=True)

    checks = 0
    reinserts = 0

    for (u, v) in F_sorted:
        # budgets
        if MAX_CHECKS is not None and checks >= MAX_CHECKS:
            if log_path:
                log_message(f"[WMSF] minimize_fas: hit MAX_CHECKS={MAX_CHECKS} | checks={checks} | keptF={len(F)}", log_path)
            break
        if MAX_SEC is not None and (time.time() - start) > MAX_SEC:
            if log_path:
                log_message(f"[WMSF] minimize_fas: hit MAX_SEC={MAX_SEC}s | checks={checks} | keptF={len(F)}", log_path)
            break

        checks += 1

        # If arc already reinserted by some earlier logic, skip
        if (u, v) in Eprime:
            if (u, v) in F:
                F.remove((u, v))
            continue

        # Paper logic: try to add; if it creates a cycle, keep it in F; else remove from F.
        # Efficient test: adding (u->v) creates a cycle iff v reaches u in current graph.
        if not path_exists(u, v):
            # safe to reinsert
            Eprime.add((u, v))
            out_adj.setdefault(u, set()).add(v)
            indeg.setdefault(v, 0)
            indeg[v] += 1
            # keep bookkeeping consistent for nodes that may not exist in dicts
            out_adj.setdefault(v, set())
            indeg.setdefault(u, indeg.get(u, 0))

            if (u, v) in F:
                F.remove((u, v))
            reinserts += 1

        if log_path and (checks % 5000 == 0):
            log_message(
                f"[WMSF] minimize_fas progress | checks={checks} | reinserts={reinserts} | keptF={len(F)}",
                log_path
            )

    if log_path:
        log_message(
            f"[WMSF] minimize_fas done | checks={checks} | reinserts={reinserts} | keptF={len(F)}",
            log_path
        )

    return F



def wmsf_stabilize_fas(nodes, edges, F, log_path=None):
    """
    Paper-faithful WMSF phase-2: StabilizeFas(G, F)

    Goal:
      Enforce the paper's weight-stability inequalities by applying the described swaps
      along a topological order of G* = (V, A \ F). Repeat up to floor(log2(|V|)) passes.

    IMPORTANT (paper-faithful behavior):
      - G* must remain acyclic; the paper's transformation is designed to preserve acyclicity.
      - We DO NOT "rollback" as a control mechanism. Instead, we:
          * compute topo order of current G*
          * apply swap rules exactly as described
          * (optional) do a safety DAG check only for debugging/logging; if it fails, we warn but keep going

    Inputs:
      nodes: iterable of nodes
      edges: list of (u, v, w) arcs in the ORIGINAL graph G (weights used)
      F:    iterable of eliminated arcs (u, v) defining current FAS
    Returns:
      Fset: stabilized FAS (set of (u, v))
    """
    import math
    from collections import defaultdict

    nodes = list(nodes)
    node_set = set(nodes)

    # --- aggregate parallel edges into one weight per (u,v) ---
    w = {}
    A = set()
    for (u, v, ww) in edges:
        ww = float(ww)
        w[(u, v)] = w.get((u, v), 0.0) + ww
        A.add((u, v))
        if u not in node_set:
            node_set.add(u)
            nodes.append(u)
        if v not in node_set:
            node_set.add(v)
            nodes.append(v)

    # Precompute W_in(v,G), W_out(v,G) on original G (weights)
    W_in_G = {v: 0.0 for v in nodes}
    W_out_G = {v: 0.0 for v in nodes}
    for (u, v), ww in w.items():
        W_out_G[u] += ww
        W_in_G[v] += ww

    # Incident arc lists from original graph (for swaps)
    in_arcs = {v: [] for v in nodes}   # list of (a,b) with b=v
    out_arcs = {v: [] for v in nodes}  # list of (a,b) with a=v
    for (u, v) in A:
        out_arcs[u].append((u, v))
        in_arcs[v].append((u, v))

    max_passes = max(1, int(math.log2(max(2, len(nodes)))))

    def build_present_from_F(Fset):
        """Return present arc-set and adjacency list for G* = A \\ Fset."""
        present = A - set(Fset)
        out_adj = {x: [] for x in nodes}
        for (a, b) in present:
            out_adj[a].append(b)
        return present, out_adj

    Fset = set(F)

    for p in range(1, max_passes + 1):
        present, out_adj = build_present_from_F(Fset)

        acyclic, topo = kahn_toposort(nodes, out_adj)
        if not acyclic:
            # This should not happen if phase-1 was correct.
            if log_path:
                log_message(f"[WMSF] stabilizeFAS | PASS{p} | WARNING: input G* not acyclic; aborting stabilize.", log_path)
            break

        # Compute W_in(v,G*), W_out(v,G*) on current present arc set
        W_in_Gs = {v: 0.0 for v in nodes}
        W_out_Gs = {v: 0.0 for v in nodes}
        for (a, b) in present:
            ww = w.get((a, b), 0.0)
            W_out_Gs[a] += ww
            W_in_Gs[b] += ww

        local_changes = 0

        # Apply stabilization rules along topo order (paper)
        # Definitions:
        #   elim_in(v)  = W_in(G)  - W_in(G*)
        #   elim_out(v) = W_out(G) - W_out(G*)
        #   rem_in(v)   = W_in(G*)
        #   rem_out(v)  = W_out(G*)
        #
        # Violation (1): elim_in(v)  > rem_out(v)
        #   => eliminate all remaining OUT arcs of v; restore all eliminated IN arcs of v
        #
        # Violation (2): elim_out(v) > rem_in(v)
        #   => eliminate all remaining IN arcs of v; restore all eliminated OUT arcs of v
        #
        # If both violated, the paper discusses cases; we choose a deterministic tie-break:
        # apply the violation with larger "excess" magnitude.
        for v in topo:
            elim_in = W_in_G[v] - W_in_Gs[v]
            elim_out = W_out_G[v] - W_out_Gs[v]
            rem_in = W_in_Gs[v]
            rem_out = W_out_Gs[v]

            violated1 = (elim_in > rem_out)
            violated2 = (elim_out > rem_in)

            if not (violated1 or violated2):
                continue

            # Tie-break if both hold: act on the more severe violation (excess amount).
            if violated1 and violated2:
                excess1 = elim_in - rem_out
                excess2 = elim_out - rem_in
                do_v1 = (excess1 >= excess2)
            else:
                do_v1 = violated1

            if do_v1:
                # eliminate remaining outgoing arcs of v: add (v,*) present arcs to Fset
                for (a, b) in out_arcs[v]:
                    if (a, b) in present and (a, b) not in Fset:
                        Fset.add((a, b))
                # restore eliminated incoming arcs of v: remove (*,v) arcs from Fset
                for (a, b) in in_arcs[v]:
                    if (a, b) in Fset:
                        Fset.remove((a, b))
            else:
                # eliminate remaining incoming arcs of v: add (*,v) present arcs to Fset
                for (a, b) in in_arcs[v]:
                    if (a, b) in present and (a, b) not in Fset:
                        Fset.add((a, b))
                # restore eliminated outgoing arcs of v: remove (v,*) arcs from Fset
                for (a, b) in out_arcs[v]:
                    if (a, b) in Fset:
                        Fset.remove((a, b))

            local_changes += 1

        # Optional sanity check (debug only): verify acyclicity after the pass.
        # We do NOT rollback; we just warn if something is wrong.
        if log_path:
            present2, out_adj2 = build_present_from_F(Fset)
            acyclic2, _ = kahn_toposort(nodes, out_adj2)
            if not acyclic2:
                log_message(f"[WMSF] stabilizeFAS | PASS{p}/{max_passes} | WARNING: cycle detected after pass (unexpected).", log_path)
            log_message(f"[WMSF] stabilizeFAS | PASS{p}/{max_passes} | changes={local_changes}", log_path)

        if local_changes == 0:
            break

    return Fset


def wmsf_produce_ranking(csv_path, input_ranking_path, output_ranking_path, log_path,
                        ordering="L2", MAX_SEC=None, MAX_NODES=None):
    """
    Paper-faithful WMSF wrapper that follows Algorithm 1:

      1) F  = removeArcs(G, L)
      2) Fs = StabilizeFas(G, F)
      3) Fm = MinimizeFas(G, Fs)
      4) return Fm   (and we derive a ranking from E \\ Fm via a topological order)

    Notes:
      - No custom time-splitting between phases (paper does not specify it).
      - If MAX_SEC is provided, it is a *soft* cap for the whole routine:
          removeArcs: gets the bulk early; stabilize: typically fast; minimize: uses remaining time.
      - We derive scores from E\\F only at the end (paper returns FAS; ranking is our output format).

    Returns:
      scores_final, F_final
    """
    import time

    t0 = time.time()
    edges, node_to_index, index_to_node = read_graph(csv_path)
    n = len(node_to_index)

    if MAX_NODES is not None and n > int(MAX_NODES):
        log_message(f"[WMSF] SKIP produce_ranking: n={n} > MAX_NODES={MAX_NODES}", log_path)
        return None, None

    # seed scores are ONLY for tie-breaking when we convert DAG to a ranking
    scores_seed = load_initial_scores(input_ranking_path, node_to_index)

    if ordering not in ("L1", "L2"):
        ordering = "L2"

    # soft budget bookkeeping
    MAX_SEC = None if MAX_SEC is None else float(MAX_SEC)

    nodes = list(range(n))
    log_message(f"[WMSF] start | ordering={ordering} | n={n} | MAX_SEC={MAX_SEC}", log_path)

    # ------------------ 1) removeArcs ------------------
    # Give removeArcs a good chunk early, but do NOT hard-split if MAX_SEC is None.
    # If MAX_SEC is small, stabilization/minimization may be skipped naturally by their own MAX_SEC handling.
    if MAX_SEC is None:
        t_remove = None
        t_min = None
    else:
        # Spend up to 70% on removeArcs (it is the main construction), keep time for minimize.
        t_remove = 0.70 * MAX_SEC
        # Minimize can use remaining time; we pass a cap below after stabilization.
        t_min = MAX_SEC  # we will compute remaining and pass that

    F = wmsf_remove_arcs(nodes, edges, ordering=ordering, log_path=log_path, MAX_SEC=t_remove)
    log_message(f"[WMSF] remove_arcs done | ordering={ordering} | F={len(F)}", log_path)

    # If global MAX_SEC hit inside removeArcs, we still proceed (paper doesn't define timeouts),
    # but we respect the user's MAX_SEC by limiting later phases to the remaining time.
    if MAX_SEC is not None:
        elapsed = time.time() - t0
        if elapsed >= MAX_SEC:
            log_message(f"[WMSF] done early (budget) | dt={elapsed:.2f}s | F={len(F)} | out=None", log_path)
            return None, F

    # ------------------ 2) StabilizeFas ------------------
    F_stab = wmsf_stabilize_fas(nodes, edges, F, log_path=log_path)

    if MAX_SEC is not None:
        elapsed = time.time() - t0
        if elapsed >= MAX_SEC:
            # Return stabilized FAS; no ranking written (budget hit)
            log_message(f"[WMSF] done early after stabilize (budget) | dt={elapsed:.2f}s | F={len(F_stab)} | out=None", log_path)
            return None, F_stab

    # ------------------ 3) MinimizeFas ------------------
    # Use remaining time, if any.
    if MAX_SEC is None:
        t_min_remaining = None
    else:
        t_min_remaining = max(0.0, MAX_SEC - (time.time() - t0))
        # If nothing left, skip minimize (return stabilized FAS)
        if t_min_remaining <= 1e-9:
            log_message(f"[WMSF] minimize_fas skipped (no remaining budget) | F={len(F_stab)}", log_path)
            F_final = F_stab
        else:
            F_final = wmsf_minimize_fas(nodes, edges, F_stab, log_path=log_path, MAX_SEC=t_min_remaining)
    if MAX_SEC is None:
        F_final = wmsf_minimize_fas(nodes, edges, F_stab, log_path=log_path, MAX_SEC=None)

    # ------------------ derive ranking from E \\ F_final ------------------
    # Convert the DAG to a ranking using topo order, tie-broken by seed scores.
    scores_final = wmsf_scores_from_E_minus_F(nodes, edges, F_final, seed_scores=scores_seed, log_path=log_path)
    if scores_final is None:
        log_message("[WMSF] produce_ranking abort: E\\F cyclic after remove+stabilize+minimize (unexpected).", log_path)
        return None, F_final

    save_scores_to_csv(scores_final, output_ranking_path, index_to_node)

    dt = time.time() - t0
    log_message(f"[WMSF] done | ordering={ordering} | dt={dt:.2f}s | F={len(F_final)} | out={output_ranking_path}", log_path)
    return scores_final, F_final

def _normalize_F_to_pairs(F, log_path=None):
    """
    Normalize F into a set of (u, v) pairs.

    Accepts:
      - None / empty
      - iterable of (u,v) or (u,v,...) tuples/lists
      - dict-like keys (iterating yields keys)
      - iterables that might contain junk -> those entries are dropped

    Guarantees:
      - output contains ONLY 2-tuples (u,v)
      - logs how many were dropped (if any)

    Notes:
      - Does NOT cast node types (keeps u and v as-is). This is important because your
        nodes are typically ints (0..n-1) and mixing strings would silently break lookups.
      - If F accidentally contains edge triples (u,v,w), we ignore w.
    """
    if F is None:
        return set()

    out = set()
    malformed = 0
    total = 0

    # If someone passed a dict, iterating yields keys (fine).
    try:
        iterator = iter(F)
    except TypeError:
        if log_path:
            log_message(f"[WMSF] normalize_F: input not iterable (type={type(F).__name__}); returning empty.", log_path)
        return set()

    for e in iterator:
        total += 1
        try:
            # Most common: tuple/list
            if isinstance(e, (tuple, list)):
                if len(e) >= 2:
                    u, v = e[0], e[1]
                    out.add((u, v))
                else:
                    malformed += 1
                continue

            # Sometimes edges come as "u,v" strings -> treat as malformed (safer than guessing)
            # Sometimes edges come as objects with .u/.v -> also treat as malformed unless you
            # explicitly support it later.
            malformed += 1

        except Exception:
            malformed += 1

    if log_path and malformed:
        log_message(
            f"[WMSF] normalize_F: dropped malformed={malformed} kept={len(out)} total={total}",
            log_path
        )

    return out



def wmsf_scores_from_E_minus_F(nodes, edges, F, seed_scores, log_path=None):
    """
    Build a topo order of G* = (V, E \\ F) and convert it to a scores dict {node: rank}.

    Paper-faithful + consistent with the updated WMSF phases:
      - Treat parallel edges consistently by AGGREGATING them to a single arc (u,v) for DAG logic.
        (Multiplicity does not matter for acyclicity; weights don't matter for topo.)
      - Normalize F robustly to a set of (u,v) pairs.
      - Use seed_scores ONLY as a tie-break among currently zero-indegree nodes,
        with deterministic secondary tie-break by node id.

    Returns:
      scores dict mapping node -> integer rank, or None if E\\F is cyclic.
    """
    import heapq

    nodes = list(nodes)
    node_set = set(nodes)

    # Robust normalize for F
    Fset = _normalize_F_to_pairs(F, log_path=log_path)

    # --- Build present adjacency for E\\F, aggregating parallel edges ---
    out_adj = {u: set() for u in nodes}
    indeg = {u: 0 for u in nodes}

    # Ensure all endpoints exist in nodes (defensive)
    def _ensure_node(x):
        if x not in node_set:
            node_set.add(x)
            nodes.append(x)
            out_adj[x] = set()
            indeg[x] = 0

    for (u, v, _) in edges:
        _ensure_node(u)
        _ensure_node(v)

        if (u, v) in Fset:
            continue

        # aggregate / de-dup parallel edges for indeg correctness
        if v not in out_adj[u]:
            out_adj[u].add(v)
            indeg[v] += 1

    # --- Kahn topo with heap tie-break (seed rank, then node id) ---
    BIG = 10**18

    def seed_key(u):
        # seed_scores can be dict of ranks; missing => BIG
        try:
            return int(seed_scores.get(u, BIG))
        except Exception:
            return BIG

    heap = []
    for u in nodes:
        if indeg[u] == 0:
            heapq.heappush(heap, (seed_key(u), u))

    topo = []
    while heap:
        _, u = heapq.heappop(heap)
        topo.append(u)
        for v in out_adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (seed_key(v), v))

    if len(topo) != len(nodes):
        if log_path:
            # Optional: compute how many still have indeg>0 for a slightly more informative log
            remaining = sum(1 for u in nodes if indeg[u] > 0)
            log_message(
                f"[WMSF] ERROR: E\\F is still cyclic (topo={len(topo)}/{len(nodes)} remaining_indeg_pos={remaining}).",
                log_path
            )
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
    """
    Worker-side block refine for func2 (BW-MONOTONE ENFORCED BY CONTROLLER):

    - ALWAYS runs refine_block_scc_interval (even if no improvement).
    - Returns proposed block ranks + local ΔFW on induced subgraph.
    - Enforces critical structural validity:
        * proposed ranks on block_nodes must be a permutation of original ranks
          (same sorted multiset). Otherwise returns valid=False.
    - Uses ONLY log_message for output. Never prints.
    """
    # --------- UPDATED: accept optional trailing seed_sid ----------
    if len(args) == 11:
        (
            worker_id,
            block_nodes,
            r_low,
            r_high,
            brute_force_min_size,
            brute_force_max_size,
            debug,
            log_path,
            dp_exact_max_size,
            large_scc_ls_iters,
            eps_improve,
        ) = args
        seed_sid = None
    elif len(args) == 12:
        (
            worker_id,
            block_nodes,
            r_low,
            r_high,
            brute_force_min_size,
            brute_force_max_size,
            debug,
            log_path,
            dp_exact_max_size,
            large_scc_ls_iters,
            eps_improve,
            seed_sid,   # NEW (optional, may be None)
        ) = args
    else:
        raise ValueError(f"_worker_refine_block: unexpected args length={len(args)}")

    import os, time

    global G_GLOBAL, EDGES_GLOBAL, SCORES_BEFORE_GLOBAL
    G = G_GLOBAL
    edges = EDGES_GLOBAL
    scores_before = SCORES_BEFORE_GLOBAL

    pid = os.getpid()
    block_nodes = list(block_nodes)

    def _log(msg: str):
        # ultra-quiet: only log when debug or important failure
        if log_path is not None:
            log_message(f"[func2-worker pid={pid} {worker_id}] {msg}", log_path)

    t0 = time.time()

    # Baseline ranks multiset (must be preserved)
    orig_ranks_sorted = sorted(scores_before[n] for n in block_nodes)

    # Baseline block-internal FW (induced subgraph)
    fw_block_before = _compute_block_fw(G, scores_before, block_nodes)

    if debug:
        _log(
            f"Start block | n={len(block_nodes)} | seed_sid={seed_sid} | "
            f"r=[{float(r_low):.3f},{float(r_high):.3f}] | "
            f"dp_exact_max_size={int(dp_exact_max_size)} | FW_block_before={fw_block_before:.6f}"
        )

    # Always run solver.
    # IMPORTANT: if controller disabled DP for this block, it passes dp_exact_max_size=0;
    # we forward it as-is so refine_block_scc_interval will skip DP attempts naturally.
    solver_improved = False
    try:
        new_scores, _, _, solver_improved = refine_block_scc_interval(
            G=G,
            edges=edges,
            node_order=scores_before,
            block_nodes=block_nodes,
            brute_force_min_size=brute_force_min_size,
            brute_force_max_size=brute_force_max_size,
            debug=debug,
            dp_exact_max_size=int(dp_exact_max_size),
            large_scc_ls_iters=int(large_scc_ls_iters),
            eps_improve=float(eps_improve),
            log_path=log_path,
            # keep summaries only when not improved (your quiet policy)
            log_solvers_always=False,
            log_only_when_no_gain=True,
        )
    except Exception as e:
        elapsed = time.time() - t0
        # important failure => log one line even if debug=False
        _log(f"❌ Exception in refine_block_scc_interval ({type(e).__name__}: {e}).")
        return {
            "block_id": worker_id,
            "worker_id": worker_id,
            "pid": pid,
            "seed_sid": seed_sid,
            "r_low": float(r_low),
            "r_high": float(r_high),
            "n_block": len(block_nodes),
            "valid": False,
            "why_invalid": f"solver_exception:{type(e).__name__}:{e}",
            "block_ranks": {n: scores_before[n] for n in block_nodes},  # safe fallback
            "fw_block_before": float(fw_block_before),
            "fw_block_after": float(fw_block_before),
            "local_delta_fw": 0.0,
            "solver_improved": bool(solver_improved),
            "elapsed_worker": float(elapsed),
            "dp_exact_max_size": int(dp_exact_max_size),
        }

    # Compute proposed internal gain
    fw_block_after = _compute_block_fw(G, new_scores, block_nodes)
    local_delta_fw = fw_block_after - fw_block_before

    # Validate rank multiset (MUST be the same ranks, permuted)
    new_ranks_sorted = sorted(new_scores[n] for n in block_nodes)
    valid = (orig_ranks_sorted == new_ranks_sorted)

    if debug:
        _log(
            f"Done solver | solver_improved={int(bool(solver_improved))} "
            f"| FW_block_after={fw_block_after:.6f} | ΔFW_block={local_delta_fw:+.6f} "
            f"| rank_multiset_ok={int(valid)}"
        )

    elapsed = time.time() - t0

    if not valid:
        # important correctness failure => log one line even if debug=False
        _log("❌ INVALID: rank multiset mismatch (not a permutation of original ranks).")
        return {
            "block_id": worker_id,
            "worker_id": worker_id,
            "pid": pid,
            "seed_sid": seed_sid,
            "r_low": float(r_low),
            "r_high": float(r_high),
            "n_block": len(block_nodes),
            "valid": False,
            "why_invalid": "rank_multiset_mismatch",
            "block_ranks": {n: new_scores[n] for n in block_nodes},
            "fw_block_before": float(fw_block_before),
            "fw_block_after": float(fw_block_after),
            "local_delta_fw": float(local_delta_fw),
            "solver_improved": bool(solver_improved),
            "elapsed_worker": float(elapsed),
            "dp_exact_max_size": int(dp_exact_max_size),
        }

    # Valid proposal
    return {
        "block_id": worker_id,
        "worker_id": worker_id,
        "pid": pid,
        "seed_sid": seed_sid,
        "r_low": float(r_low),
        "r_high": float(r_high),
        "n_block": len(block_nodes),
        "valid": True,
        "why_invalid": "",
        "block_ranks": {n: new_scores[n] for n in block_nodes},
        "fw_block_before": float(fw_block_before),
        "fw_block_after": float(fw_block_after),
        "local_delta_fw": float(local_delta_fw),
        "solver_improved": bool(solver_improved),
        "elapsed_worker": float(elapsed),
        "dp_exact_max_size": int(dp_exact_max_size),
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

    # ---------------- knobs ----------------
    dp_exact_max_size=18,          # subset-DP exact solver for SCC sizes up to this (if available)
    large_scc_ls_iters=200,        # local-search budget for large SCCs
    eps_improve=1e-12,             # numerical tolerance for “improved”

    # ---------------- logging knobs ----------------
    log_path=None,                 # REQUIRED if debug=True (so we can log)
    log_solvers_always=False,      # if True, print solver summary even when debug=False
    log_only_when_no_gain=True,    # if True, solver summary prints only when no global improvement

    # ---------------- NEW: deadline ----------------
    deadline_ts: float | None = None,
):
    """
    DEADLINE-SAFE + MONOTONE (Option 2):
      - For each SCC, baseline order is original order within SCC.
      - Solvers propose a candidate order.
      - ACCEPT candidate only if INTERNAL FW within that SCC is >= baseline_INTERNAL_FW + eps_improve.
        Otherwise revert to baseline order for that SCC.

    Deadline behavior:
      - If deadline hits (or is too close), we stop solving further SCCs and keep baseline orders
        for the remaining SCCs. This preserves correctness/monotonicity.
      - Brute-force permutation enumeration is interrupted safely (baseline is always feasible).
      - Global FW computation at the end is skipped if deadline is too close (worker/controller
        usually does exact global checks anyway).
    """
    import time
    import inspect
    import networkx as nx
    from itertools import permutations
    from collections import Counter

    if debug and log_path is None:
        raise TypeError("refine_block_scc_interval: log_path is required when debug=True.")

    def _log(msg: str):
        if log_path is not None:
            log_message(msg, log_path)

    # ---------------- deadline helpers ----------------
    if deadline_ts is not None:
        try:
            deadline_ts = float(deadline_ts)
        except Exception:
            deadline_ts = None

    def _deadline_hit() -> bool:
        return (deadline_ts is not None) and (time.time() >= deadline_ts)

    def _time_left() -> float:
        if deadline_ts is None:
            return float("inf")
        return float(deadline_ts - time.time())

    # ---------------- internal FW helper on induced SCC subgraph ----------------
    def _internal_fw_of_order(subG_local, order_list):
        pos = {n: i for i, n in enumerate(order_list)}
        wsum = 0.0
        for (u, v) in subG_local.edges():
            pu = pos.get(u, None)
            pv = pos.get(v, None)
            if pu is None or pv is None:
                continue
            if pu < pv:
                wsum += float(subG_local[u][v].get("weight", 1.0))
        return wsum

    # ---------------- safe call with optional deadline ----------------
    def _call_with_deadline(fn, *a, **kw):
        """
        If the callee supports deadline_ts, pass it. Otherwise call without.
        """
        if deadline_ts is None:
            return fn(*a, **kw)
        try:
            sig = inspect.signature(fn)
            if "deadline_ts" in sig.parameters:
                kw = dict(kw)
                kw["deadline_ts"] = float(deadline_ts)
        except Exception:
            pass
        return fn(*a, **kw)

    block_nodes = list(block_nodes)

    if debug:
        _log(f"[FUNC2][refine_block] start | block_size={len(block_nodes)} | time_left={_time_left():.3f}s")

    # Sort block nodes by current rank for stability
    block_nodes = sorted(block_nodes, key=lambda n: node_order[n])

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

    # SCC DAG (condensation on the induced subgraph)
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

    # ---- solver usage counters ----
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

    # ---- Option 2 extra counters ----
    rejected_cnt = 0
    accepted_cnt = 0
    accepted_strict_cnt = 0
    equal_cnt = 0

    # Construct new order for the block
    new_block_order = []
    for scc_id in scc_order:
        # If time is up, append baseline orders for the rest and stop.
        if _deadline_hit():
            if debug:
                _log(f"[FUNC2][refine_block] deadline hit mid-loop -> append remaining SCCs as baseline and stop.")
            # append baseline for this SCC and remaining SCCs
            remaining_ids = [scc_id] + [x for x in scc_order if x > scc_id and x not in set([scc_id])]
            # The above “remaining_ids” isn’t guaranteed in topo order; easier:
            # just continue with baseline-only by setting a flag and not solving further.
            # We'll do: append baseline for current SCC, then mark deadline_ts hit and keep baseline for later SCCs.
            # (Implementation below uses a simple flag.)
            pass

        scc_set = sub_sccs[scc_id]
        scc_nodes = list(scc_set)
        scc_size = len(scc_nodes)
        if scc_size > max_scc_size_seen:
            max_scc_size_seen = scc_size

        base_order = sorted(scc_nodes, key=lambda n: prev_order[n])

        if scc_size == 1:
            single_cnt += 1
            new_block_order.extend(base_order)
            continue

        # Build induced SCC subgraph once
        subG_scc = subG.subgraph(scc_nodes).copy()

        # Baseline internal FW
        base_fw = _internal_fw_of_order(subG_scc, base_order)

        # Default candidate = baseline (safe)
        cand_order = list(base_order)
        cand_fw = float(base_fw)
        used_solver = False

        # If deadline is too close, skip any solver work (baseline is already correct).
        # (Tune this: 0.02s is basically “don’t start anything”.)
        if deadline_ts is not None and _time_left() < 0.02:
            fallback_cnt += 1
            used_solver = False
        else:
            # -------- Small SCC: brute force (exact) --------
            if brute_force_min_size <= scc_size <= brute_force_max_size:
                brute_cnt += 1
                used_solver = True
                if debug:
                    _log(f"[FUNC2][block_scc] id={scc_id} size={scc_size} solver=bruteforce")

                # Always keep baseline as a feasible best so far
                best_perm = tuple(base_order)
                best_fw = float(base_fw)

                # Enumerate all permutations; interrupt safely on deadline
                for k, perm in enumerate(permutations(base_order)):
                    if (k & 1023) == 0 and _deadline_hit():  # check every 1024 perms
                        if debug:
                            _log(f"[FUNC2][block_scc] id={scc_id} bruteforce interrupted by deadline after {k} perms")
                        break
                    w_sum = _internal_fw_of_order(subG_scc, perm)
                    if w_sum > best_fw + float(eps_improve):
                        best_fw = float(w_sum)
                        best_perm = perm

                cand_order = list(best_perm)
                cand_fw = float(best_fw)

                if debug:
                    _log(
                        f"[FUNC2][block_scc] id={scc_id} size={scc_size} "
                        f"bruteforce_best_internalFW={cand_fw:.2f} base_internalFW={base_fw:.2f}"
                    )

            else:
                # -------- Mid SCC: try exact subset-DP if available --------
                if dp_exact_max_size is not None and scc_size <= dp_exact_max_size and not _deadline_hit():
                    try:
                        # IMPORTANT: pass subG_scc (not subG) if your DP expects the SCC-subgraph
                        best_order, best_fw = _call_with_deadline(_max_fw_order_dp, subG_scc, base_order)
                        dp_cnt += 1
                        used_solver = True
                        cand_order = list(best_order)
                        cand_fw = float(best_fw)
                        if debug:
                            _log(
                                f"[FUNC2][block_scc] id={scc_id} size={scc_size} "
                                f"solver=dp_exact internalFW={cand_fw:.2f} base_internalFW={base_fw:.2f}"
                            )
                    except NameError:
                        dp_failed_cnt += 1
                        used_solver = False
                    except Exception as e:
                        dp_failed_cnt += 1
                        used_solver = False
                        if debug:
                            _log(f"[FUNC2][block_scc] id={scc_id} size={scc_size} dp_exact_failed err={type(e).__name__}:{e}")

                # -------- Large SCC: heuristic + optional local search --------
                if not used_solver and not _deadline_hit():
                    try:
                        net = _scc_net_out_in_score(subG_scc, base_order)
                        base_heur = sorted(
                            base_order,
                            key=lambda n: (-net.get(n, 0.0), prev_order[n]),
                        )

                        base_heur_fw = _internal_fw_of_order(subG_scc, base_heur)

                        # If local search exists, improve from base_heur
                        try:
                            # If _local_search_scc supports deadline_ts, pass it.
                            base2, base2_fw = _call_with_deadline(_local_search_scc, subG_scc, base_heur, iters=int(large_scc_ls_iters))
                            heur_ls_cnt += 1
                            used_solver = True
                            cand_order = list(base2)
                            cand_fw = float(base2_fw)
                            if debug:
                                _log(
                                    f"[FUNC2][block_scc] id={scc_id} size={scc_size} "
                                    f"solver=heuristic+LS internalFW={cand_fw:.2f} base_internalFW={base_fw:.2f} iters={int(large_scc_ls_iters)}"
                                )
                        except NameError:
                            ls_missing_cnt += 1
                            heur_only_cnt += 1
                            used_solver = True
                            cand_order = list(base_heur)
                            cand_fw = float(base_heur_fw)
                            if debug:
                                _log(
                                    f"[FUNC2][block_scc] id={scc_id} size={scc_size} "
                                    f"solver=heuristic_only internalFW={cand_fw:.2f} base_internalFW={base_fw:.2f} ls_helper_missing"
                                )

                    except NameError:
                        helpers_missing_cnt += 1
                        fallback_cnt += 1
                        used_solver = False
                        # baseline already set

        # -------- Option 2 enforcement: never worse than baseline --------
        if cand_fw < base_fw - float(eps_improve):
            rejected_cnt += 1
            if debug:
                _log(
                    f"[FUNC2][block_scc] id={scc_id} size={scc_size} REJECT "
                    f"(candFW={cand_fw:.2f} < baseFW={base_fw:.2f}) -> revert baseline"
                )
            cand_order = list(base_order)
            cand_fw = float(base_fw)
        else:
            accepted_cnt += 1
            if cand_fw > base_fw + float(eps_improve):
                accepted_strict_cnt += 1
            else:
                equal_cnt += 1

        new_block_order.extend(cand_order)

        # If deadline hit right after finishing this SCC, we’ll keep baseline for remaining SCCs
        # by simply skipping solvers (the loop already does that when time_left < 0.02).
        # (No extra action needed.)

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

    # --- Global FW before/after (optional; can be expensive) ---
    # Your controller does exact global checks anyway. If deadline is tight, skip.
    existing_fw = None
    new_fw = None
    improved = False

    # Only compute if we have time (tune this threshold as needed).
    if deadline_ts is None or _time_left() >= 0.05:
        existing_fw = compute_forward_weight(edges, prev_order)
        new_fw = compute_forward_weight(edges, new_node_order)
        improved = (new_fw > existing_fw + float(eps_improve))
        delta = (new_fw - existing_fw)
    else:
        # Safe fallback: we can’t confirm global improvement under deadline pressure.
        existing_fw = 0.0
        new_fw = 0.0
        improved = False
        delta = 0.0

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
            f"accepted={accepted_cnt} acceptedStrict={accepted_strict_cnt} equal={equal_cnt} rejected={rejected_cnt} "
            f"ΔFW={delta:.2f} improved={int(improved)} time_left={_time_left():.3f}s"
        )

    if debug:
        _log(
            f"[FUNC2][refine_block] "
            f"FW_before={existing_fw:.2f} FW_after={new_fw:.2f} ΔFW={delta:.2f} improved={int(improved)} "
            f"time_left={_time_left():.3f}s"
        )

    return new_node_order, existing_fw, new_fw, improved




    
def _apply_block_to_scores(scores, block_ranks):
    """
    Apply a proposed re-ranking for a block.

    IMPORTANT SEMANTICS (matches your func2 usage):
      - DOES NOT mutate `scores` in-place (so you can do speculative checks like:
            cand_scores = _apply_block_to_scores(current_scores, block_ranks)
        without corrupting current_scores).
      - `block_ranks` is a dict {node: new_absolute_rank}.
      - Returns a NEW dict.

    Parameters
    ----------
    scores : dict
        Global ranking map {node: rank}.
    block_ranks : dict
        Proposed ranks for a subset of nodes {node: new_rank}.

    Returns
    -------
    dict
        New scores dict with the block update applied.
    """
    if not block_ranks:
        return scores.copy()

    new_scores = scores.copy()
    for n, r in block_ranks.items():
        new_scores[n] = r
    return new_scores

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
    debug=False,
    max_no_improvement_batches=30,
    max_empty_batch_builds=15,
    log_path=None,

    # adaptive sizing knobs
    block_frac_min=0.01,
    block_frac_max=0.08,
    blocks_per_core_target=2.0,
    max_block_abs=5000,
    min_block_abs=50,

    # avoid retry bad regions
    cooldown_batches=25,
    cold_if_delta_fw_le=-1e-9,
    good_interval_factor=2.0,
    exploit_prob=0.45,
    ucb_prob=0.35,
    random_prob=0.20,

    # monotonicity / tolerances
    eps_improve_fw=1e-9,            # "meaningful improvement" threshold
    eps_bw_increase=1e-12,          # if BW increases more than this => RuntimeError
    best_save_eps_bw=1e-9,

    # deadlines
    deadline_ts: float | None = None,
    POOL_ASYNC_POLL_SEC: float = 0.5,
    MAX_BUILD_SEC: float | None = None,

    # processors cap (so you can run on 48 even if node has 128)
    num_procs_cap: int | None = None,

    # solver knobs forwarded to worker
    dp_exact_max_size: int = 18,
    large_scc_ls_iters: int = 200,
    worker_eps_improve: float = 1e-12,

    # ---- QUIET logging knob ----
    LOG_EVERY_BATCHES: int = 25,

    # ---- prefer large intervals ----
    BIG_INTERVAL_BIAS_GAMMA: float = 3.0,      # >1 => pushes sizes toward block_frac_max
    BIG_INTERVAL_TARGET_BLOCKS_SCALE: float = 0.75,  # fewer blocks when blocks are larger
    BIG_INTERVAL_EXPLOIT_WIDTH_MULT_MAX: float = 2.2, # allow widening on exploit

    # ---- NEW: cap DP-enabled blocks per batch ----
    MAX_DP_BLOCKS_PER_BATCH: int = 20,         # <= your requirement

    # ---- NEW: ML logging (optional) ----
    ml_csv_path: str | None = None,
):
    """
    func2 with BW-monotonicity enforced:
      - Always execute block solver on selected intervals.
      - Always APPLY a block move if it does NOT increase BW (plateau moves OK).
      - If BW increases: log compact diagnostic + raise RuntimeError.

    DEADLINE-SAFE:
      - Never starts a new batch if too little time remains.
      - Any time the deadline hits mid-batch (build / pool / apply), it safely exits:
          * If in the middle of APPLY: it discards partial batch (reverts to scores_before_batch)
            so we never commit unverified changes right at timeout.
      - Uses fw_current/bw_current for the batch baseline (no redundant full FW recompute).
    """
    import time, os, math, random
    import multiprocessing as mp
    import networkx as nx
    import pandas as pd
    from collections import deque
    from bisect import bisect_left

    if log_path is None:
        raise TypeError("parallel_refine_largest_scc_intervals: log_path is required.")

    # ---------------- logging (quiet) ----------------
    def _imp(msg: str):
        log_message(f"[FUNC2] {msg}", log_path)

    def _dbg(msg: str):
        if debug:
            log_message(f"[FUNC2][DBG] {msg}", log_path)

    def _time_left_sec():
        if deadline_ts is None:
            return float("inf")
        return float(deadline_ts - time.time())

    def _deadline_hit():
        return (deadline_ts is not None) and (time.time() >= float(deadline_ts))

    def _fatal(title: str, **kv):
        _imp(f"❌ FATAL: {title}")
        for k, v in kv.items():
            try:
                _imp(f"❌ {k}={v}")
            except Exception:
                _imp(f"❌ {k}=(unprintable)")
        raise RuntimeError(title)

    # ---------------- ML helper ----------------
    def _ml_row_base(**kw):
        row = {
            "phase": "func2",
            "ts": time.time(),
            "batch": int(batch_idx),
        }
        row.update(kw)
        return row

    # ---- normalize probs
    if exploit_prob + ucb_prob + random_prob <= 0:
        raise ValueError("exploit_prob + ucb_prob + random_prob must be > 0.")
    sprob = exploit_prob + ucb_prob + random_prob
    exploit_p = exploit_prob / sprob
    ucb_p = ucb_prob / sprob
    # random prob is implicit

    # ---- processors
    try:
        detected = len(os.sched_getaffinity(0))
    except Exception:
        detected = os.cpu_count() or 1
    num_procs = max(1, int(detected))
    if num_procs_cap is not None:
        num_procs = max(1, min(num_procs, int(num_procs_cap)))

    # ---- read graph + scores
    edges_indexed, node_to_index, index_to_node = read_graph(csv_path)
    scores = load_initial_scores(initial_ranking_path, node_to_index)

    n_nodes = len(scores)
    m_edges = len(edges_indexed)
    total_weight = float(sum(float(w) for (_, _, w) in edges_indexed))

    # ---- cadence autos (quiet)
    if verify_every is None or verify_every <= 0:
        verify_every = int(200 * max(1.0, math.sqrt(m_edges / 2e5)))
        verify_every = max(100, min(5000, verify_every))

    if save_every is None or save_every <= 0:
        save_every = int(3 * max(1.0, math.sqrt(n_nodes / 2e4)))
        save_every = max(1, min(50, save_every))

    if output_path is None:
        output_path = initial_ranking_path.replace(".csv", "_scc_opt.csv")

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

    # ---- biased sampler toward large fractions in [min,max]
    def _sample_block_frac(bias_gamma: float, uniform_fallback: bool = False) -> float:
        lo = float(block_frac_min)
        hi = float(block_frac_max)
        if hi <= lo:
            return lo
        if uniform_fallback or bias_gamma <= 1.0:
            u = random.random()
            return lo + (hi - lo) * u
        u = random.random()
        v = 1.0 - (1.0 - u) ** float(bias_gamma)
        return lo + (hi - lo) * v

    # ---- ensure uniqueness
    if not all_scores_unique(scores):
        _imp("⚠️ Initial scores not unique -> reindex to enforce uniqueness.")
        _reindex_to_unique(scores)

    # ---- build graph
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges_indexed)

    _imp(
        f"Start | procs={num_procs} (cap={num_procs_cap}) | "
        f"verify_every={verify_every} save_every={save_every} | "
        f"MAX_DP_BLOCKS_PER_BATCH={MAX_DP_BLOCKS_PER_BATCH} | "
        f"BIG_INTERVAL_BIAS_GAMMA={BIG_INTERVAL_BIAS_GAMMA} | "
        f"eps_bw_increase={eps_bw_increase} | "
        f"deadline={'None' if deadline_ts is None else f'{float(deadline_ts):.0f}'}"
    )

    if _deadline_hit():
        _imp("Stop immediately: deadline already reached.")
        _save_best(scores, output_path)
        bw0 = float(total_weight - compute_forward_weight(edges_indexed, scores))
        return scores, [bw0], set(), output_path

    # ---- SCC topo reorder (contiguity)
    scores = global_scc_topo_reorder_scores(
        G, edges_indexed, scores,
        log_path=log_path, debug=debug,
        total_weight=total_weight
    )
    if not all_scores_unique(scores):
        _reindex_to_unique(scores)

    # ---- recompute objective AFTER reorder (important!)
    fw_current = float(compute_forward_weight(edges_indexed, scores))
    bw_current = float(total_weight - fw_current)
    bw_history = [bw_current]

    best_bw_global = bw_current
    best_scores_global = scores.copy()
    last_saved_best_bw = best_bw_global

    # ---- SCC decomposition (nontrivial SCCs)
    sccs = list(nx.strongly_connected_components(G))
    scc_list = [set(s) for s in sccs if len(s) >= 2]
    scc_list.sort(key=len, reverse=True)
    if not scc_list:
        _imp("Done: no nontrivial SCCs.")
        _save_best(scores, output_path)
        return scores, bw_history, set(), output_path

    _imp(f"SCCs | total={len(sccs)} nontrivial={len(scc_list)} largest={len(scc_list[0])}")

    # ---- node->sccid and edges_in_scc
    node_to_sccid = {}
    for sid, nodeset in enumerate(scc_list):
        for n in nodeset:
            node_to_sccid[n] = sid

    edges_in_scc = [[] for _ in range(len(scc_list))]
    for (u, v, w) in edges_indexed:
        su = node_to_sccid.get(u)
        if su is not None and su == node_to_sccid.get(v):
            edges_in_scc[su].append((u, v, w))

    def _rank_interval_has_backward_edge(edges_local, scores_loc, r_low, r_high):
        for (u, v, _w) in edges_local:
            ru = scores_loc[u]
            rv = scores_loc[v]
            if r_low <= ru <= r_high and r_low <= rv <= r_high and ru > rv:
                return True
        return False

    # ---- per-SCC state
    def _init_scc_state(sid, nodeset, scores_now):
        ranks = [scores_now[n] for n in nodeset]
        mn, mx = min(ranks), max(ranks)
        span = int(mx - mn + 1)

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

    scc_state = {sid: _init_scc_state(sid, scc_list[sid], scores) for sid in range(len(scc_list))}

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

    def _in_cooldown(st, r_low, r_high, current_batch):
        for (cl, ch, bidx) in st["cold_intervals"]:
            if current_batch - bidx <= cooldown_batches:
                if _interval_overlaps(r_low, r_high, cl, ch):
                    return True
        return False

    # ---- objective tracking
    current_scores = scores.copy()
    f2b_edges_global = set()  # kept for API compatibility

    batch_idx = 0
    no_improve_batches_global = 0
    empty_build_streak = 0

    _imp(f"Initial BW={bw_current:.6f}")

    # When very little time remains, refuse to start a new batch
    MIN_SEC_TO_START_BATCH = 1.0

    # ================= MAIN LOOP =================
    while True:
        if _deadline_hit():
            _imp(f"Stop: deadline reached (time_left={_time_left_sec():.3f}s).")
            break

        if len(f2b_edges_global) >= max_backward_flips:
            _imp(f"Stop: reached max_backward_flips={max_backward_flips}.")
            break

        active_sids = [sid for sid, st in scc_state.items() if st["active"]]
        if not active_sids:
            _imp("Stop: all SCCs inactive.")
            break

        if max_no_improvement_batches is not None and no_improve_batches_global >= max_no_improvement_batches:
            _imp(f"Stop: no BW improvement in {no_improve_batches_global} consecutive batches (patience={max_no_improvement_batches}).")
            break

        if max_empty_batch_builds is not None and empty_build_streak >= max_empty_batch_builds:
            _imp(f"Stop: failed to build any valid batch for {empty_build_streak} attempts (patience={max_empty_batch_builds}).")
            break

        if deadline_ts is not None and _time_left_sec() < MIN_SEC_TO_START_BATCH:
            _imp(f"Stop: not enough time left to start another batch (time_left={_time_left_sec():.3f}s).")
            break

        snapshot_scores = current_scores  # stable snapshot for building + workers
        cache = {}

        # global order arrays (for global slicing)
        global_nodes_sorted = sorted(snapshot_scores.keys(), key=lambda n: snapshot_scores[n])
        global_ranks_sorted = [snapshot_scores[n] for n in global_nodes_sorted]

        # ---- fewer blocks when blocks are larger
        target_blocks = int(
            max(
                1,
                min(
                    num_procs,
                    round(num_procs * blocks_per_core_target * float(BIG_INTERVAL_TARGET_BLOCKS_SCALE) / 2.0)
                )
            )
        )

        batch_blocks = []
        used_rank_ranges = []

        # block_id -> meta (for ML logging)
        block_meta = {}

        weights = [scc_state[sid]["size"] for sid in active_sids]
        max_attempts = num_procs * 90
        attempts = 0
        build_t0 = time.time()

        # deadline-aware build cap (optional clamp)
        build_cap = MAX_BUILD_SEC
        if deadline_ts is not None:
            # never allow build to consume essentially all remaining time
            build_cap = min(float(build_cap) if build_cap is not None else float("inf"),
                            max(0.0, 0.50 * _time_left_sec()))

        while len(batch_blocks) < target_blocks and attempts < max_attempts:
            attempts += 1
            if _deadline_hit():
                break
            if build_cap is not None and (time.time() - build_t0) >= float(build_cap):
                break

            sid = random.choices(active_sids, weights=weights, k=1)[0]
            st = scc_state[sid]

            if st["no_improve_batches"] >= max(5, max_no_improvement_batches or 30):
                st["active"] = False
                continue

            cached = _get_scc_cache(sid, snapshot_scores, cache)
            scc_nodes_sorted = cached["nodes_sorted"]
            scc_ranks_sorted = cached["ranks_sorted"]
            bucket_to_indices = cached["bucket_to_indices"]
            n_scc = st["size"]
            if n_scc <= 1:
                st["active"] = False
                continue

            uniform_fallback = (empty_build_streak >= 1) or (attempts > max_attempts * 0.70)

            frac = _sample_block_frac(BIG_INTERVAL_BIAS_GAMMA, uniform_fallback=uniform_fallback)
            size_target = int(round(frac * n_scc))
            size_target = max(int(min_block_abs), size_target)
            if max_block_abs is not None:
                size_target = min(int(max_block_abs), size_target)
            size_target = min(size_target, n_nodes)
            if size_target < 2:
                continue

            r = random.random()
            center_rank = None

            if st["good_intervals"] and r < exploit_p:
                base_r_low, base_r_high, _gain, _bidx = random.choice(st["good_intervals"])
                width = max(1.0, base_r_high - base_r_low)
                r_center = (base_r_low + base_r_high) / 2.0 + random.uniform(-0.20, 0.20) * width
                center_rank = float(r_center)
            elif r < exploit_p + ucb_p:
                b = _pick_bucket_index(st, bucket_to_indices)
                if b is None:
                    continue
                idxs = bucket_to_indices[b]
                if not idxs:
                    continue
                center_idx = random.choice(idxs)
                center_rank = float(scc_ranks_sorted[center_idx])
            else:
                center_idx = random.randint(0, n_scc - 1)
                center_rank = float(scc_ranks_sorted[center_idx])

            if center_rank is None:
                continue

            center_pos = bisect_left(global_ranks_sorted, center_rank)
            gL = max(0, int(center_pos) - int(size_target // 2))
            gR = min(n_nodes, gL + int(size_target))
            gL = max(0, gR - int(size_target))
            if gR - gL < 2:
                continue

            r_low = float(global_ranks_sorted[gL])
            r_high = float(global_ranks_sorted[gR - 1])
            if not (r_high > r_low):
                continue

            if _in_cooldown(st, r_low, r_high, batch_idx + 1):
                continue
            if any(_interval_overlaps(r_low, r_high, rl, rh) for (rl, rh) in used_rank_ranges):
                continue

            if not _rank_interval_has_backward_edge(edges_in_scc[sid], snapshot_scores, r_low, r_high):
                continue

            block_nodes = global_nodes_sorted[gL:gR]
            if len(block_nodes) < 2:
                continue

            blk = {"seed_sid": sid, "r_low": r_low, "r_high": r_high, "block_nodes": block_nodes}
            batch_blocks.append(blk)
            used_rank_ranges.append((r_low, r_high))

        if _deadline_hit():
            _imp("Stop: deadline hit during build.")
            break

        if not batch_blocks:
            empty_build_streak += 1
            continue
        empty_build_streak = 0

        if deadline_ts is not None and _time_left_sec() < MIN_SEC_TO_START_BATCH:
            _imp(f"Stop: not enough time left to EXECUTE batch (time_left={_time_left_sec():.3f}s).")
            break

        batch_idx += 1
        scores_before_batch = current_scores.copy()

        # ---- baseline: DO NOT recompute FW/BW from scratch; use tracked values
        fw_before = float(fw_current)
        bw_before = float(bw_current)

        # ---- DP budget per batch (do not mutate parameter)
        dp_enabled_ids = set()
        k_dp = max(0, min(int(MAX_DP_BLOCKS_PER_BATCH or 0), len(batch_blocks)))
        if k_dp > 0:
            dp_enabled_ids = set(range(k_dp))

        # ---- prepare worker args
        worker_args = []
        for i, blk in enumerate(batch_blocks):
            dp_cap_for_this_block = int(dp_exact_max_size) if (i in dp_enabled_ids) else 0  # 0 disables DP
            block_id = f"b{batch_idx}_{i}"

            block_meta[block_id] = {
                "seed_sid": int(blk.get("seed_sid", -1)) if blk.get("seed_sid", None) is not None else -1,
                "r_low": float(blk["r_low"]),
                "r_high": float(blk["r_high"]),
                "interval_len": int(len(blk["block_nodes"])),
                "dp_enabled": int(1 if (i in dp_enabled_ids) else 0),
            }

            worker_args.append((
                block_id,
                blk["block_nodes"],
                blk["r_low"],
                blk["r_high"],
                brute_force_min_size,
                brute_force_max_size,
                debug,
                log_path,
                int(dp_cap_for_this_block),
                int(large_scc_ls_iters),
                float(worker_eps_improve),
                blk.get("seed_sid", None),
            ))

        # ---- pool async (deadline-safe)
        pool_procs = min(num_procs, len(batch_blocks))
        results = []
        pool = None
        async_res = None
        t_exec0 = time.time()

        try:
            pool = mp.Pool(
                processes=pool_procs,
                initializer=_init_refine_worker,
                initargs=(G, edges_indexed, scores_before_batch),
            )
            async_res = pool.map_async(_worker_refine_block, worker_args)

            while True:
                if _deadline_hit():
                    _imp(f"Stop: deadline reached during pool (batch={batch_idx}) -> terminate.")
                    try:
                        pool.terminate()
                    except Exception:
                        pass
                    try:
                        pool.join()
                    except Exception:
                        pass
                    pool = None
                    results = []
                    break

                if async_res.ready():
                    results = async_res.get()
                    try:
                        pool.close()
                    except Exception:
                        pass
                    try:
                        pool.join()
                    except Exception:
                        pass
                    pool = None
                    break

                time.sleep(max(0.05, float(POOL_ASYNC_POLL_SEC)))
        finally:
            if pool is not None:
                try:
                    pool.terminate()
                except Exception:
                    pass
                try:
                    pool.join()
                except Exception:
                    pass

        if _deadline_hit():
            # No commit: keep best so far and exit
            break

        # ---- APPLY monotone (plateau allowed; crash if BW increases)
        random.shuffle(results)

        current_scores = scores_before_batch.copy()
        sum_delta_fw_applied = 0.0
        applied_blocks = 0
        ml_applied = 0

        for res in results:
            if _deadline_hit():
                # SAFETY: discard partial batch (unverified) and exit cleanly
                _imp(f"Stop: deadline hit during APPLY (batch={batch_idx}) -> discard partial batch and exit.")
                current_scores = scores_before_batch.copy()
                applied_blocks = 0
                sum_delta_fw_applied = 0.0
                break

            if not res:
                continue

            if not bool(res.get("valid", False)):
                _fatal(
                    "Worker returned invalid proposal",
                    batch=batch_idx,
                    block_id=res.get("block_id"),
                    why=res.get("why_invalid"),
                )

            block_id = res.get("block_id")
            meta = block_meta.get(block_id, {})

            block_ranks = res.get("block_ranks", {}) or {}
            local_delta_fw = float(res.get("local_delta_fw", 0.0))
            n_block = int(res.get("n_block", len(block_ranks)))
            if n_block < 2 or not block_ranks:
                if ml_csv_path is not None:
                    ml_append_row(
                        ml_csv_path,
                        _ml_row_base(
                            cand_type="interval_shuffle",
                            block_id=str(block_id),
                            seed_sid=int(meta.get("seed_sid", -1)),
                            r_low=float(meta.get("r_low", float("nan"))),
                            r_high=float(meta.get("r_high", float("nan"))),
                            interval_len=int(meta.get("interval_len", 0)),
                            dp_enabled=int(meta.get("dp_enabled", 0)),
                            n_block=int(n_block),
                            local_delta_fw=float(local_delta_fw),
                            applied=int(0),
                            note="empty_block_ranks",
                        ),
                        log_path,
                    )
                continue

            # If worker claims ΔFW negative enough, exact-check and crash
            if local_delta_fw < -float(eps_bw_increase):
                cand_scores = _apply_block_to_scores(current_scores, block_ranks)
                fw_cand = float(compute_forward_weight(edges_indexed, cand_scores))
                bw_cand = float(total_weight - fw_cand)

                if ml_csv_path is not None:
                    ml_append_row(
                        ml_csv_path,
                        _ml_row_base(
                            cand_type="interval_shuffle",
                            block_id=str(block_id),
                            seed_sid=int(meta.get("seed_sid", -1)),
                            r_low=float(meta.get("r_low", float("nan"))),
                            r_high=float(meta.get("r_high", float("nan"))),
                            interval_len=int(meta.get("interval_len", 0)),
                            dp_enabled=int(meta.get("dp_enabled", 0)),
                            n_block=int(n_block),
                            local_delta_fw=float(local_delta_fw),
                            applied=int(0),
                            bw_before_batch=float(bw_before),
                            bw_after_exact=float(bw_cand),
                            violates=int(1),
                            note="fatal_bw_increase_predicted",
                        ),
                        log_path,
                    )

                _fatal(
                    "BW would increase by applying block",
                    batch=batch_idx,
                    bw_before=bw_before,
                    bw_after_exact=bw_cand,
                    local_delta_fw=local_delta_fw,
                    n_block=n_block,
                    note="Monotonicity violated: invariants/solver must be wrong.",
                )

            # Accept block even if Δ≈0 (plateau move)
            current_scores = _apply_block_to_scores(current_scores, block_ranks)
            sum_delta_fw_applied += local_delta_fw
            applied_blocks += 1

            if ml_csv_path is not None:
                ml_append_row(
                    ml_csv_path,
                    _ml_row_base(
                        cand_type="interval_shuffle",
                        block_id=str(block_id),
                        seed_sid=int(meta.get("seed_sid", -1)),
                        r_low=float(meta.get("r_low", float("nan"))),
                        r_high=float(meta.get("r_high", float("nan"))),
                        interval_len=int(meta.get("interval_len", 0)),
                        dp_enabled=int(meta.get("dp_enabled", 0)),
                        n_block=int(n_block),
                        local_delta_fw=float(local_delta_fw),
                        applied=int(1),
                    ),
                    log_path,
                )
                ml_applied += 1

        if _deadline_hit():
            break

        if not all_scores_unique(current_scores):
            _reindex_to_unique(current_scores)

        # ---- hard exact verify (batch-level)
        fw_after_exact = float(compute_forward_weight(edges_indexed, current_scores))
        bw_after_exact = float(total_weight - fw_after_exact)

        if bw_after_exact > bw_before + float(eps_bw_increase):
            _fatal(
                "GLOBAL BW increased after batch (hard verify)",
                batch=batch_idx,
                bw_before=bw_before,
                bw_after_exact=bw_after_exact,
                fw_before=fw_before,
                fw_after_exact=fw_after_exact,
                sum_delta_fw_applied=sum_delta_fw_applied,
                applied_blocks=applied_blocks,
            )

        # accept tracking
        fw_current = fw_after_exact
        bw_current = bw_after_exact
        bw_history.append(bw_current)
        delta_fw_exact = fw_after_exact - fw_before

        if ml_csv_path is not None:
            ml_append_row(
                ml_csv_path,
                _ml_row_base(
                    cand_type="batch_summary",
                    bw_before=float(bw_before),
                    bw_after=float(bw_current),
                    delta_fw_exact=float(delta_fw_exact),
                    applied_blocks=int(applied_blocks),
                    dp_blocks=int(k_dp),
                    ml_rows_blocks=int(ml_applied),
                    dt_exec=float(time.time() - t_exec0),
                    time_left=float(_time_left_sec()),
                ),
                log_path,
            )

        improved = bw_current < best_bw_global - float(best_save_eps_bw)
        if improved:
            best_bw_global = bw_current
            best_scores_global = current_scores.copy()
            no_improve_batches_global = 0
        else:
            no_improve_batches_global += 1

        exec_dt = time.time() - t_exec0

        if improved:
            _imp(
                f"Batch {batch_idx}: NEW_BEST BW={bw_current:.6f} | ΔFW={delta_fw_exact:+.6f} "
                f"| blocks={len(batch_blocks)} applied={applied_blocks} dp_blocks={k_dp} "
                f"| dt={exec_dt:.2f}s left={_time_left_sec():.1f}s"
            )
        elif (LOG_EVERY_BATCHES and (batch_idx % int(LOG_EVERY_BATCHES) == 0)):
            _imp(
                f"Batch {batch_idx}: BW={bw_current:.6f} | ΔFW={delta_fw_exact:+.6f} "
                f"| blocks={len(batch_blocks)} applied={applied_blocks} dp_blocks={k_dp} "
                f"| noImp={no_improve_batches_global}/{max_no_improvement_batches} "
                f"| dt={exec_dt:.2f}s left={_time_left_sec():.1f}s"
            )

        if save_every and (batch_idx % int(save_every) == 0) and (best_bw_global < last_saved_best_bw - float(best_save_eps_bw)):
            _save_best(best_scores_global, output_path)
            last_saved_best_bw = best_bw_global
            _imp(f"Checkpoint: saved BEST (BestBW={best_bw_global:.6f}) -> {output_path}")

        if verify_every and (batch_idx % int(verify_every) == 0):
            fw_manual = float(compute_forward_weight(edges_indexed, current_scores))
            bw_manual = float(total_weight - fw_manual)
            if bw_manual > bw_current + float(eps_bw_increase):
                _fatal("Verification mismatch", batch=batch_idx, bw_tracked=bw_current, bw_manual=bw_manual)
            if debug:
                _imp(f"Verify {batch_idx}: BW OK (BW={bw_manual:.6f})")

    # ---- final save (best)
    try:
        _save_best(best_scores_global, output_path)
    except Exception as e:
        _imp(f"⚠️ Final save failed: {repr(e)}")

    _imp(
        f"Done | currentBW={bw_current:.6f} bestBW={best_bw_global:.6f} "
        f"| batches={batch_idx} | time_left={_time_left_sec():.1f}s"
    )

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
    h_hours: float = 72.0,      # used as the default wall-clock budget if no minutes stop is provided
    func2_f2b_limit: int = 200,
    func1_full_every: int = 50,
    func1_log_every_round: int = 10000,
    func1_bw_check_every_round: int = 500,   # BW not FW
    func2_verify_every: int = 5000,
    c: float = 1.0,             # kept for compatibility, IGNORED
    EPS_BW_IMPROVE: float = 1e-9,  # improvement threshold

    # ---------------- quiet + long-run knobs ----------------
    NEVER_STOP: bool = True,            # OK to leave True; we still enforce a hard stop by default
    HEARTBEAT_SEC: float = 6 * 3600.0,  # extremely sparse heartbeat (default 6 hours)
    IDLE_SLEEP_SEC: float = 0.5,

    # ---------------- WMSF seed + escape ----------------
    WMSF_AT_START: bool = False,
    WMSF_START_ORDERING: str = "L2",
    WMSF_ESCAPE_SEC: float = 3 * 60.0,
    WMSF_ESCAPE_MIN_GAP_SEC: float = 15 * 60.0,
    WMSF_ORDERINGS=("L2", "L1"),
    WMSF_MAX_SEC: float = 1800.0,
    WMSF_MAX_NODES: int = 2000,
    WMSF_POLISH_GLOBAL_DP_BATCHES: int = 8,

    # ---------------- hard wall-clock stop ----------------
    RUN_FOR_MINUTES: float = None,           # preferred
    STOP_AFTER_MINUTES: float = None,        # alias

    # ---------------- func2-on-empty behavior (kept) ----------------
    FUNC2_ON_EMPTY_SEC: float = 20.0,
    FUNC2_ON_EMPTY_MIN_SEC_LEFT: float = 3.0,

    # ---------------- func1 targeted DP budget per call ----------------
    FUNC1_DP_BATCHES_PER_CALL: int = 20,

    # ---------------- NEW: fair time slicing ----------------
    # Target fraction of (func1_time / (func1_time + func2_time)).
    # 0.50 => equal time. 0.65 => func1 gets 65% of time.
    FUNC1_TARGET_TIME_FRACTION: float = 0.50,

    # Wall-clock time slice per call (seconds). Both phases get time-boxed.
    FUNC1_SLICE_SEC: float = 60.0,
    FUNC2_SLICE_SEC: float = 60.0,

    # Hysteresis to avoid flip-flopping decisions due to tiny differences.
    FAIRNESS_EPS: float = 0.03,   # 3% slack
):
    """
    Hybrid loop (func2 ↔ func1 targeted) + optional WMSF seed/escape.

    HARD REQUIREMENTS ENFORCED HERE:
      - One global wall-clock budget (RUN_FOR_MINUTES / STOP_AFTER_MINUTES / h_hours fallback).
      - Every expensive phase call gets a strict per-call deadline clamped by the global deadline.
      - Fair scheduling by cumulative time: time_f1 / (time_f1 + time_f2) ~= FUNC1_TARGET_TIME_FRACTION.
    """
    import os
    import time
    import shutil
    import inspect

    if log_path is None:
        raise TypeError("hybrid_refine_func1_func2: log_path is required.")

    # ------------------------------
    # Resolve minutes stop (alias) + default behavior
    # ------------------------------
    if RUN_FOR_MINUTES is None and STOP_AFTER_MINUTES is not None:
        RUN_FOR_MINUTES = STOP_AFTER_MINUTES

    if RUN_FOR_MINUTES is not None and STOP_AFTER_MINUTES is not None:
        try:
            RUN_FOR_MINUTES = min(float(RUN_FOR_MINUTES), float(STOP_AFTER_MINUTES))
        except Exception:
            pass

    if RUN_FOR_MINUTES is None:
        try:
            RUN_FOR_MINUTES = float(h_hours) * 60.0
        except Exception:
            RUN_FOR_MINUTES = 72.0 * 60.0

    # ------------------------------
    # Load graph once
    # ------------------------------
    edges, node_to_index, index_to_node = read_graph(csv_path)
    total_weight = float(sum(float(w) for (_, _, w) in edges))

    # ------------------------------
    # Ensure output directory exists
    # ------------------------------
    out_dir = os.path.dirname(output_ranking_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------
    # Global deadline
    # ------------------------------
    start_ts = time.time()
    global_deadline = start_ts + float(RUN_FOR_MINUTES) * 60.0

    def _now_str():
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def _sec_left():
        return max(0.0, global_deadline - time.time())

    def _time_exceeded():
        return time.time() >= global_deadline

    def _stop_if_needed(where: str):
        if _time_exceeded():
            log_message(
                f"[HYBRID] STOP reached at {_now_str()} | where={where} | "
                f"elapsed_h={(time.time()-start_ts)/3600.0:.3f} | "
                f"RUN_FOR_MINUTES={RUN_FOR_MINUTES} STOP_AFTER_MINUTES={STOP_AFTER_MINUTES} | "
                f"NEVER_STOP={NEVER_STOP} h_hours={h_hours}",
                log_path
            )
            return True
        return False

    def _deadline_for_slice(slice_sec: float):
        """
        Strict per-call deadline, always clamped by global_deadline.
        If no time left, returns time.time() (already exceeded).
        """
        now = time.time()
        if now >= global_deadline:
            return now
        # Clamp slice by remaining time to prevent any call from running past global.
        return min(global_deadline, now + max(0.0, float(slice_sec)), now + _sec_left())

    def _hours_left_until(deadline_ts: float):
        return max(0.0, (float(deadline_ts) - time.time()) / 3600.0)

    # ------------------------------
    # Very sparse heartbeat
    # ------------------------------
    last_heartbeat_ts = start_ts

    def heartbeat(msg: str):
        nonlocal last_heartbeat_ts
        now = time.time()
        if (now - last_heartbeat_ts) >= float(HEARTBEAT_SEC):
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

    def _phase_log_if_signal(phase_idx, phase_name, t0, bw_before, bw_after, extra=""):
        nonlocal best_bw_seen, last_best_ts
        dt = time.time() - t0
        dBW = bw_after - bw_before

        is_improve = (bw_before - bw_after) > float(EPS_BW_IMPROVE)
        is_new_best = bw_after + float(EPS_BW_IMPROVE) < best_bw_seen

        if is_new_best:
            best_bw_seen = bw_after
            last_best_ts = time.time()
            log_message(
                f"[HYBRID] phase={phase_idx:04d} {phase_name} | dt={dt:.2f}s | "
                f"BW {bw_before:.6f} -> {bw_after:.6f} (dBW={dBW:+.6f}) | "
                f"bestBW={best_bw_seen:.6f} [NEW_BEST] | sec_left={_sec_left():.1f} | {extra}",
                log_path
            )
        elif is_improve:
            log_message(
                f"[HYBRID] phase={phase_idx:04d} {phase_name} | dt={dt:.2f}s | "
                f"BW {bw_before:.6f} -> {bw_after:.6f} (dBW={dBW:+.6f}) | "
                f"bestBW={best_bw_seen:.6f} | sec_left={_sec_left():.1f} | {extra}",
                log_path
            )
        else:
            heartbeat(
                f"[HYBRID] heartbeat | phase={phase_idx:04d} | func1_runs={func1_runs} func2_runs={func2_runs} | "
                f"time_f1={time_spent_func1:.1f}s time_f2={time_spent_func2:.1f}s target_f1={FUNC1_TARGET_TIME_FRACTION:.2f} | "
                f"BW={bw_after:.6f} bestBW={best_bw_seen:.6f} | sec_left={_sec_left():.1f}"
            )

    def _count_backward_in_pairs(scores, pairs_iterable):
        b = 0
        for (u, v) in pairs_iterable:
            if scores[u] > scores[v]:
                b += 1
        return b

    def _all_current_backward_pairs(scores):
        out = set()
        for (u, v, _w) in edges:
            if scores[u] > scores[v]:
                out.add((u, v))
        return out

    def _call_with_optional_deadline(fn, kwargs, deadline_ts):
        """
        Inject deadline_ts if the function supports it.
        """
        try:
            sig = inspect.signature(fn)
            if deadline_ts is not None and "deadline_ts" in sig.parameters:
                kwargs = dict(kwargs)
                kwargs["deadline_ts"] = float(deadline_ts)
            return fn(**kwargs)
        except Exception:
            return fn(**kwargs)

    # ------------------------------
    # Fairness scheduler
    # ------------------------------
    def _pick_next_phase(cycle_remaining_nonempty: bool) -> str:
        """
        Return "func1" or "func2" based on cumulative time spent.
        Try to keep: time_spent_func1 / (time_spent_func1 + time_spent_func2) ~= FUNC1_TARGET_TIME_FRACTION.
        """
        total = time_spent_func1 + time_spent_func2
        if total <= 1e-9:
            return "func1"  # start with func1 by default

        frac_f1 = time_spent_func1 / total
        target = float(FUNC1_TARGET_TIME_FRACTION)
        eps = float(FAIRNESS_EPS)

        if frac_f1 > target + eps:
            return "func2"
        if frac_f1 < target - eps:
            return "func1"

        # Near target: prefer func1 if there's targeted work to do, else func2.
        return "func1" if cycle_remaining_nonempty else "func2"

    # ------------------------------
    # State tracking
    # ------------------------------
    phase_idx = 0
    func1_runs = 0
    func2_runs = 0
    current_input_path = initial_ranking_path
    best_bw_seen = float("inf")
    last_best_ts = start_ts
    last_escape_attempt_ts = 0.0  # for optional WMSF escape (if you later enable it)

    # fairness counters (wall-clock time spent inside the calls)
    time_spent_func1 = 0.0
    time_spent_func2 = 0.0

    # ------------------------------
    # Header
    # ------------------------------
    log_message(
        f"[HYBRID] START {_now_str()} | RUN_FOR_MINUTES={RUN_FOR_MINUTES} (~{RUN_FOR_MINUTES/60.0:.2f}h) | "
        f"h_hours={h_hours} NEVER_STOP={NEVER_STOP} | output={output_ranking_path} | "
        f"EPS_BW_IMPROVE={EPS_BW_IMPROVE} | HEARTBEAT_SEC={HEARTBEAT_SEC} | "
        f"FUNC1_DP_BATCHES_PER_CALL={FUNC1_DP_BATCHES_PER_CALL} | "
        f"FAIR target_f1={FUNC1_TARGET_TIME_FRACTION} slice_f1={FUNC1_SLICE_SEC}s slice_f2={FUNC2_SLICE_SEC}s",
        log_path
    )

    if _stop_if_needed("at_start"):
        return output_ranking_path

    # =====================================================
    # (0) WMSF seed at start (STRICTLY time-boxed + clamped)
    # =====================================================
    if WMSF_AT_START and (not _time_exceeded()):
        try:
            _, _, bw_init = bw_from_ranking_path(current_input_path)
            seed_path = output_ranking_path + ".wmsf_seed.csv"

            # Clamp WMSF time by global remaining time
            wmsf_sec = min(float(WMSF_MAX_SEC), _sec_left())
            if wmsf_sec > 0.25:  # tiny guard
                ok = maybe_run_wmsf_reseed(
                    csv_path=csv_path,
                    current_input_path=current_input_path,
                    output_ranking_path=seed_path,
                    log_path=log_path,
                    ordering=WMSF_START_ORDERING,
                    MAX_SEC=wmsf_sec,
                    MAX_NODES=WMSF_MAX_NODES,
                )

                if ok:
                    _, _, bw_seed = bw_from_ranking_path(seed_path)
                    if bw_seed + float(EPS_BW_IMPROVE) < bw_init:
                        shutil.copyfile(seed_path, output_ranking_path)
                        current_input_path = output_ranking_path
                        log_message(f"[HYBRID] WMSF_START accepted | BW {bw_init:.6f} -> {bw_seed:.6f}", log_path)
                    else:
                        log_message(f"[HYBRID] WMSF_START rejected | BW {bw_init:.6f} -> {bw_seed:.6f}", log_path)
        except Exception as e:
            log_message(f"[HYBRID] ERROR WMSF_START exception: {type(e).__name__}: {e}", log_path)

    if _stop_if_needed("after_wmsf_start"):
        return output_ranking_path

    # =====================================================
    # PHASE 1: func1 GLOBAL (initial) -- STRICT time-sliced
    # =====================================================
    phase_idx += 1
    func1_runs += 1
    t0 = time.time()

    _, _, bw_before = bw_from_ranking_path(current_input_path)

    f1_deadline = _deadline_for_slice(FUNC1_SLICE_SEC)
    if f1_deadline <= time.time():
        return output_ranking_path

    _call_with_optional_deadline(
        refine_ranking_parallel_dynamic,
        dict(
            csv_path=csv_path,
            initial_ranking_path=current_input_path,
            output_excel=output_ranking_path,
            log_path=log_path,
            MAX_HOURS=_hours_left_until(f1_deadline),
            LOG_EVERY_ROUND=func1_log_every_round,
            BW_CHECK_EVERY_ROUND=func1_bw_check_every_round,
            edge_subset=None,  # GLOBAL
        ),
        deadline_ts=f1_deadline,
    )

    time_spent_func1 += (time.time() - t0)

    _, _, bw_after = bw_from_ranking_path(output_ranking_path)
    _phase_log_if_signal(phase_idx, "func1_initial_global_slice", t0, bw_before, bw_after, extra="mode=GLOBAL(sliced)")
    current_input_path = output_ranking_path

    if _stop_if_needed("after_func1_initial_global"):
        return output_ranking_path

    # =====================================================
    # Cycle state
    # =====================================================
    cycle_id = 0
    cycle_remaining = set()  # pairs (u,v) not yet assessed in THIS cycle

    # =====================================================
    # MAIN LOOP
    # =====================================================
    try:
        while True:
            if _stop_if_needed("top_of_main_loop"):
                break

            # Load current scores/BW once per loop
            scores_now, _, bw_now = bw_from_ranking_path(current_input_path)

            # If globally no backward edges: idle and keep checking until time ends
            all_bw_pairs = _all_current_backward_pairs(scores_now)
            if len(all_bw_pairs) == 0:
                heartbeat(
                    f"[HYBRID] heartbeat | cycle_id={cycle_id} | no_backward_edges_global | "
                    f"time_f1={time_spent_func1:.1f}s time_f2={time_spent_func2:.1f}s | "
                    f"BW={bw_now:.6f} bestBW={best_bw_seen:.6f} | sec_left={_sec_left():.1f}"
                )
                if IDLE_SLEEP_SEC and IDLE_SLEEP_SEC > 0:
                    time.sleep(float(IDLE_SLEEP_SEC))
                continue

            # If cycle is empty OR has no currently-backward edges left, refresh it.
            backward_in_cycle = _count_backward_in_pairs(scores_now, cycle_remaining) if cycle_remaining else 0
            if (not cycle_remaining) or (backward_in_cycle == 0):
                # Optional legacy behavior: brief func2 burst before refilling (now it can actually trigger)
                if (not cycle_remaining) and _sec_left() >= float(FUNC2_ON_EMPTY_MIN_SEC_LEFT) and float(FUNC2_ON_EMPTY_SEC) > 0.0:
                    phase_idx += 1
                    func2_runs += 1
                    t0 = time.time()

                    _, _, bw_before2 = bw_from_ranking_path(current_input_path)
                    tmp_deadline = _deadline_for_slice(FUNC2_ON_EMPTY_SEC)
                    if tmp_deadline > time.time():
                        _call_with_optional_deadline(
                            parallel_refine_largest_scc_intervals,
                            dict(
                                max_backward_flips=func2_f2b_limit,
                                verify_every=func2_verify_every,
                                csv_path=csv_path,
                                initial_ranking_path=current_input_path,
                                output_path=output_ranking_path,
                                debug=False,
                                log_path=log_path,
                            ),
                            deadline_ts=tmp_deadline,
                        )

                    time_spent_func2 += (time.time() - t0)

                    _, _, bw_after2 = bw_from_ranking_path(output_ranking_path)
                    _phase_log_if_signal(phase_idx, f"func2_on_empty_{func2_runs}", t0, bw_before2, bw_after2, extra="")
                    current_input_path = output_ranking_path

                    # After func2, force a refresh next iteration
                    cycle_remaining.clear()
                    if _stop_if_needed("after_func2_on_empty"):
                        break
                    if IDLE_SLEEP_SEC and IDLE_SLEEP_SEC > 0:
                        time.sleep(float(IDLE_SLEEP_SEC))
                    continue

                # Refill cycle
                cycle_id += 1
                cycle_remaining = _all_current_backward_pairs(scores_now)

            cycle_nonempty = len(cycle_remaining) > 0

            # ------------------------------
            # Choose next phase fairly
            # ------------------------------
            next_phase = _pick_next_phase(cycle_remaining_nonempty=cycle_nonempty)

            # ------------------------------
            # Run func2 (STRICT time-sliced)
            # ------------------------------
            if next_phase == "func2":
                if _stop_if_needed("before_func2_fair"):
                    break

                phase_idx += 1
                func2_runs += 1
                t0 = time.time()

                _, _, bw_before2 = bw_from_ranking_path(current_input_path)

                f2_deadline = _deadline_for_slice(FUNC2_SLICE_SEC)
                if f2_deadline > time.time():
                    _call_with_optional_deadline(
                        parallel_refine_largest_scc_intervals,
                        dict(
                            max_backward_flips=func2_f2b_limit,
                            verify_every=func2_verify_every,
                            csv_path=csv_path,
                            initial_ranking_path=current_input_path,
                            output_path=output_ranking_path,
                            debug=False,
                            log_path=log_path,
                        ),
                        deadline_ts=f2_deadline,
                    )

                time_spent_func2 += (time.time() - t0)

                _, _, bw_after2 = bw_from_ranking_path(output_ranking_path)
                _phase_log_if_signal(
                    phase_idx, f"func2_fair_{func2_runs}", t0, bw_before2, bw_after2,
                    extra=f"time_f1={time_spent_func1:.1f}s time_f2={time_spent_func2:.1f}s"
                )
                current_input_path = output_ranking_path

                # After func2, backward set changes; refresh next iteration.
                cycle_remaining.clear()

                if _stop_if_needed("after_func2_fair"):
                    break

            # ------------------------------
            # Run func1 (targeted or global) (STRICT time-sliced)
            # ------------------------------
            else:
                if _stop_if_needed("before_func1_fair"):
                    break

                phase_idx += 1
                func1_runs += 1
                t0 = time.time()

                scores_before1, _, bw_before1 = bw_from_ranking_path(current_input_path)

                # Decide targeted vs periodic global refresh
                do_global = (func1_full_every > 0 and (func1_runs % func1_full_every == 0))
                edge_subset = None
                subset_list = None

                if not do_global:
                    subset_list = list(cycle_remaining)
                    edge_subset = subset_list

                f1_deadline = _deadline_for_slice(FUNC1_SLICE_SEC)
                if f1_deadline > time.time():
                    ret = _call_with_optional_deadline(
                        refine_ranking_parallel_dynamic,
                        dict(
                            csv_path=csv_path,
                            initial_ranking_path=current_input_path,
                            output_excel=output_ranking_path,
                            log_path=log_path,
                            MAX_HOURS=_hours_left_until(f1_deadline),
                            LOG_EVERY_ROUND=func1_log_every_round,
                            BW_CHECK_EVERY_ROUND=func1_bw_check_every_round,
                            edge_subset=edge_subset,
                            MAX_DP_BATCHES_PER_CALL=(int(FUNC1_DP_BATCHES_PER_CALL) if edge_subset is not None else None),
                        ),
                        deadline_ts=f1_deadline,
                    )
                else:
                    ret = None

                time_spent_func1 += (time.time() - t0)

                # Update cycle_remaining robustly based on return contract if available (TARGETED only).
                assessed_pairs = set()
                if (edge_subset is not None) and (subset_list is not None):
                    remaining_pairs_unassessed = None
                    try:
                        if isinstance(ret, tuple):
                            if len(ret) >= 3:
                                remaining_pairs_unassessed = ret[2]
                            elif len(ret) == 2:
                                remaining_pairs_unassessed = ret[1]
                        else:
                            remaining_pairs_unassessed = ret
                    except Exception:
                        remaining_pairs_unassessed = None

                    if remaining_pairs_unassessed is not None:
                        remaining_set = set(remaining_pairs_unassessed or [])
                        assessed_pairs = set(subset_list) - remaining_set
                        cycle_remaining.difference_update(assessed_pairs)
                    else:
                        # safest fallback: force refresh next iteration
                        cycle_remaining.clear()
                else:
                    # global pass: refresh cycle next iteration
                    cycle_remaining.clear()

                _, _, bw_after1 = bw_from_ranking_path(output_ranking_path)

                phase_name = "func1_global_fair" if edge_subset is None else "func1_targeted_fair"
                extra = (
                    f"mode={'GLOBAL' if edge_subset is None else 'TARGETED'} "
                    f"cycle_id={cycle_id} assessed={len(assessed_pairs)} remaining={len(cycle_remaining)} "
                    f"dp_batches={(int(FUNC1_DP_BATCHES_PER_CALL) if edge_subset is not None else 0)} | "
                    f"time_f1={time_spent_func1:.1f}s time_f2={time_spent_func2:.1f}s"
                )

                _phase_log_if_signal(phase_idx, f"{phase_name}_{func1_runs}", t0, bw_before1, bw_after1, extra=extra)
                current_input_path = output_ranking_path

                if _stop_if_needed("after_func1_fair"):
                    break

            if IDLE_SLEEP_SEC and IDLE_SLEEP_SEC > 0:
                time.sleep(float(IDLE_SLEEP_SEC))

    except KeyboardInterrupt:
        log_message(
            f"[HYBRID] INTERRUPTED {_now_str()} | phases={phase_idx} func1_runs={func1_runs} func2_runs={func2_runs} "
            f"cycle_id={cycle_id} bestBW={best_bw_seen:.6f} | "
            f"time_f1={time_spent_func1:.1f}s time_f2={time_spent_func2:.1f}s | "
            f"elapsed_h={(time.time()-start_ts)/3600.0:.3f}",
            log_path
        )
        return output_ranking_path

    log_message(
        f"[HYBRID] END {_now_str()} | phases={phase_idx} func1_runs={func1_runs} func2_runs={func2_runs} "
        f"cycle_id={cycle_id} bestBW={best_bw_seen:.6f} | "
        f"time_f1={time_spent_func1:.1f}s time_f2={time_spent_func2:.1f}s | "
        f"elapsed_h={(time.time()-start_ts)/3600.0:.3f}",
        log_path
    )
    return output_ranking_path



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


import heapq
import numpy as np
from collections import defaultdict
import pandas as pd

def run_adaptive_out_over_inplus1_dimacs(graph_file_path, output_prefix, log_path):
    """
    Builds an initial ranking using adaptive (out_w+1)/(in_w+1) peeling,
    but reads the graph from DIMACS via read_graph().
    Writes: output_prefix + "_ranking.csv" with columns: "Node ID", "Order".
    Prints nothing; logs only via log_message.
    """
    output_csv = output_prefix + "_ranking.csv"

    log_message("[ADAPT_INIT] Reading DIMACS graph via read_graph() ...", log_path)
    edges, node_to_index, index_to_node = read_graph(graph_file_path)
    n_nodes = len(node_to_index)
    n_edges = len(edges)
    log_message(f"[ADAPT_INIT] Loaded: n_nodes={n_nodes}, n_edges={n_edges}", log_path)

    # Build adjacency with original node IDs (strings)
    out_edges = defaultdict(list)   # u -> list[(v, w)]
    in_edges  = defaultdict(list)   # v -> list[(u, w)]
    out_w = defaultdict(float)
    in_w  = defaultdict(float)

    # edges from read_graph are usually (u_idx, v_idx, w)
    for e in edges:
        # support either tuple length 3 or 2 (if weight implicit)
        if len(e) == 3:
            u_idx, v_idx, w = e
        else:
            u_idx, v_idx = e
            w = 1.0
        u = str(index_to_node[u_idx]).strip()
        v = str(index_to_node[v_idx]).strip()
        w = float(w)

        out_edges[u].append((v, w))
        in_edges[v].append((u, w))
        out_w[u] += w
        in_w[v]  += w

    node_ids = [str(index_to_node[i]).strip() for i in range(n_nodes)]

    # Score: (out+1)/(in+1)
    score = {u: (out_w[u] + 1.0) / (in_w[u] + 1.0) for u in node_ids}
    heap = [(-score[u], u) for u in node_ids]
    heapq.heapify(heap)

    unranked = set(node_ids)
    ranking = []

    log_message("[ADAPT_INIT] Starting adaptive ranking loop ...", log_path)
    while heap:
        neg_s, u = heapq.heappop(heap)
        if u not in unranked:
            continue

        ranking.append(u)
        unranked.remove(u)

        if len(ranking) % 100000 == 0:
            log_message(f"[ADAPT_INIT] ranked={len(ranking)} last={u}", log_path)

        # Removing u changes neighbors' in/out weights
        for v, w in out_edges.get(u, []):
            if v in unranked:
                in_w[v] -= w
                score[v] = (out_w[v] + 1.0) / (in_w[v] + 1.0)
                heapq.heappush(heap, (-score[v], v))

        for v, w in in_edges.get(u, []):
            if v in unranked:
                out_w[v] -= w
                score[v] = (out_w[v] + 1.0) / (in_w[v] + 1.0)
                heapq.heappush(heap, (-score[v], v))

    if unranked:
        # Should be rare; but keep deterministic-ish behavior (still random shuffle)
        log_message(f"[ADAPT_INIT] WARNING: {len(unranked)} nodes unranked; appending shuffled.", log_path)
        remaining = list(unranked)
        np.random.shuffle(remaining)
        ranking.extend(remaining)

    log_message(f"[ADAPT_INIT] Finished: ranked {len(ranking)} nodes. Writing {output_csv}", log_path)

    df_out = pd.DataFrame({"Node ID": ranking, "Order": list(range(len(ranking)))})
    df_out.to_csv(output_csv, index=False)

    log_message(f"[ADAPT_INIT] Wrote adaptive init ranking CSV: {output_csv}", log_path)
    return output_csv

def _load_order_map_from_csv(ranking_csv_path: str) -> dict[str, int]:
    df = pd.read_csv(ranking_csv_path)
    if "Node ID" not in df.columns or "Order" not in df.columns:
        raise RuntimeError(f"Ranking CSV must have columns 'Node ID' and 'Order': {ranking_csv_path}")
    df["Node ID"] = df["Node ID"].astype(str).str.strip()
    df["Order"] = pd.to_numeric(df["Order"], errors="raise").astype(int)
    return dict(zip(df["Node ID"], df["Order"]))


def compute_backward_weight_from_dimacs_edges(edges, index_to_node, order_map: dict[str, int]) -> float:
    bw = 0.0
    for e in edges:
        if len(e) == 3:
            u_idx, v_idx, w = e
        else:
            u_idx, v_idx = e
            w = 1.0

        u = str(index_to_node[u_idx]).strip()
        v = str(index_to_node[v_idx]).strip()

        ru = order_map.get(u, None)
        rv = order_map.get(v, None)

        # Treat missing nodes as "bad" so baselines can't win by omission
        if ru is None or rv is None:
            bw += float(w)
            continue

        if ru > rv:
            bw += float(w)

    return bw


# 1) Define Base Directory and Input File
import os
import pandas as pd

# -------------------- paths --------------------
base_dir = "/mmfs1/home/sv96/Feedback-arc-set-paper/datasets/"
graph_file_path = os.path.join(base_dir, "connectome.d")

graph_basename = os.path.basename(graph_file_path).replace(".d", "")

# ✅ log file now includes "learning"
log_path = os.path.join(base_dir, f"{graph_basename}_dimacs_hybrid_phases_log_learning-48cpu.txt")

# clear log
open(log_path, "w").close()

log_message(f"Reading graph from {graph_file_path} ...", log_path)
edges, node_to_index, index_to_node = read_graph(graph_file_path)
log_message(f"Graph loaded: n_nodes={len(node_to_index)}, n_edges={len(edges)}", log_path)


# -------------------- helpers (no prints; logs only) --------------------


# -------------------- baseline 1: SCC init --------------------
log_message("Generating SCC-based initial ranking ...", log_path)
scc_ranking_dict = generate_scc_ranking(edges, index_to_node, log_path)

initial_scc_csv_path = os.path.join(base_dir, f"{graph_basename}_dimacs_scc_init_learning.csv")
data_rows = [{"Node ID": str(node_name).strip(), "Order": int(score)}
             for node_name, score in scc_ranking_dict.items()]
pd.DataFrame(data_rows).to_csv(initial_scc_csv_path, index=False)
log_message(f"SCC init written: {initial_scc_csv_path}", log_path)


# -------------------- baseline 2: adaptive out/(in+1) init (DIMACS) --------------------
init_prefix = os.path.join(base_dir, f"{graph_basename}_adaptive_out_over_inplus1_learning")
adaptive_csv_path = run_adaptive_out_over_inplus1_dimacs(
    graph_file_path=graph_file_path,
    output_prefix=init_prefix,
    log_path=log_path
)
# run_adaptive_out_over_inplus1_dimacs writes: f"{init_prefix}_ranking.csv"
# so adaptive_csv_path already includes "learning" in the filename.


# -------------------- score both baselines and choose --------------------
log_message("Scoring SCC init baseline (backward weight) ...", log_path)
bw_scc = compute_backward_weight_from_dimacs_edges(
    edges, index_to_node, _load_order_map_from_csv(initial_scc_csv_path)
)
log_message(f"SCC init BW = {bw_scc:.6f}", log_path)

log_message("Scoring adaptive init baseline (backward weight) ...", log_path)
bw_adaptive = compute_backward_weight_from_dimacs_edges(
    edges, index_to_node, _load_order_map_from_csv(adaptive_csv_path)
)
log_message(f"Adaptive init BW = {bw_adaptive:.6f}", log_path)

if bw_adaptive < bw_scc:
    initial_ranking_path = adaptive_csv_path
    log_message("Chosen baseline: ADAPTIVE (smaller BW).", log_path)
else:
    initial_ranking_path = initial_scc_csv_path
    log_message("Chosen baseline: SCC (smaller/equal BW).", log_path)


# -------------------- run hybrid from best baseline --------------------
final_output_path = os.path.join(base_dir, f"{graph_basename}_dimacs_hybrid_ranking_learning-48cpu.csv")
final_ranking_path = hybrid_refine_func1_func2(
    csv_path=graph_file_path,
    initial_ranking_path=initial_ranking_path,
    output_ranking_path=final_output_path,
    log_path=log_path,
    RUN_FOR_MINUTES=4320.0,  # 72 hours
    h_hours=72.0,
    c=1.0,  # ignored (kept for compatibility)
)

log_message(f"Hybrid finished. Final ranking path: {final_ranking_path}", log_path)




