#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import random
import heapq
import time
import logging
import multiprocessing
import sys
import os
import gc
import glob
from datetime import datetime
from functools import cmp_to_key

# ==========================================
# 1. Global Data Structures (Shared Memory)
# ==========================================
global_u = None
global_v = None
global_w = None
global_node_count = 0
global_template_probs = None

# ==========================================
# 2. Crossover Engine & Worker Functions
# ==========================================
class FastCrossover:
    def __init__(self, n):
        self.n = n
        self.parent = np.arange(n, dtype=np.int32)
        self.left_bound = np.arange(n, dtype=np.int32)
        self.right_bound = np.arange(n, dtype=np.int32)
        self.occupied = np.zeros(n, dtype=bool)
        self.pos_a = np.empty(n + 1, dtype=np.int32)
        self.pos_b = np.empty(n + 1, dtype=np.int32)
        self.child = np.zeros(n, dtype=np.int32)

    def _reset(self):
        self.parent[:] = np.arange(self.n, dtype=np.int32)
        self.left_bound[:] = np.arange(self.n, dtype=np.int32)
        self.right_bound[:] = np.arange(self.n, dtype=np.int32)
        self.occupied.fill(False)

    def _find(self, i):
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        curr = i
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def _union(self, i, j):
        root_i = self._find(i)
        root_j = self._find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            self.left_bound[root_j] = min(self.left_bound[root_j], self.left_bound[root_i])
            self.right_bound[root_j] = max(self.right_bound[root_j], self.right_bound[root_i])

    def create_child(self, parent_a, parent_b, template_c):
        self._reset()
        self.pos_a[parent_a] = np.arange(self.n)
        self.pos_b[parent_b] = np.arange(self.n)
        
        for val in template_c:
            p_a = self.pos_a[val]
            p_b = self.pos_b[val]
            target = (p_a + p_b) // 2
            
            final_pos = -1
            if not self.occupied[target]:
                final_pos = target
            else:
                root = self._find(target)
                l_idx = self.left_bound[root] - 1
                r_idx = self.right_bound[root] + 1
                valid_l = l_idx >= 0
                valid_r = r_idx < self.n
                
                if valid_l and valid_r:
                    if (target - l_idx) <= (r_idx - target): final_pos = l_idx
                    else: final_pos = r_idx
                elif valid_l: final_pos = l_idx
                elif valid_r: final_pos = r_idx
                else: continue

            self.child[final_pos] = val
            self.occupied[final_pos] = True
            
            if final_pos > 0 and self.occupied[final_pos - 1]:
                self._union(final_pos, final_pos - 1)
            if final_pos < self.n - 1 and self.occupied[final_pos + 1]:
                self._union(final_pos, final_pos + 1)
                
        return self.child.copy()

worker_engine = None

def init_worker(n):
    global worker_engine
    worker_engine = FastCrossover(n)

def generate_template_worker():
    items = list(range(global_node_count))
    def compare(i, j):
        prob = global_template_probs.get((i, j), 0.5)
        if random.random() < prob: return -1 
        return 1
    sorted_items = sorted(items, key=cmp_to_key(compare))
    return np.array(sorted_items, dtype=np.int32)

def calculate_fitness_worker(perm):
    pos = np.empty(global_node_count, dtype=np.int32)
    pos[perm] = np.arange(global_node_count, dtype=np.int32)
    mask = pos[global_u] < pos[global_v]
    return np.sum(global_w[mask])

def worker_task(args):
    parent_a, parent_b = args
    template_c = generate_template_worker()
    child = worker_engine.create_child(parent_a, parent_b, template_c)
    
    # Mutation (10% chance)
    if random.random() < 0.1:
        idx1, idx2 = random.sample(range(global_node_count), 2)
        child[idx1], child[idx2] = child[idx2], child[idx1]
        
    fitness = calculate_fitness_worker(child)
    return fitness, child

# ==========================================
# 3. Data Loading Helpers
# ==========================================
def load_initial_scores(csv_path, node_to_index):
    """
    Loads ranking from CSV.
    Expects 'Node ID' and 'Order' (or 'Rank').
    Returns dict: {node_index: rank}
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Robust column name handling (Node ID vs Node_ID, Order vs Rank)
        cols = df.columns
        if 'Node ID' not in cols and 'Node_ID' in cols:
            df.rename(columns={'Node_ID': 'Node ID'}, inplace=True)
        if 'Order' not in cols and 'Rank' in cols:
            df.rename(columns={'Rank': 'Order'}, inplace=True)

        # Standardize Strings
        df['Node ID'] = df['Node ID'].astype(str).str.strip()
        
        rank_map = {row['Node ID']: row['Order'] for _, row in df.iterrows()}
        scores = {}
        
        # Map Node Strings to Indices
        for node_str, idx in node_to_index.items():
            if node_str in rank_map:
                scores[idx] = int(rank_map[node_str])

        # Assign unique ranks to unranked nodes (if any)
        max_rank = max(scores.values(), default=0) + 1
        for node_str, idx in node_to_index.items():
            if idx not in scores:
                scores[idx] = max_rank
                max_rank += 1
                
        return scores
    except Exception as e:
        print(f"Warning: Failed to load {csv_path}: {e}")
        return None

# ==========================================
# 4. Main Genetic Algorithm Class
# ==========================================
class WFASParallelSolver:
    def __init__(self, u_arr, v_arr, w_arr, node_to_index, index_to_node, output_dir, pop_size=70):
        self.node_to_index = node_to_index
        self.index_to_node = index_to_node
        self.n = len(node_to_index)
        self.pop_size = pop_size
        self.output_dir = output_dir
        
        # Shared Memory Init
        global global_u, global_v, global_w, global_node_count, global_template_probs
        global_u = u_arr
        global_v = v_arr
        global_w = w_arr
        global_node_count = self.n
        
        # Build Template Probabilities
        print("Building probability map...")
        edge_map = {}
        for i in range(len(u_arr)):
            edge_map[(u_arr[i], v_arr[i])] = w_arr[i]
            
        global_template_probs = {}
        for (u, v), w in edge_map.items():
            w_ji = edge_map.get((v, u), 0.0)
            if (w + w_ji) > 0:
                global_template_probs[(u, v)] = w / (w + w_ji)
            else:
                global_template_probs[(u, v)] = 0.5
                
        # --- File Identifiers ---
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Updated Naming Scheme for Semi-Random Initialization
        self.unified_csv = os.path.join(output_dir, "unified_rankings_genetic_semi_random.csv")
        self.log_file = os.path.join(output_dir, f"run_genetic_semi_random_{self.run_id}.log")
        
        # --- Enhanced Logging ---
        logging.basicConfig(
            filename=self.log_file, 
            level=logging.INFO, 
            format='%(asctime)s | %(levelname)s | %(message)s'
        )
        self.logger = logging.getLogger()
        
        self.population = []
        self.next_uid = 0
        self.history = set()

    def generate_random_perm(self):
        items = list(range(self.n))
        random.shuffle(items)
        return np.array(items, dtype=np.int32)

    def append_result(self, generation, fitness, perm):
        names = [self.index_to_node[i] for i in perm]
        perm_str = ";".join(names)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = f"{self.run_id},{timestamp},{generation},{fitness},{perm_str}\n"
        
        with open(self.unified_csv, "a") as f:
            if os.stat(self.unified_csv).st_size == 0:
                f.write("Run_ID,Timestamp,Generation,Fitness,Permutation\n")
            f.write(row)

    def solve(self, max_seconds, stagnation_limit):
        start_time = time.time()
        self.logger.info(f"STARTING OPTIMIZATION. Nodes: {self.n}, PopSize: {self.pop_size}")
        self.logger.info("Mode: Semi-Random Initialization (Loading existing rankings)")
        
        print(f"Initializing Population...")
        init_start = time.time()
        
        # --- 1. Semi-Random Initialization ---
        # Search for CSV files containing "ranking"
        search_pattern = os.path.join(self.output_dir, "*ranking*.csv")
        existing_files = glob.glob(search_pattern)
        
        seeded_count = 0
        
        # Process existing files
        for csv_file in existing_files:
            # Skip the unified output file itself to avoid self-loop reading errors
            if "unified_rankings" in csv_file:
                continue
                
            print(f"Loading seed: {os.path.basename(csv_file)}")
            scores_dict = load_initial_scores(csv_file, self.node_to_index)
            
            if scores_dict:
                # Convert dictionary {idx: rank} to sorted permutation list
                # Logic: Sort indices based on their rank value
                sorted_nodes = sorted(scores_dict.keys(), key=lambda k: scores_dict[k])
                perm_arr = np.array(sorted_nodes, dtype=np.int32)
                
                # Check consistency
                if len(perm_arr) == self.n:
                    fit = calculate_fitness_worker(perm_arr)
                    self.population.append({'id': self.next_uid, 'perm': perm_arr, 'fit': fit})
                    self.next_uid += 1
                    seeded_count += 1
                else:
                    self.logger.warning(f"Skipped {csv_file}: Node count mismatch ({len(perm_arr)} vs {self.n})")

        self.logger.info(f"Loaded {seeded_count} valid seeds from CSV files.")
        print(f"Loaded {seeded_count} seeds from disk.")

        # Fill the rest of the population with random permutations
        needed = self.pop_size - len(self.population)
        if needed > 0:
            print(f"Generating {needed} random permutations to fill population...")
            random_perms = [self.generate_random_perm() for _ in range(needed)]
            for perm in random_perms:
                fit = calculate_fitness_worker(perm)
                self.population.append({'id': self.next_uid, 'perm': perm, 'fit': fit})
                self.next_uid += 1
        
        # Sort initial population
        self.population.sort(key=lambda x: x['fit'], reverse=True)
        # Ensure we don't exceed pop_size if we loaded too many files
        self.population = self.population[:self.pop_size]
        
        best_global = self.population[0]
        init_duration = time.time() - init_start
        
        self.logger.info(f"Init Complete. Duration: {init_duration:.2f}s. Best: {best_global['fit']}")
        self.append_result(0, best_global['fit'], best_global['perm'])
        print(f"Gen 0 | Best: {best_global['fit']} (Seeded)")
        
        # --- 2. Evolution Loop ---
        rounds = 0
        rounds_stag = 0
        
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cpu_count, initializer=init_worker, initargs=(self.n,))
        
        try:
            while True:
                gen_start_time = time.time()
                total_elapsed = gen_start_time - start_time
                
                if total_elapsed > max_seconds:
                    self.logger.info("STOPPING: Time limit.")
                    print("Time limit reached.")
                    break
                if rounds_stag >= stagnation_limit:
                    self.logger.info("STOPPING: Stagnation limit.")
                    print("Stagnation limit reached.")
                    break
                
                rounds += 1
                
                # Identify Pairs
                tasks = []
                current_pop_size = len(self.population)
                for i in range(current_pop_size):
                    for j in range(i + 1, current_pop_size):
                        p1, p2 = self.population[i], self.population[j]
                        
                        if p1['id'] < p2['id']:
                            key, pa, pb = (p1['id'], p2['id']), p1['perm'], p2['perm']
                        else:
                            key, pa, pb = (p2['id'], p1['id']), p2['perm'], p1['perm']
                            
                        if key not in self.history:
                            self.history.add(key)
                            tasks.append((pa, pb))
                
                if not tasks:
                    self.logger.info("STOPPING: Converged.")
                    break
                
                # Execute
                chunk = max(1, len(tasks)//(cpu_count*4))
                results = pool.map(worker_task, tasks, chunksize=chunk)
                
                # Update
                for fit, perm in results:
                    self.population.append({'id': self.next_uid, 'perm': perm, 'fit': fit})
                    self.next_uid += 1
                
                self.population.sort(key=lambda x: x['fit'], reverse=True)
                self.population = self.population[:self.pop_size]
                
                # Metrics
                gen_duration = time.time() - gen_start_time
                current_best = self.population[0]
                
                log_msg = f"Gen {rounds} | Dur: {gen_duration:.2f}s | Pairs: {len(tasks)} | Best: {current_best['fit']}"
                
                if current_best['fit'] > best_global['fit']:
                    best_global = current_best
                    rounds_stag = 0
                    self.logger.info(f"{log_msg} | **NEW BEST**")
                    print(f"Gen {rounds} | **NEW BEST**: {best_global['fit']} (in {gen_duration:.2f}s)")
                    self.append_result(rounds, best_global['fit'], best_global['perm'])
                else:
                    rounds_stag += 1
                    self.logger.info(f"{log_msg} | Stag: {rounds_stag}")
                    print(f"Gen {rounds} | Best: {best_global['fit']} | Stag: {rounds_stag}", end='\r')
                
                del results
                del tasks
                gc.collect()

        finally:
            pool.close()
            pool.join()
            
        return best_global['fit']

# ==========================================
# 5. Entry Point
# ==========================================
if __name__ == "__main__":
    INPUT_PATH = "/mmfs1/home/sv96/Feedback-arc-set-paper/datasets/connectome_graph.csv"
    MAX_HOURS = 72
    STAGNATION_LIMIT = 5
    POP_SIZE = 130
    
    if not os.path.exists(INPUT_PATH):
        print(f"Input not found: {INPUT_PATH}")
        output_dir = "."
    else:
        output_dir = os.path.dirname(os.path.abspath(INPUT_PATH))
        
    print("Reading Data...")
    try:
        df = pd.read_csv(INPUT_PATH)
        df.rename(columns={'Source Node  ID': 'source', 'Target Node ID': 'target', 'Edge Weight': 'weight'}, inplace=True)
        
        unique_nodes = np.unique(np.concatenate((df['source'].astype(str).unique(), df['target'].astype(str).unique())))
        node_to_idx = {n: i for i, n in enumerate(unique_nodes)}
        idx_to_node = {i: n for n, i in node_to_idx.items()}
        
        u_arr = df['source'].astype(str).map(node_to_idx).values.astype(np.int32)
        v_arr = df['target'].astype(str).map(node_to_idx).values.astype(np.int32)
        w_arr = df['weight'].values.astype(np.float32)
        
        print(f"Graph: {len(unique_nodes)} nodes, {len(u_arr)} edges.")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
        
    solver = WFASParallelSolver(u_arr, v_arr, w_arr, node_to_idx, idx_to_node, output_dir, pop_size=POP_SIZE)
    final_score = solver.solve(max_seconds=MAX_HOURS*3600, stagnation_limit=STAGNATION_LIMIT)
    
    print("\n" + "="*30)
    print(f"DONE. Final Score: {final_score}")


# In[ ]:




