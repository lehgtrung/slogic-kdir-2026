import argparse
import os
import networkx as nx
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import read_rules, load_data, construct_inverse_map, input_graph, get_rules_rank_pct
from utils import read_top_k_rules_by_wilson_score
from scipy.sparse import csr_matrix
import concurrent.futures
from functools import partial

# ==========================================
# Global variables for worker processes
# This prevents passing massive objects through IPC (Inter-Process Communication)
# ==========================================
WORKER_ADJ_TENSOR = None
WORKER_RULES = None
WORKER_HEAD_RELATIONS = None


def worker_init(adj_tensor, rules, head_to_relations):
    """Initializes global memory inside each worker process ONCE."""
    global WORKER_ADJ_TENSOR, WORKER_RULES, WORKER_HEAD_RELATIONS
    WORKER_ADJ_TENSOR = adj_tensor
    WORKER_RULES = rules
    WORKER_HEAD_RELATIONS = head_to_relations


def compile_adj_data(_df: pd.DataFrame, _entities, _relations):
    n = len(_entities)
    _adj_tensor = {}

    for rel in _relations:
        df_rel = _df.query(f'relation == {rel}')
        rows = df_rel['head'].to_numpy()
        cols = df_rel['tail'].to_numpy()
        # Using int8 instead of default float64 saves memory and speeds up operations
        data = np.ones(len(rows), dtype=np.int8)

        sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
        _adj_tensor[rel] = sparse_matrix
    return _adj_tensor


def check_simple_metapath_with_row_multiplication_optimized(adj_matrices, start_node, metapath):
    """
    Highly optimized metapath checker using direct matrix indices rather than full matrix multiplication masks.
    """
    relations = [int(e) for e in metapath.split('|')]

    # Get the true dimension 'n' directly from the tensor to prevent dimension mismatches
    n = adj_matrices[relations[0]].shape[0]

    # Initialize a fast 1xN CSR sparse row
    data = np.array([1], dtype=np.int8)
    indices = np.array([start_node], dtype=np.int32)
    indptr = np.array([0, 1], dtype=np.int32)
    current_row = csr_matrix((data, indices, indptr), shape=(1, n))

    visited_mask = np.zeros(n, dtype=bool)
    visited_mask[start_node] = True

    for relation in relations:
        next_row = current_row @ adj_matrices[relation]

        if next_row.nnz == 0:
            return False

        # Directly extract nonzero indices instead of doing a heavy sparse matrix multiplication
        next_indices = next_row.indices

        # Filter unvisited directly
        unvisited_mask = ~visited_mask[next_indices]
        unvisited_indices = next_indices[unvisited_mask]

        if len(unvisited_indices) == 0:
            return False

        # Update visited map
        visited_mask[unvisited_indices] = True

        # Reconstruct the current_row manually using ONLY unvisited indices.
        # This keeps 'nnz' (number of non-zeros) extremely low and fast.
        new_data = np.ones(len(unvisited_indices), dtype=np.int8)
        new_indptr = np.array([0, len(unvisited_indices)], dtype=np.int32)
        current_row = csr_matrix((new_data, unvisited_indices, new_indptr), shape=(1, n))

    return current_row.nnz > 0


def process_node_rules(node, num_rules):
    """
    Worker function that reads from the globally initialized memory footprint.
    """
    global WORKER_ADJ_TENSOR, WORKER_RULES, WORKER_HEAD_RELATIONS

    node_results = {}

    if node in WORKER_HEAD_RELATIONS:
        for relation in WORKER_HEAD_RELATIONS[node]:
            applicable_rules_with_stats = []

            if relation in WORKER_RULES and 'body' in WORKER_RULES[relation]:
                for i, metapath in enumerate(WORKER_RULES[relation]['body']):
                    if check_simple_metapath_with_row_multiplication_optimized(WORKER_ADJ_TENSOR, node, metapath):
                        rule_data = {
                            'body': metapath,
                            'count': WORKER_RULES[relation]['count'][i],
                            'support': WORKER_RULES[relation]['support'][i],
                            'confidence': WORKER_RULES[relation]['confidence'][i],
                            'laplace_confidence': WORKER_RULES[relation]['laplace_confidence'][i],
                            'wilson_lower_bound': WORKER_RULES[relation]['wilson_lower_bound'][i]
                        }
                        applicable_rules_with_stats.append(rule_data)

                        if len(applicable_rules_with_stats) == num_rules:
                            break

            if applicable_rules_with_stats:
                node_results[relation] = applicable_rules_with_stats

    return node, node_results


def extract_locally_applicable_rules_parallel(_graph: nx.MultiDiGraph,
                                              _table: pd.DataFrame,
                                              _adj_tensor,
                                              _rules,
                                              _num_rules,
                                              max_workers=None):
    print("Pre-computing head-to-relation mapping...")
    head_to_relations = _table.groupby('head')['relation'].unique().to_dict()
    print("Pre-computation finished.")

    result = {}
    nodes_to_process = list(_graph.nodes)

    # --- PARALLEL EXECUTION WITH INITIALIZER AND CHUNKING ---
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=worker_init,
            initargs=(_adj_tensor, _rules, head_to_relations)  # Passed ONCE per worker
    ) as executor:

        # Use functools.partial to lock in the trailing arguments
        worker_func = partial(process_node_rules, num_rules=_num_rules)

        # executor.map handles job chunking automatically, drastically reducing overhead
        # chunksize batched IPC transfers (100 is generally a sweet spot)
        results_iterator = executor.map(worker_func, nodes_to_process, chunksize=100)

        # --- BUG FIX: Removed zip() wrapper to prevent tuple nesting ---
        for node, node_results in tqdm(results_iterator,
                                       total=len(nodes_to_process),
                                       desc="Applying rules in parallel"):
            if node_results:
                result[node] = node_results

    return result


if __name__ == '__main__':
    from datetime import datetime
    print(datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument("--rule_path", help="Rule path", required=True)
    parser.add_argument("--pos_rule_path", help="Positive Rule path", required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument("--num_rules", help="Number of rules", type=int, required=False, default=100)
    args = parser.parse_args()

    dataset = args.dataset
    output = args.output
    rule_path = args.rule_path
    num_rules = args.num_rules

    df_train, ent_map, rel_map = load_data(f'datasets/{dataset}/train.txt',
                                           f'datasets/{dataset}/entities.dict',
                                           f'datasets/{dataset}/relations.dict')

    graph = input_graph(df_train)
    entities = list(range(len(ent_map)))
    relations = list(range(len(rel_map)))

    adj_tensor = compile_adj_data(df_train, entities, relations)

    print('Pairs before merging :', len(df_train[['head', 'relation']].drop_duplicates()))
    pos_rules = pd.read_csv(args.pos_rule_path, sep='\t')
    pos_rules = pos_rules[['head', 'relation', 'tail']]
    df_train = df_train.merge(pos_rules, on=['head', 'relation', 'tail'])
    print('Pairs after merging :', len(df_train[['head', 'relation']].drop_duplicates()))

    rules = read_top_k_rules_by_wilson_score(rule_path, k=10000)
    local_rules = extract_locally_applicable_rules_parallel(graph, df_train, adj_tensor, rules, num_rules)

    os.makedirs(os.path.dirname(output), exist_ok=True)

    with open(output, 'wb') as f:
        pickle.dump(local_rules, f)
    print(datetime.now())