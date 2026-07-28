import argparse
import os
import networkx as nx
import scipy.sparse as sp
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import load_data, input_graph
from utils import read_top_k_rules_by_wilson_score
from scipy.sparse import csr_matrix
import concurrent.futures
from functools import partial

# ==========================================
# Global variables for worker processes
# ==========================================
WORKER_ADJ_TENSOR = None
WORKER_RULES = None


def worker_init(adj_tensor, rules):
    """Initializes global memory inside each worker process ONCE."""
    global WORKER_ADJ_TENSOR, WORKER_RULES
    WORKER_ADJ_TENSOR = adj_tensor
    WORKER_RULES = rules


def compile_adj_data(_df: pd.DataFrame, _entities, _relations):
    n = len(_entities)
    _adj_tensor = {}

    for rel in _relations:
        df_rel = _df.query(f'relation == {rel}')
        rows = df_rel['head'].to_numpy()
        cols = df_rel['tail'].to_numpy()
        # Using int8 instead of float64 for cache-friendly matrix operations
        data = np.ones(len(rows), dtype=np.int8)

        sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
        _adj_tensor[rel] = sparse_matrix
    return _adj_tensor


def process_query_chunk(queries, num_rules, n_nodes):
    """
    Worker function that processes a chunk of queries.
    It checks both applicability AND completion in the exact same vector pass.
    """
    global WORKER_ADJ_TENSOR, WORKER_RULES
    results = []

    for head, tail, relation in queries:
        if relation not in WORKER_RULES:
            continue

        rule_stats = WORKER_RULES[relation]
        num_available_rules = len(rule_stats.get('body', []))
        found_rules_count = 0

        for i in range(num_available_rules):
            metapath = rule_stats['body'][i]
            relations = [int(e) for e in metapath.split('|')]

            # Initialize a fast 1xN CSR sparse row at the head node
            data = np.array([1], dtype=np.int8)
            indices = np.array([head], dtype=np.int32)
            indptr = np.array([0, 1], dtype=np.int32)
            current_row = csr_matrix((data, indices, indptr), shape=(1, n_nodes))

            # Traverse the metapath
            for rel in relations:
                current_row = current_row @ WORKER_ADJ_TENSOR[rel]
                if current_row.nnz == 0:
                    break

            # If nnz > 0, the rule is APPLICABLE
            if current_row.nnz > 0:

                # Check COMPLETION directly from the resulting vector indices
                # No need to multiply full NxN matrices!
                is_complete = 1 if tail in current_row.indices else 0
                n_tails = int(current_row.getnnz())

                results.append((
                    head, tail, relation, metapath,
                    rule_stats['count'][i],
                    rule_stats['support'][i],
                    rule_stats['confidence'][i],
                    rule_stats['laplace_confidence'][i],
                    rule_stats['wilson_lower_bound'][i],
                    is_complete,
                    n_tails
                ))

                found_rules_count += 1
                if found_rules_count >= num_rules:
                    break

    return results


def parallel_create_and_check_rules(_df: pd.DataFrame, _adj_tensor, _rules, _num_rules, n_nodes, max_workers=None):
    """
    Splits the test queries into chunks and distributes them across CPU cores.
    """
    print(f"Preparing {len(_df)} queries for parallel processing...")

    # Convert dataframe to a list of tuples for fast iteration
    queries = list(_df[['head', 'tail', 'relation']].itertuples(index=False, name=None))

    # Chunk size logic (keeps IPC overhead low)
    chunk_size = max(1, len(queries) // (max_workers or os.cpu_count() or 4) // 4)
    chunks = [queries[i:i + chunk_size] for i in range(0, len(queries), chunk_size)]

    all_results = []

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=worker_init,
            initargs=(_adj_tensor, _rules)
    ) as executor:
        worker_func = partial(process_query_chunk, num_rules=_num_rules, n_nodes=n_nodes)

        # Process chunks in parallel
        for chunk_result in tqdm(executor.map(worker_func, chunks),
                                 total=len(chunks),
                                 desc="Evaluating Rules & Completions"):
            all_results.extend(chunk_result)

    columns = [
        'head', 'tail', 'relation', 'rule',
        'count', 'support', 'confidence',
        'laplace_confidence', 'wilson_lower_bound', 'complete', 'n_tails'
    ]

    return pd.DataFrame(all_results, columns=columns)


def check_dataset_statistics(df, column):
    print(f'\nChecking dataset statistics on column: {column}...')
    pos_count = df[df[column] != 0].shape[0]
    neg_count = df[df[column] == 0].shape[0]
    ratio = pos_count / neg_count if neg_count > 0 else float('inf')
    print("Statistics:")
    print(f"Data shape : {df.shape}")
    print(f"Positive samples: {pos_count}")
    print(f"Negative samples: {neg_count}")
    print(f"Positive-to-negative ratio: {ratio:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--split", help="valid/test", required=True)
    parser.add_argument("--rule_path", help="Rule path", required=False)
    parser.add_argument("--output", help="Output path", required=False)
    parser.add_argument("--num_rules", help="Number of rules", type=int, required=False, default=100)
    parser.add_argument("--workers", help="Max CPU workers", type=int, required=False, default=None)

    args = parser.parse_args()
    dataset = args.dataset
    split = args.split
    rule_path = args.rule_path
    output_path = args.output
    num_rules = args.num_rules

    if rule_path is None:
        rule_path = f'data/{dataset}/rules.tsv'

    if output_path is None:
        output_path = f'data/{dataset}/{split}_wilson_top_{num_rules}.tsv'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Working on {dataset}")
    df_train, ent_map, rel_map = load_data(f'datasets/{dataset}/train.txt',
                                           f'datasets/{dataset}/entities.dict',
                                           f'datasets/{dataset}/relations.dict')

    df_valid, _, _ = load_data(f'datasets/{dataset}/valid.txt',
                               f'datasets/{dataset}/entities.dict',
                               f'datasets/{dataset}/relations.dict')

    df_test, _, _ = load_data(f'datasets/{dataset}/test.txt',
                              f'datasets/{dataset}/entities.dict',
                              f'datasets/{dataset}/relations.dict')

    entities = list(range(len(ent_map)))
    relations = list(range(len(rel_map)))
    n_nodes = len(entities)

    print('Compiling adj tensor...')
    adj_tensor = compile_adj_data(df_train, entities, relations)

    table = df_valid if split == 'valid' else df_test

    print('Loading rules...')
    rules = read_top_k_rules_by_wilson_score(rule_path, k=10000)

    # Process queries, rules, and completions concurrently
    final_table = parallel_create_and_check_rules(table, adj_tensor, rules, num_rules, n_nodes,
                                                  max_workers=args.workers)

    check_dataset_statistics(final_table, 'complete')

    print(f'Saving output to {output_path}...')
    final_table.to_csv(output_path, index=False, sep='\t')
    print("Done.")
