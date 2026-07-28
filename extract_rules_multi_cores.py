import argparse
import itertools
import math
import os
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import natsort
import numpy as np
import pandas as pd
import rustworkx as rx
from tqdm import tqdm
from utils import construct_inverse_map, load_data

# This global dictionary will be populated by the initializer in each worker process
worker_globals = {}

def init_worker(_graph, _node_index_map, _inv_rel_map, _hops):
    """
    Initializer for the worker pool.
    This runs once per worker process, setting up large, read-only objects
    to be shared efficiently without repeated serialization.
    """
    worker_globals['graph'] = _graph
    worker_globals['node_index_map'] = _node_index_map
    worker_globals['inv_rel_map'] = _inv_rel_map
    worker_globals['hops'] = _hops

def wilson_lower_bound(p_hat, n, z=1.96):
    """Calculate the lower bound of the Wilson score interval."""
    if n == 0:
        return 0.0
    numerator = (p_hat + (z ** 2 / (2 * n)) -
                 z * math.sqrt((p_hat * (1 - p_hat) / n) + (z ** 2 / (4 * n ** 2))))
    denominator = 1 + (z ** 2 / n)
    return numerator / denominator

def add_unique_node(graph, node_index_map, label):
    if label in node_index_map:
        return node_index_map[label]
    else:
        node_index = graph.add_node(label)
        node_index_map[label] = node_index
        return node_index

def input_graph_rx(_df: pd.DataFrame):
    _node_index_map = {}
    _graph = rx.PyDiGraph()
    for _, row in tqdm(_df.iterrows(), desc="Building graph"):
        head, relation, tail = row['head'], row['relation'], row['tail']
        head_node = add_unique_node(_graph, _node_index_map, head)
        tail_node = add_unique_node(_graph, _node_index_map, tail)
        _graph.add_edge(head_node, tail_node, relation)
    return _graph, _node_index_map

def generate_metapaths_from_paths(graph, paths, head_node, relation, inv_relation, tail_node):
    all_meta_paths = set()
    for path in paths:
        # Create a generator of edge data for each step in the path
        edge_data_generator = (
            graph.get_all_edge_data(start, end)
            # ⭐️ FIX IS HERE: Changed path[:-1] to path[1:]
            for start, end in zip(path, path[1:])
        )

        # Use itertools.product to find all combinations of relations along the path
        for meta_path_relations in itertools.product(*edge_data_generator):
            is_original_fact = False
            for i in range(len(meta_path_relations)):
                start, end = path[i], path[i+1]
                rel = meta_path_relations[i]
                if (start == head_node and end == tail_node and rel == relation) or \
                   (start == tail_node and end == head_node and rel == inv_relation):
                    is_original_fact = True
                    break
            if not is_original_fact:
                all_meta_paths.add('|'.join(map(str, meta_path_relations)))

    return list(all_meta_paths)

def process_chunk(df_chunk: pd.DataFrame):
    """
    Worker function executed in parallel. It accesses the large shared data
    (graph, maps) from the `worker_globals` dictionary.
    """
    # Retrieve shared objects from the global dictionary
    graph = worker_globals['graph']
    node_index_map = worker_globals['node_index_map']
    inv_rel_map = worker_globals['inv_rel_map']
    hops = worker_globals['hops']

    _rules_counter = defaultdict(Counter)
    _body_counter = Counter()
    _pos_rules = []

    for _, row in df_chunk.iterrows():
        head, tail, relation = row['head'], row['tail'], row['relation']
        node_head = node_index_map.get(head)
        node_tail = node_index_map.get(tail)

        if node_head is None or node_tail is None:
            continue

        try:
            paths_iter = rx.all_simple_paths(graph, node_head, node_tail, cutoff=hops)
            # Ensure path has at least one edge (i.e., length > 1 node)
            paths = {tuple(p) for p in paths_iter if len(p) > 1}
        except (TypeError, rx.NoPathFound):
            continue

        if not paths:
            continue

        inv_relation = inv_rel_map.get(relation)
        meta_paths = generate_metapaths_from_paths(graph, paths, node_head, relation, inv_relation, node_tail)

        if len(meta_paths) > 0:
            for meta_path in meta_paths:
                _rules_counter[relation][meta_path] += 1
                _body_counter[meta_path] += 1
                _pos_rules.append([row['head'], row['tail'], row['relation'], meta_path])

    return _rules_counter, _body_counter, _pos_rules

def extract_rules(_df: pd.DataFrame, _inv_rel_map: dict, _graph: rx.PyDiGraph,
                  _node_index_map: dict, _hops: int, _num_cores: int):
    """Orchestrates the parallel rule extraction."""
    if _num_cores <= 0:
        _num_cores = cpu_count() - 1 or 1
    print(f"Starting rule extraction with {_num_cores} processes...")

    df_chunks = np.array_split(_df, _num_cores)
    init_args = (_graph, _node_index_map, _inv_rel_map, _hops)

    final_rules_counter = defaultdict(Counter)
    final_body_counter = Counter()
    final_pos_rules = []

    with Pool(processes=_num_cores, initializer=init_worker, initargs=init_args) as pool:
        results = list(tqdm(pool.imap(process_chunk, df_chunks), total=len(df_chunks), desc="Extracting rules"))

    print("Aggregating results...")
    for _rules_counter, _body_counter, _pos_rules in results:
        for relation, meta_path_counts in _rules_counter.items():
            final_rules_counter[relation].update(meta_path_counts)
        final_body_counter.update(_body_counter)
        final_pos_rules.extend(_pos_rules)

    return final_rules_counter, final_body_counter, final_pos_rules

def write_rules(_rules_counter, _body_counter, _relation_counts, path):
    """Writes the final rules and their metrics to a file."""
    flat_rules = ["rule_head\trule_body\tcount\tsupport\tconfidence\tlaplace_confidence\twilson_lower_bound"]

    for rule_head in natsort.natsorted(_rules_counter.keys()):
        rules = sorted(_rules_counter[rule_head].most_common(), key=lambda x: int(x[1]), reverse=True)
        for rule_body, count_instance in rules:
            count_body = _body_counter.get(rule_body, 0)
            count_head = _relation_counts.get(rule_head, 0)
            support = count_instance / count_head if count_head > 0 else 0.0
            confidence = count_instance / count_body if count_body > 0 else 0.0
            laplace_confidence = (count_instance + 1) / (count_body + 2)
            wilson = wilson_lower_bound(p_hat=confidence, n=count_body)
            line = (f'{rule_head}\t{rule_body}\t{count_instance}\t'
                    f'{support:.6f}\t{confidence:.6f}\t'
                    f'{laplace_confidence:.6f}\t{wilson:.6f}')
            flat_rules.append(line)

    with open(path, 'w') as f:
        f.write('\n'.join(flat_rules))


def sample_facts(df: pd.DataFrame, samples: int, sample_per_relation: int, seed: int = 42) -> pd.DataFrame:
    if samples > 0 and sample_per_relation > 0:
        raise ValueError("Use either --samples or --sample_per_relation, not both.")

    if sample_per_relation > 0:
        print(f"Sampling up to {sample_per_relation} rows per relation...")
        sampled = (
            df.groupby("relation", group_keys=False)
            .apply(lambda group: group.sample(n=min(len(group), sample_per_relation), random_state=seed))
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
        print(f"Sampled {len(sampled)} rows across {sampled['relation'].nunique()} relations.")
        return sampled

    if samples > 0:
        sample_size = min(samples, len(df))
        print(f"Sampling {sample_size} rows...")
        return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    return df


def parse_sample_sizes(sample_sizes: str):
    if not sample_sizes:
        return []

    parsed_sizes = []
    for raw_size in sample_sizes.split(','):
        raw_size = raw_size.strip()
        if not raw_size:
            continue
        if raw_size.lower() == 'full':
            parsed_sizes.append(-1)
        else:
            sample_size = int(raw_size)
            if sample_size <= 0:
                raise ValueError("--sample_sizes entries must be positive integers or 'full'.")
            parsed_sizes.append(sample_size)
    return parsed_sizes


def sample_label(samples: int):
    return 'full' if samples <= 0 else f'samples_{samples}'


def write_extraction_outputs(rules_counter, body_counter, pos_rules, relation_counts, output, output_pos):
    if output_pos:
        os.makedirs(os.path.dirname(output_pos), exist_ok=True)
        pd.DataFrame(pos_rules, columns=['head', 'tail', 'relation', 'pos_rule']).to_csv(
            output_pos, index=False, sep='\t'
        )
        print(f"Positive rule instances saved to {output_pos}")

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        print(f"Writing final rules to {output}...")
        write_rules(rules_counter, body_counter, relation_counts, output)
        print(f"Rules saved to {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract chain-like rules from a knowledge graph.")
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--hops", help="Maximum path length for rules", type=int, default=-1)
    parser.add_argument("--output_pos", help="Path to save positive rule instances")
    parser.add_argument("--output", help="Path to save the final rule file")
    parser.add_argument("--samples", help="Number of facts to sample for rule extraction", type=int, default=-1)
    parser.add_argument(
        "--sample_sizes",
        help="Comma-separated sample sizes for ablation, e.g. 10000,50000,200000,full",
    )
    parser.add_argument(
        "--output_dir",
        help="Base output directory for --sample_sizes. Defaults to data/<dataset>/sample_ablation.",
    )
    parser.add_argument(
        "--sample_per_relation",
        help="Sample up to this many facts per relation for rule extraction",
        type=int,
        default=-1,
    )
    parser.add_argument("--cores", help="Number of CPU cores to use (default: all - 1)", type=int, default=0)
    args = parser.parse_args()

    # --- Setup ---
    dataset = args.dataset
    hops = args.hops
    samples = args.samples
    sample_per_relation = args.sample_per_relation
    sample_sizes = parse_sample_sizes(args.sample_sizes)
    num_cores = args.cores

    if sample_sizes and sample_per_relation > 0:
        raise ValueError("Use either --sample_sizes or --sample_per_relation, not both.")
    if sample_sizes and samples > 0:
        raise ValueError("Use either --sample_sizes or --samples, not both.")

    if args.output is None:
        args.output = f'data/{args.dataset}/rules.tsv'
    if args.output_pos is None:
        args.output_pos = f'data/{args.dataset}/train_pos_rules.tsv'
    if args.output_dir is None:
        args.output_dir = f'data/{args.dataset}/sample_ablation'

    if hops == -1:
        hops_map = {'wn18rr': 6, 'FB15k-237': 4, 'yago3-10': 4}
        if dataset in hops_map:
            hops = hops_map[dataset]
        else:
            raise ValueError(f'Unknown dataset "{dataset}". Please specify --hops.')

    print(f'Dataset: {dataset} | Hops: {hops}')

    # --- Data Loading ---
    data_path = f'datasets/{dataset}'
    df_train, _, _ = load_data(f'{data_path}/train.txt',
                               f'{data_path}/entities.dict',
                               f'{data_path}/relations.dict')
    graph, node_index_map = input_graph_rx(df_train)

    df, _, _ = load_data(f'{data_path}/train.txt',
                         f'{data_path}/entities.dict',
                         f'{data_path}/relations.dict')

    inv_rel_map = construct_inverse_map(f'{data_path}/relations.dict')
    relation_counts = df['relation'].value_counts()
    print(f'Length of dataset: {len(df)}')

    if sample_sizes:
        print(f"Running sample-size ablation: {[sample_label(size) for size in sample_sizes]}")
        for sample_size in sample_sizes:
            print(f"\n--- Extracting rules for {sample_label(sample_size)} ---")
            sampled_df = sample_facts(df, samples=sample_size, sample_per_relation=-1)
            rules_counter, body_counter, pos_rules = extract_rules(
                _df=sampled_df, _inv_rel_map=inv_rel_map, _graph=graph,
                _node_index_map=node_index_map, _hops=hops, _num_cores=num_cores
            )
            run_dir = os.path.join(args.output_dir, sample_label(sample_size))
            write_extraction_outputs(
                rules_counter=rules_counter,
                body_counter=body_counter,
                pos_rules=pos_rules,
                relation_counts=relation_counts,
                output=os.path.join(run_dir, 'rules.tsv'),
                output_pos=os.path.join(run_dir, 'train_pos_rules.tsv')
            )
    else:
        df = sample_facts(df, samples=samples, sample_per_relation=sample_per_relation)
        rules_counter, body_counter, pos_rules = extract_rules(
            _df=df, _inv_rel_map=inv_rel_map, _graph=graph,
            _node_index_map=node_index_map, _hops=hops, _num_cores=num_cores
        )
        write_extraction_outputs(
            rules_counter=rules_counter,
            body_counter=body_counter,
            pos_rules=pos_rules,
            relation_counts=relation_counts,
            output=args.output,
            output_pos=args.output_pos
        )

    print("Done!")
