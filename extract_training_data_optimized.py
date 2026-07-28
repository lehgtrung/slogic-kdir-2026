import argparse
import os
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
from functools import partial

# --- WORKER FUNCTIONS FOR PARALLELISM ---

# Global variable to hold local rules in worker processes (avoids pickling overhead)
LOCAL_RULES_GLOBAL = None


def worker_init(local_rules):
    """Initializes the worker process with the shared read-only local rules."""
    global LOCAL_RULES_GLOBAL
    LOCAL_RULES_GLOBAL = local_rules


def process_triplet_group(args):
    """
    Processes a batch of triplets.
    args: tuple of ((h, r, t), pos_rows, exclusion_set, k_neg, top_down)
    """
    (head, relation, tail), pos_rows, exclusion_set, k_neg, top_down = args
    results = []

    # 1. Fetch locally applicable rules for this head+relation
    # Access the global dictionary directly
    candidates = LOCAL_RULES_GLOBAL.get(head, {}).get(relation, [])

    if not candidates:
        return results

    # 2. Filter Candidates (Identify valid negatives)
    # We do this ONCE per triplet, not once per positive sample
    neg_candidates = [
        rule for rule in candidates
        if rule['body'] not in exclusion_set
    ]

    if not neg_candidates:
        return results

    # 3. Pair each positive rule with sampled negatives
    # pos_rows is a list of dicts (the sampled positive rules for this triplet)
    for pos_data in pos_rows:

        # Select negatives based on strategy
        current_neg_samples = []
        if top_down:
            # Top-down: take top k valid ones
            current_neg_samples = neg_candidates[:k_neg]
        else:
            # Random sampling
            if len(neg_candidates) <= k_neg:
                current_neg_samples = neg_candidates
            else:
                current_neg_samples = random.sample(neg_candidates, k_neg)

        # Build training instances
        for neg_rule in current_neg_samples:
            training_instance = {
                "head": head,
                "tail": tail,
                "relation": relation,
                # Positive Info
                "pos_body": pos_data['body'],
                "pos_count": pos_data['count'],
                "pos_support": pos_data['support'],
                "pos_confidence": pos_data['confidence'],
                "pos_laplace_confidence": pos_data['laplace_confidence'],
                "pos_wilson_lower_bound": pos_data['wilson_lower_bound'],
                # Negative Info
                "neg_body": neg_rule['body'],
                "neg_count": neg_rule['count'],
                "neg_support": neg_rule['support'],
                "neg_confidence": neg_rule['confidence'],
                "neg_laplace_confidence": neg_rule['laplace_confidence'],
                "neg_wilson_lower_bound": neg_rule['wilson_lower_bound']
            }
            results.append(training_instance)

    return results


def make_training_for_contrastive_parallel(_df_pos, _local_rules, k_pos, k_neg, top_down=False, max_workers=None):
    print("Preparing groups for parallel processing...")

    # 1. Build Exclusion Sets (Vectorized)
    # Map (h, r, t) -> set of all KNOWN positive bodies (from full dataframe)
    print("Building exclusion sets...")
    exclusion_map = _df_pos.groupby(['head', 'relation', 'tail'])['body'].apply(set).to_dict()

    # 2. Sample Positive Rules (Vectorized)
    print(f"Sampling {k_pos} positive rules per triplet...")
    sampled_df = _df_pos.groupby(['head', 'relation', 'tail'], group_keys=False).apply(
        lambda x: x.sample(min(len(x), k_pos))
    )

    # 3. Prepare Tasks
    # Group the SAMPLED data by triplet so we can process one triplet at a time
    print("Grouping data for workers...")
    # Convert to list of dicts for easier handling
    sampled_records = sampled_df.to_dict('records')

    # Organize by triplet: { (h,r,t): [row1, row2...] }
    triplet_groups = defaultdict(list)
    for row in sampled_records:
        key = (row['head'], row['relation'], row['tail'])
        triplet_groups[key].append(row)

    # Create task arguments
    tasks = []
    for key, rows in triplet_groups.items():
        excl_set = exclusion_map.get(key, set())
        tasks.append((key, rows, excl_set, k_neg, top_down))

    print(f"Created {len(tasks)} triplet tasks. Starting parallel execution...")

    results = []

    # 4. Parallel Execution
    # usage of 'chunksize' helps reduce IPC overhead
    chunk_size = max(1, len(tasks) // (max_workers or os.cpu_count() or 4) // 4)

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=worker_init,
            initargs=(_local_rules,)
    ) as executor:
        # map maintains order, but as_completed is fine here too.
        # map is cleaner for progress bars with total count.
        for batch_res in tqdm(executor.map(process_triplet_group, tasks, chunksize=chunk_size),
                              total=len(tasks),
                              desc="Processing triplets"):
            results.extend(batch_res)

    return pd.DataFrame(results)


# --- MAIN BLOCK ---

from utils import load_data  # Ensure this import is available

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--pos_rule_path", help="Table path", required=False)
    parser.add_argument("--output", help="Output path", required=False)
    parser.add_argument("--local_rule_path", help="Local rule path", required=False)
    parser.add_argument("--rule_path", help="Rule path", required=False)
    parser.add_argument("--kpos", type=int, help="Number of pos rules", required=True)
    parser.add_argument("--kneg", type=int, help="Number of neg rules", required=True)
    parser.add_argument("--samples", type=int, help="Number of final pairs", required=False)
    parser.add_argument("--top_down", action='store_true',
                        help="Use top-down exclusion instead of random sampling for negative rules.")
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU workers")

    args = parser.parse_args()

    # Defaults
    if args.local_rule_path is None:
        args.local_rule_path = f'locally_applicable_metapaths/{args.dataset}/train_local_rules.pkl'
    if args.rule_path is None:
        args.rule_path = f'data/{args.dataset}/rules.tsv'
    if args.pos_rule_path is None:
        args.pos_rule_path = f'data/{args.dataset}/train_pos_rules.tsv'
    if args.output is None:
        args.output = f'data/{args.dataset}/train_data/train_wilson_kpos_{args.kpos}_kneg_{args.kneg}_random.tsv'

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Working on {args.dataset}")
    print(f"Loading local rules from {args.local_rule_path}...")

    with open(args.local_rule_path, 'rb') as f:
        local_rules = pickle.load(f)

    print("Loading rule stats...")
    rules = pd.read_csv(args.rule_path, sep='\t')

    print("Loading positive examples...")
    pos_rules = pd.read_csv(args.pos_rule_path, sep='\t')

    # Merge stats into positive rules
    pre_length = len(pos_rules)
    pos_rules = (
        pos_rules.rename(columns={'relation': 'rule_head', 'pos_rule': 'rule_body'})
        .merge(rules, on=['rule_head', 'rule_body'])
        .rename(columns={'rule_head': 'relation', 'rule_body': 'body'})
    )
    assert len(pos_rules) == pre_length, "Merge changed dataset size!"

    # --- Run Optimized Generation ---
    table = make_training_for_contrastive_parallel(
        pos_rules,
        local_rules,
        args.kpos,
        args.kneg,
        top_down=args.top_down,
        max_workers=args.workers
    )

    # Save
    print(f"Saving {len(table)} training pairs to {args.output}...")
    if args.samples and args.samples < len(table):
        table.sample(n=args.samples).to_csv(args.output, sep='\t', index=False)
    else:
        table.sample(frac=1.0).to_csv(args.output, sep='\t', index=False)

    print("Done.")