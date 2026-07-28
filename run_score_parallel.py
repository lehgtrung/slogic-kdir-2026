import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import load_data
import scipy.sparse as sp
from collections import defaultdict
import math
import random
# --- NEW IMPORTS ---
from joblib import Parallel, delayed


def get_rank_at(arr, at, positions_to_filter):
    """
    Calculates the rank of a target element in an array, after filtering.
    """
    # Filter out known true answers (except the one we're ranking)
    for pos in positions_to_filter:
        if pos != at:
            arr[pos] = -np.inf  # Set score to negative infinity to ignore it

    target_value = arr[at]

    # Count elements with a higher score
    num_larger = np.sum(arr > target_value)
    # Count elements with the same score (for tie-breaking)
    num_equal = np.sum(arr == target_value) - 1  # Subtract the target element itself

    # The correct formula for expected rank is:
    # (number of better answers) + 1 (for the current position) + (number of ties / 2)
    rank = num_larger + 1 + (num_equal / 2)
    return rank, num_larger, num_equal, np.count_nonzero(arr)


def ground_rule(rule_body, adj_tensor, binarization, tanh_scale):
    """
    Helper function to ground a single rule.
    This will be run in parallel.
    """
    if not isinstance(rule_body, str):
        return rule_body, None  # Return the key, but None for the value

    try:
        prod = adj_tensor[int(rule_body.split('|')[0])]
        for elem in rule_body.split('|')[1:]:
            prod = prod @ adj_tensor[int(elem)]

        # --- Apply selected binarization method ---
        if binarization == "tanh":
            prod.data = np.tanh(prod.data / tanh_scale)
        elif binarization == "hard":
            prod.data = np.ones_like(prod.data)

        return rule_body, prod  # Return the original rule string as the key
    except Exception as e:
        print(f"Warning: Failed to ground rule {rule_body}: {e}")
        return rule_body, None


def reason_with_file_softmax(_df: pd.DataFrame, _adj_tensor, _facts, n_entities,
                             temperature: float = 1.0, top_k: int = 0, aggregation: str = "sum",
                             binarization: str = "hard", tanh_scale: float = 1.0):
    """
    Aggregates rule scores for each query and calculates ranks for evaluation.
    """
    # --- Step 1: Pre-ground all unique rules (OPTIMIZED & PARALLELIZED) ---
    print(f"Pre-grounding all unique rules with '{binarization}' binarization...")

    unique_rules = [rule for rule in _df['rule'].unique() if isinstance(rule, str)]

    # --- This replaces your entire first loop ---
    # n_jobs=-1 uses all available CPU cores
    # backend="threading" is often good for SciPy/NumPy as they release the GIL
    results = Parallel(n_jobs=4, backend="threading")(
        delayed(ground_rule)(
            rule_body=rule,
            adj_tensor=_adj_tensor,
            binarization=binarization,
            tanh_scale=tanh_scale
        ) for rule in tqdm(unique_rules)
    )

    # --- Re-assemble the dictionary from the parallel results ---
    rule_dct = {rule_key: matrix for rule_key, matrix in results if matrix is not None}
    print(f"Successfully grounded {len(rule_dct)} rules.")

    # --- Step 2: Group all rules and scores by the query (head, relation) ---
    query_dct = defaultdict(list)
    for _, row in _df.iterrows():
        head, tail, relation = row['head'], row['tail'], row['relation']
        rule, score = row['rule'], row['score']
        query_dct[(head, relation)].append({'rule': rule, 'score': score, 'true_tail': tail})

    # --- Step 3: Process each query to get the final ranks ---
    ranks = []
    num_equals = []
    num_largers = []
    num_tails = []
    print("Evaluating queries...")
    for (head, relation), rules in tqdm(query_dct.items()):

        # --- Top-K Rule Selection ---
        if top_k > 0 and len(rules) > top_k:
            rules = sorted(rules, key=lambda x: x['score'], reverse=True)[:top_k]

        scores = np.array([r['score'] for r in rules])
        true_tails_for_query = set([r['true_tail'] for r in rules])

        if not np.any(np.isfinite(scores)):
            num_candidates = n_entities - len(_facts.get((head, relation), set())) + len(true_tails_for_query)
            random_rank = (num_candidates + 1) / 2
            for _ in true_tails_for_query:
                ranks.append(random_rank)
            continue

        # --- Softmax Normalization with Temperature ---
        scaled_scores = scores / temperature
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        softmax_weights = exp_scores / np.sum(exp_scores)

        # --- Aggregation of Rule Outcomes (Sum vs. Max) ---
        final_answer_vector = sp.csr_matrix((1, n_entities), dtype=float)
        if aggregation == "sum":
            for i, rule_info in enumerate(rules):
                if not np.isnan(softmax_weights[i]) and rule_info['rule'] in rule_dct:
                    rule_matrix = rule_dct[rule_info['rule']]
                    head_row = rule_matrix.getrow(head)
                    final_answer_vector += softmax_weights[i] * head_row

        elif aggregation == "max":
            weighted_rows = []
            for i, rule_info in enumerate(rules):
                if not np.isnan(softmax_weights[i]) and rule_info['rule'] in rule_dct:
                    rule_matrix = rule_dct[rule_info['rule']]
                    head_row = rule_matrix.getrow(head)
                    weighted_rows.append(softmax_weights[i] * head_row)
            if weighted_rows:
                stacked_rows = sp.vstack(weighted_rows, format='csr')
                final_answer_vector = stacked_rows.max(axis=0)

        final_answer_vector = final_answer_vector.toarray().flatten()

        # --- Ranking ---
        for true_tail in true_tails_for_query:
            rank, num_larger, num_equal, num_tail = get_rank_at(
                arr=final_answer_vector.copy(),
                at=true_tail,
                positions_to_filter=_facts.get((head, relation), set())
            )
            ranks.append(rank)
            num_largers.append(num_larger)
            num_equals.append(num_equal)
            num_tails.append(num_tail)

    return ranks, num_largers, num_equals, num_tails


def show_results(ranks):
    """Calculates and prints MRR and Hits@k."""
    if not ranks:
        print("\n--- No valid ranks to evaluate. ---")
        return
    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)
    hits_at_1 = np.mean(ranks <= 1)
    hits_at_3 = np.mean(ranks <= 3)
    hits_at_10 = np.mean(ranks <= 10)
    print(f"\n--- Evaluation Results ---")
    print(f"Evaluated on {len(ranks)} total queries.")
    print(f"MRR:      {mrr:.4f}")
    print(f"Hits@1:   {hits_at_1:.4f}")
    print(f"Hits@3:   {hits_at_3:.4f}")
    print(f"Hits@10:  {hits_at_10:.4f}")


def compile_adj_data(df, entities, relations):
    """Compiles adjacency matrices for each relation."""
    adj_data = []
    print("Compiling adjacency matrices...")
    for i in tqdm(relations):
        df_rel = df[df['relation'] == i]
        adj = sp.csr_matrix((np.ones(len(df_rel)), (df_rel['head'], df_rel['tail'])),
                            shape=(len(entities), len(entities)))
        adj_data.append(adj)
    return adj_data


def compile_facts(df):
    """Compiles a dictionary of all true facts for filtering."""
    facts = defaultdict(set)
    for _, row in df.iterrows():
        facts[(row['head'], row['relation'])].add(row['tail'])
    return facts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rule-based reasoning for knowledge graph link prediction.")
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--scored_file", help="Path to the scored test file", required=False)
    parser.add_argument("--num_rules", type=int, required=False,
                        help="Number of test rules - alternative to test file")
    parser.add_argument("--split", help="test/valid", required=True)

    # --- Argument for not using phi score but Wilson score ---
    parser.add_argument("--raw", action='store_true', help="Use raw score instead of learned score")
    parser.add_argument("--get_rmn", action='store_true', help="Get rank, m, n on a separate file")
    parser.add_argument("--oracle", action='store_true', help="Get Oracle performance")

    # --- All Tuning Arguments ---
    parser.add_argument(
        "--temperature", type=float, default=0.5,
        help="Softmax temperature for weighting rule scores. (default: 1.0)"
    )
    parser.add_argument(
        "--top_k", type=int, default=0,
        help="Consider only the top-k scoring rules. 0 means all rules are used. (default: 0)"
    )
    parser.add_argument(
        "--aggregation", type=str, default="sum", choices=["sum", "max"],
        help="Aggregation method for rule outputs ('sum' or 'max'). (default: 'sum')"
    )
    parser.add_argument(
        "--score_threshold", type=float, default=-np.inf,
        help="Minimum score for a rule to be considered. (default: -inf, no filtering)"
    )
    parser.add_argument(
        "--wilson_threshold", type=float, default=-np.inf,
        help="Minimum Wilson score for a rule to be considered. (default: -inf, no filtering)"
    )
    parser.add_argument(
        "--length_penalty", type=float, default=0.75,
        help="Lambda for rule length penalty. 0 means no penalty. (default: 0.0)"
    )
    parser.add_argument(
        "--binarization", type=str, default="tanh", choices=["hard", "tanh", "none"],
        help="Method for binarizing rule outputs. 'hard' for 0/1, 'tanh' for soft binarization. (default: 'hard')"
    )

    parser.add_argument(
        "--n_tails_penalty", type=float, default=0.0,
        help="Lambda for n tails penalty. 0 means no penalty. (default: 0.0)"
    )

    parser.add_argument(
        "--tanh_scale", type=float, default=2.0,
        help="Scaling factor S for tanh(path_count / S). Used only if --binarization is 'tanh'. (default: 1.0)"
    )
    # --- NEW ARGUMENT FOR REDUNDANCY PENALTY ---
    parser.add_argument(
        "--redundancy_penalty", type=float, default=0.0,
        help="Penalty for rules that contain the target relation in their body. (default: 0.0)"
    )
    args = parser.parse_args()

    if args.scored_file is not None and args.num_rules is not None:
        raise ValueError('Test file and number of test rules cannot be specified at the same time')

    if args.scored_file is None and args.num_rules is None:
        raise ValueError('Either test_file or num_test_rules must be specified')

    if args.num_rules:
        args.scored_file = f'data/{args.dataset}/{args.split}_wilson_top_{args.num_rules}_scored.tsv'

    # --- Load Data ---
    df_train, ent_map, rel_map = load_data(f'datasets/{args.dataset}/train.txt',
                                           f'datasets/{args.dataset}/entities.dict',
                                           f'datasets/{args.dataset}/relations.dict')
    df_valid, _, _ = load_data(f'datasets/{args.dataset}/valid.txt',
                               f'datasets/{args.dataset}/entities.dict',
                               f'datasets/{args.dataset}/relations.dict')
    df_test, _, _ = load_data(f'datasets/{args.dataset}/test.txt',
                              f'datasets/{args.dataset}/entities.dict',
                              f'datasets/{args.dataset}/relations.dict')

    df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True, sort=False)
    entities = list(range(len(ent_map)))
    relations = list(range(len(rel_map)))
    n_entities = len(entities)
    facts = compile_facts(df_all)
    adj_tensor = compile_adj_data(df_train, entities, relations)
    # adj_tensor = compile_adj_data(pd.concat([df_train, df_valid], ignore_index=True), entities, relations)

    # --- Pre-compute relation-specific tail popularity for fallback ---
    print("\nPre-computing relation-specific tail popularity for fallback...")
    relation_priors = df_train.groupby(['relation', 'tail']).size().unstack(fill_value=0)
    fallback_vectors = {}
    for rel_id in tqdm(relations, desc="Creating popularity vectors"):
        if rel_id in relation_priors.index:
            row = relation_priors.loc[rel_id]
            tails = row.index.values
            counts = row.values.astype(float)
            fallback_vector = sp.csr_matrix((counts, (np.zeros_like(tails), tails)), shape=(1, n_entities))
            fallback_vectors[rel_id] = fallback_vector
        else:
            fallback_vectors[rel_id] = sp.csr_matrix((1, n_entities), dtype=float)

    # --- Load and Filter/Penalize Scored Data ---
    scored_df = pd.read_csv(args.scored_file, sep='\t')
    print(f'Loading scored data... from {args.scored_file}')
    print(f"\nLoaded {len(scored_df)} scored rule applications.")
    if args.raw:
        print("Use Wilson lower bound ...")
        scored_df['score'] = scored_df['wilson_lower_bound']
    if args.oracle:
        print("Computing oracle performance ...")
        scored_df['score'] = scored_df['complete'].apply(lambda x: 10 if x == 1 else 1)

    if args.score_threshold > -np.inf:
        print(f"Filtering rules with score < {args.score_threshold}...")
        initial_count = len(scored_df)
        scored_df = scored_df[scored_df['score'] >= args.score_threshold].copy()
        print(f"Removed {initial_count - len(scored_df)} rules. {len(scored_df)} remain.")

    if args.wilson_threshold > -np.inf:
        print(f"Filtering rules with Wilson < {args.wilson_threshold}...")
        initial_count = len(scored_df)
        scored_df = scored_df[scored_df['wilson_lower_bound'] >= args.wilson_threshold].copy()
        print(f"Removed {initial_count - len(scored_df)} rules. {len(scored_df)} remain.")

    if args.length_penalty > 0.0:
        print(f"Applying rule length penalty with lambda={args.length_penalty}...")
        scored_df['length'] = scored_df['rule'].apply(lambda r: len(r.split('|')) if isinstance(r, str) else 1)
        scored_df['score'] = scored_df['score'] - args.length_penalty * (scored_df['length'] - 1)
        print("Scores adjusted for rule length.")

    if args.dataset == 'FB15k-237':
        args.n_tails_penalty = 0.65

    if args.n_tails_penalty > 0.0: # default for FB15k-237 = 0.65
        print(f"Applying rule n_tails penalty with lambda={args.n_tails_penalty}...")
        scored_df['score'] = scored_df['score'] - args.n_tails_penalty * np.log(scored_df['n_tails'])
        print("Scores adjusted for rule length.")

    # --- Run Evaluation ---
    print("\n--- Running Evaluation ---")
    settings_str = (
        f"Binarization='{args.binarization}', TanhScale={'N/A' if args.binarization != 'tanh' else args.tanh_scale}, "
        f"Temp={args.temperature}, Top-K={'All' if args.top_k <= 0 else args.top_k}, Agg='{args.aggregation}', "
        f"Threshold={args.score_threshold}, Penalty={args.length_penalty}, RedundancyPenalty={args.redundancy_penalty}"
    )
    print(f"Settings: {settings_str}")

    final_ranks, final_largers, final_equals, final_tails = reason_with_file_softmax(
        _df=scored_df,
        _adj_tensor=adj_tensor,
        _facts=facts,
        n_entities=n_entities,
        temperature=args.temperature,
        top_k=args.top_k,
        aggregation=args.aggregation,
        binarization=args.binarization,
        tanh_scale=args.tanh_scale
    )

    # --- Handle Missing Queries ---
    print("\nChecking for queries in the test set that were not in the scored file...")
    if not scored_df.empty:
        scored_queries = set(zip(scored_df['head'], scored_df['relation'], scored_df['tail']))
    else:
        scored_queries = set()

    test_queries = set(zip(df_test['head'], df_test['relation'], df_test['tail']))
    valid_queries = set(zip(df_valid['head'], df_valid['relation'], df_valid['tail']))
    if args.split == 'valid':
        missing_queries = valid_queries - scored_queries
    if args.split == 'test':
        missing_queries = test_queries - scored_queries
    else:
        missing_queries = set()

    final_ranks_original = final_ranks[:]
    if missing_queries:
        print(f"Found {len(missing_queries)} missing queries. Assigning ranks based on tail popularity.")
        for head, relation, tail in missing_queries:
            fallback_vector = fallback_vectors.get(relation, sp.csr_matrix((1, n_entities), dtype=float))
            rank, _, _, _ = get_rank_at(
                arr=fallback_vector.toarray().flatten().copy(),
                at=tail,
                positions_to_filter=facts.get((head, relation), set())
            )
            final_ranks.append(rank)
    else:
        print("No missing queries found. All test queries were present in the scored file.")

    # --- Show Final Results ---
    show_results(final_ranks)
    grouped_df = scored_df[['head', 'tail', 'relation']].drop_duplicates()
    grouped_df['rank'] = final_ranks_original
    grouped_df['num_larger'] = final_largers
    grouped_df['num_equal'] = final_equals
    grouped_df['num_tails'] = final_tails
    if args.get_rmn:
        grouped_df.to_csv(args.scored_file.replace('.tsv', 'rmn.tsv'), sep='\t')
