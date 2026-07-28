import os
import csv
import random
import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple


def load_data(data_path, ent_path, rel_path):
    _df = pd.read_csv(data_path, sep='\t', header=None, dtype=str)
    _df.columns = ['head', 'relation', 'tail']
    with open(ent_path, 'r') as f:
        ent_map = {}
        rows = [e.strip('\n') for e in f.readlines()]
        for row in rows:
            index, entity = row.split()
            ent_map[entity] = int(index)
    with open(rel_path, 'r') as f:
        rel_map = {}
        rows = [e.strip('\n') for e in f.readlines()]
        for row in rows:
            index, relation = row.split()
            rel_map[relation] = int(index)
    _df['head'] = _df['head'].map(lambda elem: ent_map[elem])
    _df['tail'] = _df['tail'].map(lambda elem: ent_map[elem])
    _df['relation'] = _df['relation'].map(lambda elem: rel_map[elem])
    return _df, ent_map, rel_map


def construct_inverse_map(filename):
    inverse_map = {}

    # Temporary dictionary to store relation by name for easy lookup
    relation_dict = {}

    with open(filename, 'r') as file:
        for line in file:
            # Split the line to get the relation ID and name
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            relation_id = int(parts[0])
            relation_name = parts[1]

            # Store the relation ID by name for later inverse lookup
            relation_dict[relation_name] = relation_id

    # Construct the inverse map based on the relation names
    for relation_name, relation_id in relation_dict.items():
        if relation_name.startswith('!'):
            # Get the original relation (without "!")
            original_relation = relation_name[1:]
            if original_relation in relation_dict:
                inverse_map[relation_id] = relation_dict[original_relation]
                inverse_map[relation_dict[original_relation]] = relation_id
        else:
            # Check if the inverse relation exists in the dictionary
            inverse_relation = '!' + relation_name
            if inverse_relation in relation_dict:
                inverse_map[relation_id] = relation_dict[inverse_relation]
                inverse_map[relation_dict[inverse_relation]] = relation_id

    return inverse_map


def input_graph(_df):
    g = nx.MultiDiGraph()
    nodes = set(_df['head'].tolist() + _df['tail'].tolist())
    g.add_nodes_from(nodes)
    for i, row in _df.iterrows():
        g.add_edge(
            row['head'],
            row['tail'],
            key=row['relation']
        )
    return g


def read_rules(path, k=1000):
    dct = defaultdict(dict)

    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            head = int(line[0])

            if 'body' not in dct[head]:
                dct[head]['body'] = []
                dct[head]['weight'] = []

            # Determine the limit for the current head
            limit = k[head] if isinstance(k, dict) and head in k else k

            if len(dct[head]['body']) < limit:
                dct[head]['body'].append(line[1])
                dct[head]['weight'].append(int(line[-1]))
    return dct


def read_top_k_rules_by_wilson_score(path, k=1000):
    """
    Reads a rule file and returns the top k rules for each rule head,
    ranked by the Wilson score lower bound, along with all their statistics.

    Args:
        path (str): The path to the rule file. The file is expected to be
                    tab-separated with a header row and the following columns:
                    rule_head, rule_body, count, support, confidence,
                    laplace_confidence, wilson_lower_bound
        k (int): The number of top rules to return for each rule head.

    Returns:
        defaultdict(dict): A dictionary where keys are rule head IDs. Each
                           value is another dictionary containing lists for:
                           'body', 'count', 'support', 'confidence',
                           'laplace_confidence', and 'wilson_lower_bound'.
    """
    # Temporarily store all rules and their stats grouped by their head relation
    temp_storage = defaultdict(list)

    with open(path, 'r') as f:
        # Read header to determine column order if needed, but we assume a fixed format
        header = next(f).strip().split('\t')

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < len(header):  # Skip empty or malformed lines
                continue

            # --- Parse all statistics from the line ---
            rule_head = int(parts[0])
            rule_body = parts[1]
            count = int(parts[2])
            support = float(parts[3])
            confidence = float(parts[4])
            laplace_confidence = float(parts[5])
            wilson_score = float(parts[6])

            # Store a tuple where the first element is the sorting key (wilson_score)
            rule_data = (
                wilson_score,
                rule_body,
                count,
                support,
                confidence,
                laplace_confidence
            )
            temp_storage[rule_head].append(rule_data)

    # Final dictionary to hold the sorted and filtered top k rules
    top_k_rules = defaultdict(dict)

    # Process each rule head to find its top k rules
    for head, rules in temp_storage.items():
        # Sort the rules by Wilson score (the first element of the tuple) in descending order
        rules.sort(key=lambda x: x[0], reverse=True)

        # Slice the list to get the top k rules
        top_rules = rules[:k]

        if top_rules:
            # Unzip the list of tuples into separate lists for each statistic
            # The order must match the tuple structure from above
            unzipped_stats = list(zip(*top_rules))

            top_k_rules[head]['wilson_lower_bound'] = list(unzipped_stats[0])
            top_k_rules[head]['body'] = list(unzipped_stats[1])
            top_k_rules[head]['count'] = list(unzipped_stats[2])
            top_k_rules[head]['support'] = list(unzipped_stats[3])
            top_k_rules[head]['confidence'] = list(unzipped_stats[4])
            top_k_rules[head]['laplace_confidence'] = list(unzipped_stats[5])

    return top_k_rules


def get_rules_rank_pct(path, pct=0.95):
    dct = defaultdict(dict)
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            head = int(line[0])
            if 'body' not in dct[head]:
                dct[head]['body'] = []
                dct[head]['weight'] = []
            dct[head]['body'].append(line[1])
            dct[head]['weight'].append(int(line[-1]))
    _df = []
    for head in dct.keys():
        for body, weight in zip(dct[head]['body'], dct[head]['weight']):
            _df.append((head, body, weight))
    _df = pd.DataFrame(_df, columns=['head', 'body', 'weight'])
    _df['rank'] = _df.groupby('head').cumcount() + 1
    _df['normalized_cumsum'] = _df.groupby('head')['weight'].transform(lambda x: x.cumsum() / x.sum())

    result = {}
    for head in _df['head'].unique().tolist():
        rank = _df.query(f'head == {head} and normalized_cumsum >= {pct}').head(1).iloc[0]['rank']
        result[head] = rank
    return result


def save_checkpoint(model, optimizer, epoch, batch_idx, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}, batch {batch_idx} to {filename}")


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)

    # Load model and optimizer state dictionaries
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Get epoch, batch index, and loss
    epoch = checkpoint['epoch']
    batch_idx = checkpoint['batch_idx']
    loss = checkpoint['loss']

    print(f"Checkpoint loaded from {filename}. Resuming at epoch {epoch}, batch {batch_idx}.")

    return epoch, batch_idx, loss


def load_data_cache(paths):
    _cache = {}
    for path in tqdm(paths, total=len(paths)):
        if os.path.exists(path):
            head = os.path.basename(path).split('.')[0]
            _cache[head] = torch.load(path)
        else:
            continue
    return _cache


def format_result(result):
    return " | ".join(f"{key}: {value:.4f}" for key, value in result.items())


def left_anti_join(left_df, right_df, on):
    merged = left_df.merge(right_df, on=on, how='left', indicator=True)
    left_only = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    return left_only


def check_dataset_statistics(df, column):
    print(f'Checking dataset statistics on column: {column}...')
    pos_count_train = df[df[column] != 0].shape[0]
    neg_count_train = df[df[column] == 0].shape[0]
    ratio = pos_count_train / neg_count_train
    print("Statistics:")
    print(f"Data shape : {df.shape}")
    print(f"Positive samples: {pos_count_train}")
    print(f"Negative samples: {neg_count_train}")
    print(f"Positive-to-negative ratio: {ratio:.2f}")
