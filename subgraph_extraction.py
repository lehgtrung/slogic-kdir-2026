import argparse
import torch
import os
import pickle
import pandas as pd
import networkx as nx
import random
from utils import load_data, input_graph, construct_inverse_map
from torch_geometric.data import Data
from tqdm import tqdm
import math
import traceback
import concurrent.futures
import multiprocessing as mp


# =====================================================================================
# 1. Optimized Subgraph Extraction and Conversion
# =====================================================================================

def extract_sampled_subgraph_and_convert(
        main_graph: nx.MultiDiGraph,
        head_node: int,
        k_hops: int,
        neighbor_sample_size: int,
        inv_rel_map: dict
):
    """
    Extracts a k-hop sampled subgraph around a head node by building it edge-by-edge
    and then converts it to a PyG Data object.
    """
    if head_node not in main_graph:
        return None

    # --- 1. Edge-Centric Subgraph Extraction (BFS-based) ---
    subgraph = nx.MultiDiGraph()
    subgraph.add_node(head_node)

    frontier = {head_node}
    visited = {head_node}

    for _ in range(k_hops):
        next_frontier = set()
        for node in frontier:
            if node in main_graph:
                # Combine successors and predecessors for a richer context
                successors = list(main_graph.successors(node))
                predecessors = list(main_graph.predecessors(node))
                neighbors = list(set(successors + predecessors))

                if len(neighbors) > neighbor_sample_size:
                    neighbors = random.sample(neighbors, neighbor_sample_size)

                for neighbor in neighbors:
                    # Explicitly add the edges to build a connected graph
                    if main_graph.has_edge(node, neighbor):
                        for rel_key in main_graph.get_edge_data(node, neighbor):
                            subgraph.add_edge(node, neighbor, key=rel_key)
                    if main_graph.has_edge(neighbor, node):
                        for rel_key in main_graph.get_edge_data(neighbor, node):
                            subgraph.add_edge(neighbor, node, key=rel_key)

                    next_frontier.add(neighbor)

        newly_discovered = next_frontier - visited
        frontier = newly_discovered
        visited.update(newly_discovered)

        if not frontier:
            break

    # --- Robustness Check ---
    if subgraph.number_of_edges() == 0:
        return None

    # --- 2. PyG Data Object Conversion ---
    node_list = list(subgraph.nodes())
    if head_node in node_list:
        node_list.remove(head_node)
    node_list = [head_node] + node_list
    node_map = {node: i for i, node in enumerate(node_list)}

    source, target, rels = [], [], []
    for u, v, rel in subgraph.edges(keys=True):
        source.append(node_map[u])
        target.append(node_map[v])
        rels.append(rel)

    edge_index = torch.tensor([source, target], dtype=torch.long)
    edge_attr = torch.tensor(rels, dtype=torch.long)

    # --- 3. Rich Node Feature Engineering ---
    try:
        distances = nx.shortest_path_length(subgraph, source=head_node)
    except nx.NodeNotFound:
        return None

    features = []
    for node in node_list:
        is_head = 1.0 if node == head_node else 0.0
        dist = float(distances.get(node, -1))
        hub_score = math.log(main_graph.degree(node) + 1.0)
        features.append([is_head, 1.0 - is_head, dist, hub_score])

    data = Data(x=torch.tensor(features, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
    data.node_ids = torch.tensor(node_list, dtype=torch.long)

    return data


# =====================================================================================
# 2. Parallel Processing Setup
# =====================================================================================

worker_graph = None
worker_inv_rel_map = None


def init_worker(graph_data, inv_rel_map_data):
    """Initializes each worker process with shared data."""
    global worker_graph, worker_inv_rel_map
    worker_graph = graph_data
    worker_inv_rel_map = inv_rel_map_data


def process_node(args):
    """Worker function to process a single node."""
    node_id, k_hops, neighbor_sample_size, save_path = args

    if os.path.exists(save_path):
        return node_id, "skipped"

    try:
        data_object = extract_sampled_subgraph_and_convert(
            worker_graph,
            node_id,
            k_hops,
            neighbor_sample_size,
            worker_inv_rel_map
        )

        if data_object:
            torch.save(data_object, save_path)
            return node_id, "success"
        else:
            if node_id not in worker_graph or worker_graph.degree(node_id) == 0:
                return node_id, "failed (isolated node)"
            else:
                return node_id, "failed (empty subgraph)"
    except Exception as e:
        return node_id, f"failed (error: {e})"


# =====================================================================================
# 3. Main Execution Block (Multi-Process)
# =====================================================================================
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
        print("--- Multiprocessing start method set to 'spawn' ---")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Extract head-centric subgraphs for all nodes.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes.")
    args = parser.parse_args()

    # --- 1. Load Data ---
    print(f"Loading data for dataset: {args.dataset}")
    df_train, ent_map, _ = load_data(f'datasets/{args.dataset}/train.txt', f'datasets/{args.dataset}/entities.dict',
                                     f'datasets/{args.dataset}/relations.dict')
    main_graph = input_graph(df_train)
    rel_inv_map = construct_inverse_map(f'datasets/{args.dataset}/relations.dict')

    base_save_path = f'sub_graphs/{args.dataset}_{args.hops}_{args.neighbors}_head_centric'
    os.makedirs(base_save_path, exist_ok=True)
    print(f"Subgraphs will be saved to: {base_save_path}")

    # --- 2. Prepare Tasks ---
    tasks = []
    # --- YOUR FIX IS HERE ---
    all_nodes = list(main_graph.nodes())
    print(f"Preparing tasks for {len(all_nodes)} nodes found in the graph.")
    for node_id in all_nodes:
        save_path = os.path.join(base_save_path, f'{node_id}.pt')
        tasks.append((node_id, args.hops, args.neighbors, save_path))

    # --- 3. Run Parallel Processing ---
    print(f"Starting subgraph extraction using {args.workers} workers...")

    success_count = 0
    skipped_count = 0
    failed_count = 0

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(main_graph, rel_inv_map)) as executor:

        future_to_task = {executor.submit(process_node, task): task for task in tasks}

        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks)):
            task = future_to_task[future]
            try:
                node_id, status = future.result()
                if status == "success":
                    success_count += 1
                elif status == "skipped":
                    skipped_count += 1
                else:
                    failed_count += 1
            except Exception:
                print(f"\n--- A WORKER PROCESS FAILED on task for node {task[0]} ---")
                traceback.print_exc()
                failed_count += 1

    print(f"\n--- Preprocessing Complete ---")
    print(f"Successfully generated: {success_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Total: {len(all_nodes)}")
