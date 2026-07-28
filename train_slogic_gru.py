import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool
from torch_geometric.data import Batch, Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import construct_inverse_map

import os
import argparse
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# --- NEW: Imports for seeding ---
import random
import numpy as np


# =====================================================================================
# 0. Reproducibility Helper Function
# =====================================================================================

def set_seed(seed: int):
    """Sets the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # --- NEW: For full CUDNN determinism ---
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    Sets the seed for a DataLoader worker.
    Ensures that data shuffling and augmentations are reproducible
    across different runs.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =====================================================================================
# 1. The Model Definition
# =====================================================================================
class HeadSubgraphGRUModel(torch.nn.Module):
    """
    A hybrid model that uses an R-GCN for a head-centric subgraph and a GRU
    for the metapath sequence.
    """

    def __init__(self, num_relations, rel_emb_dim, gnn_hidden_dim, num_gnn_layers,
                 node_feature_dim, use_gru=True, gru_hidden_dim=128, num_gru_layers=1,
                 use_rule_stats=True, dropout=0.5):
        super(HeadSubgraphGRUModel, self).__init__()
        self.use_gru = use_gru
        self.use_rule_stats = use_rule_stats

        # --- Embedding Layer (Only for Relations) ---
        self.relation_emb = nn.Embedding(num_relations + 1, rel_emb_dim, padding_idx=num_relations)

        # --- GNN Layers for Head Subgraph ---
        self.gnn_layers = nn.ModuleList()
        gnn_input_dim = node_feature_dim
        for i in range(num_gnn_layers):
            in_dim = gnn_input_dim if i == 0 else gnn_hidden_dim
            self.gnn_layers.append(
                RGCNConv(in_channels=in_dim, out_channels=gnn_hidden_dim, num_relations=num_relations)
            )

        # --- GRU for Metapath Sequence ---
        if self.use_gru:
            self.gru = nn.GRU(input_size=rel_emb_dim,
                              hidden_size=gru_hidden_dim,
                              num_layers=num_gru_layers,
                              batch_first=True)

        # --- Final Scoring MLP ---
        mlp_input_dim = gnn_hidden_dim * 2 + rel_emb_dim
        if self.use_gru:
            mlp_input_dim += gru_hidden_dim
        if self.use_rule_stats:
            mlp_input_dim += 4

        self.fc1 = nn.Linear(mlp_input_dim, mlp_input_dim // 2)
        self.fc2 = nn.Linear(mlp_input_dim // 2, 1)
        self.dropout = nn.Dropout(p=dropout)

    def embed_metapath_gru(self, metapath_sequences: torch.Tensor):
        """Encodes a batch of metapath sequences using the GRU."""
        embedded_seqs = self.relation_emb(metapath_sequences)
        _, final_hidden_state = self.gru(embedded_seqs)
        return final_hidden_state.squeeze(0)

    def forward(self, subgraph_batch: Batch, query_relation_ids: torch.Tensor,
                metapath_sequences: torch.Tensor = None,
                rule_stats: torch.Tensor = None):
        """
        Forward pass for a batch of head-centric subgraphs.
        """
        # --- 1. GNN on Head Subgraph ---
        x = subgraph_batch.x
        for i, layer in enumerate(self.gnn_layers):
            x = F.relu(layer(x, subgraph_batch.edge_index, subgraph_batch.edge_attr))
            if i < len(self.gnn_layers) - 1:
                x = self.dropout(x)

        # --- 2. Readout from GNN ---
        head_indices_in_batch = subgraph_batch.ptr[:-1]
        head_node_embs = x[head_indices_in_batch]
        graph_embs = global_mean_pool(x, subgraph_batch.batch)

        # --- 3. Prepare other features ---
        query_relation_embs = self.relation_emb(query_relation_ids)

        # --- 4. Combine All Features for Final Scoring ---
        combined_features = [head_node_embs, graph_embs, query_relation_embs]

        if self.use_gru:
            if metapath_sequences is None:
                raise ValueError("metapath_sequences must be provided when use_gru is True.")
            metapath_embs = self.embed_metapath_gru(metapath_sequences)
            combined_features.append(metapath_embs)

        if self.use_rule_stats:
            if rule_stats is None:
                raise ValueError("rule_stats must be provided when use_rule_stats is True.")
            rule_stats = rule_stats.to(x.device, dtype=torch.float)
            combined_features.append(rule_stats)

        combined_vector = torch.cat(combined_features, dim=-1)

        score = F.relu(self.fc1(combined_vector))
        score = self.dropout(score)
        score = self.fc2(score)

        return score.squeeze(-1)


# =====================================================================================
# 2. The Custom Dataset and Collate Function
# =====================================================================================
class HeadSubgraphDataset(Dataset):
    def __init__(self, root: str, metadata_df: pd.DataFrame, rel_inv_map):
        """
        Args:
            root (str): The root directory where the subgraph .pt files are stored.
            metadata_df (pd.DataFrame): DataFrame containing the training triplets
                                        and associated rule information.
            rel_inv_map (Dict[int, int]): A dictionary mapping each relation ID
                                          to its inverse relation ID.
        """
        super().__init__()
        self.root = root
        self.metadata_df = metadata_df
        self.rel_inv_map = rel_inv_map

    def len(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.metadata_df)

    def get(self, idx: int):
        """
        Loads one training sample, consisting of the subgraph and its metadata.
        This method implements the core logic for preventing data leakage.
        """
        row = self.metadata_df.iloc[idx]
        head, rel, tail = row['head'], row['relation'], row['tail']
        head_node_id = row['head']
        subgraph_path = os.path.join(self.root, f"{head_node_id}.pt")

        try:
            # Clone the data to prevent modifying a cached object. This is crucial
            # as multiple training samples might share the same head subgraph.
            subgraph_data = torch.load(subgraph_path).clone()
        except FileNotFoundError:
            # If a subgraph for a specific head doesn't exist, we skip this sample.
            # The collate_fn will handle filtering this out.
            return None, None

        # --- DATA LEAKAGE PREVENTION ---
        # The goal is to remove the direct edge (h, r, t) and its inverse
        # from the loaded subgraph so the GNN cannot use it as a shortcut.

        # 1. Find the local indices for the head and tail within the subgraph's node list.
        # We create a temporary mapping for efficient lookup.
        node_id_to_local_idx = {node_id.item(): i for i, node_id in enumerate(subgraph_data.node_ids)}
        head_local_idx = node_id_to_local_idx.get(head)
        tail_local_idx = node_id_to_local_idx.get(tail)

        # If the head or tail aren't in the subgraph for some reason, we can't proceed.
        if head_local_idx is None or tail_local_idx is None:
            return subgraph_data, row

        # 2. Get the inverse relation.
        inv_rel = self.rel_inv_map.get(rel)

        # 3. Create boolean masks to identify edges we want to *keep*.
        # An edge is defined by its source, target, and relation.
        edge_index = subgraph_data.edge_index
        edge_attr = subgraph_data.edge_attr

        # Keep all edges that are NOT the forward (h, r, t) edge.
        mask_forward = ~((edge_index[0] == head_local_idx) &
                         (edge_index[1] == tail_local_idx) &
                         (edge_attr == rel))

        # Keep all edges that are NOT the inverse (t, r_inv, h) edge.
        if inv_rel is not None:
            mask_inverse = ~((edge_index[0] == tail_local_idx) &
                             (edge_index[1] == head_local_idx) &
                             (edge_attr == inv_rel))
            # Combine the masks: an edge is kept if it's not the forward AND not the inverse.
            final_mask = mask_forward & mask_inverse
        else:
            final_mask = mask_forward

        # 4. Apply the mask to filter the edge attributes and indices.
        subgraph_data.edge_index = edge_index[:, final_mask]
        subgraph_data.edge_attr = edge_attr[final_mask]

        return subgraph_data, row


def custom_collate_fn(batch):
    """
    Custom collate function that handles missing subgraphs and batches the rest.
    """
    valid_samples = [item for item in batch if item[0] is not None]
    if not valid_samples:
        return None, None

    subgraph_data_list, metadata_rows = zip(*valid_samples)
    subgraph_batch = Batch.from_data_list(subgraph_data_list)
    metadata_batch_df = pd.DataFrame(list(metadata_rows))

    return subgraph_batch, metadata_batch_df


# =====================================================================================
# 3. The Main Training Script
# =====================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Head Subgraph + GRU model.")
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--subgraph_dir", type=str, required=True)
    parser.add_argument("--training_csv", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save the model and logs.")
    parser.add_argument("--device", type=str, required=False, help="Device", default='cuda')
    parser.add_argument("--num_relations", type=int, required=False)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--rel_emb_dim", type=int, default=128)
    parser.add_argument("--gnn_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability for the model.")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=4)
    # --- NEW: Argument for resuming training ---
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a model checkpoint to resume training from.")
    parser.add_argument("--num_gnn_layers", type=int, default=0, help="Number of GNN layers.")

    # --- NEW: Argument for seeding ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    # --- NEW: Set seed for reproducibility ---
    set_seed(args.seed)
    print(f"Using random seed: {args.seed}")

    if args.num_relations is None:
        if args.dataset == 'wn18rr':
            args.num_relations = 22
        elif args.dataset == 'FB15k-237':
            args.num_relations = 474
        elif args.dataset == 'yago3-10':
            args.num_relations = 74
        elif args.dataset == 'nations':
            args.num_relations = 112
        else:
            raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.num_gnn_layers == 0:
        if args.dataset == 'wn18rr':
            args.num_gnn_layers = 2
        elif args.dataset == 'FB15k-237':
            args.num_gnn_layers = 1
        elif args.dataset == 'yago3-10':
            args.num_gnn_layers = 1
        elif args.dataset == 'nations':
            args.num_gnn_layers = 1
        else:
            raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.batch_size is None:
        if args.dataset == 'wn18rr':
            args.batch_size = 256
        elif args.dataset == 'FB15k-237':
            args.num_relations = 10240
        elif args.dataset == 'yago3-10':
            args.num_relations = 10240
        else:
            raise ValueError('Unknown dataset: {}'.format(args.dataset))

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Num GNN layers: {args.num_gnn_layers}")

    # --- Setup Save Directory and Logging ---
    os.makedirs(args.save_path, exist_ok=True)
    log_file_path = os.path.join(args.save_path, 'training_log.tsv')

    # --- Load Datasets ---
    print("Loading training metadata from CSV...")
    metadata_df = pd.read_csv(args.training_csv, sep='\t')
    inverse_map = construct_inverse_map(f'datasets/{args.dataset}/relations.dict')
    dataset = HeadSubgraphDataset(root=args.subgraph_dir, metadata_df=metadata_df, rel_inv_map=inverse_map)

    # --- NEW: Create a generator for reproducible DataLoader shuffling ---
    g = torch.Generator()
    g.manual_seed(args.seed)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=custom_collate_fn)

    # --- Initialize Model and Optimizer ---
    model = HeadSubgraphGRUModel(
        num_relations=args.num_relations,
        rel_emb_dim=args.rel_emb_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        node_feature_dim=4,
        use_gru=True,
        use_rule_stats=True,
        dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    margin_loss = nn.MarginRankingLoss(margin=1.0)

    # --- NEW: Logic for Loading a Checkpoint ---
    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming training from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        print(f"Resuming from epoch {start_epoch}")

    # --- Start Training ---
    # --- NEW: Determine write mode for log file ---
    log_mode = 'a' if start_epoch > 0 else 'w'
    with open(log_file_path, log_mode) as log_file:
        if log_mode == 'w':
            log_file.write("timestamp\tepoch\taverage_loss\n")  # Write header only on new run

        # --- NEW: Use start_epoch in the training loop ---
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total_loss = 0

            progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch in progress_bar:
                if batch is None or batch[0] is None:
                    continue

                subgraph_batch, batch_metadata = batch
                subgraph_batch = subgraph_batch.to(device)

                # --- Prepare Positive Inputs ---
                query_relations_pos = torch.tensor(batch_metadata['relation'].values, dtype=torch.long).to(device)
                pos_stats = torch.tensor(batch_metadata[['pos_support', 'pos_confidence', 'pos_laplace_confidence',
                                                         'pos_wilson_lower_bound']].values, dtype=torch.float).to(
                    device)
                pos_seqs = [torch.tensor([int(r) for r in str(s).split('|')]) for s in batch_metadata['pos_body']]
                pos_metapaths = pad_sequence(pos_seqs, batch_first=True, padding_value=args.num_relations).to(device)

                # --- Prepare Negative Inputs ---
                query_relations_neg = query_relations_pos
                neg_stats = torch.tensor(batch_metadata[['neg_support', 'neg_confidence', 'neg_laplace_confidence',
                                                         'neg_wilson_lower_bound']].values, dtype=torch.float).to(
                    device)
                neg_seqs = [torch.tensor([int(r) for r in str(s).split('|')]) for s in batch_metadata['neg_body']]
                neg_metapaths = pad_sequence(neg_seqs, batch_first=True, padding_value=args.num_relations).to(device)

                # --- Forward and Backward Pass ---
                optimizer.zero_grad()
                pos_scores = model(subgraph_batch, query_relations_pos, metapath_sequences=pos_metapaths,
                                   rule_stats=pos_stats)
                neg_scores = model(subgraph_batch, query_relations_neg, metapath_sequences=neg_metapaths,
                                   rule_stats=neg_stats)
                target = torch.ones_like(pos_scores)
                loss = margin_loss(pos_scores, neg_scores, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})

            avg_loss = total_loss / len(loader)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            log_line = f"{timestamp}\t{epoch + 1}\t{avg_loss:.4f}\n"
            log_file.write(log_line)
            log_file.flush()
            print(f"Epoch {epoch + 1}/{args.epochs} - Average Loss: {avg_loss:.4f}")

            # --- MODIFIED: Save a more complete checkpoint ---
            model_save_path = os.path.join(args.save_path, f'saved_model_{epoch + 1}.pt')
            print(f"Saving model checkpoint to {model_save_path}...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, model_save_path)

    print("--- Training Complete ---")
