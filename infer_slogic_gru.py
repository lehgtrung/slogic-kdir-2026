import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool
from torch_geometric.data import Batch, Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import os
import argparse
from tqdm import tqdm
import pandas as pd


# =====================================================================================
# 1. The Model Definition (Must be identical to the one used for training)
# =====================================================================================
class HeadSubgraphGRUModel(torch.nn.Module):
    def __init__(self, num_relations, rel_emb_dim, gnn_hidden_dim, num_gnn_layers,
                 node_feature_dim, use_gru=True, gru_hidden_dim=128, num_gru_layers=1,
                 use_rule_stats=True, dropout=0.5):
        super(HeadSubgraphGRUModel, self).__init__()
        self.use_gru = use_gru
        self.use_rule_stats = use_rule_stats
        self.relation_emb = nn.Embedding(num_relations + 1, rel_emb_dim, padding_idx=num_relations)
        self.gnn_layers = nn.ModuleList()
        gnn_input_dim = node_feature_dim
        for i in range(num_gnn_layers):
            in_dim = gnn_input_dim if i == 0 else gnn_hidden_dim
            self.gnn_layers.append(
                RGCNConv(in_channels=in_dim, out_channels=gnn_hidden_dim, num_relations=num_relations)
            )
        if self.use_gru:
            self.gru = nn.GRU(input_size=rel_emb_dim, hidden_size=gru_hidden_dim,
                              num_layers=num_gru_layers, batch_first=True)
        mlp_input_dim = gnn_hidden_dim * 2 + rel_emb_dim
        if self.use_gru:
            mlp_input_dim += gru_hidden_dim
        if self.use_rule_stats:
            mlp_input_dim += 4
        self.fc1 = nn.Linear(mlp_input_dim, mlp_input_dim // 2)
        self.fc2 = nn.Linear(mlp_input_dim // 2, 1)
        self.dropout = nn.Dropout(p=dropout)

    def embed_metapath_gru(self, metapath_sequences: torch.Tensor):
        embedded_seqs = self.relation_emb(metapath_sequences)
        _, final_hidden_state = self.gru(embedded_seqs)
        return final_hidden_state.squeeze(0)

    def forward(self, subgraph_batch: Batch, query_relation_ids: torch.Tensor,
                metapath_sequences: torch.Tensor = None, rule_stats: torch.Tensor = None):
        x = subgraph_batch.x
        for i, layer in enumerate(self.gnn_layers):
            x = F.relu(layer(x, subgraph_batch.edge_index, subgraph_batch.edge_attr))
            if i < len(self.gnn_layers) - 1:
                x = self.dropout(x)
        head_indices_in_batch = subgraph_batch.ptr[:-1]
        head_node_embs = x[head_indices_in_batch]
        graph_embs = global_mean_pool(x, subgraph_batch.batch)
        query_relation_embs = self.relation_emb(query_relation_ids)
        combined_features = [head_node_embs, graph_embs, query_relation_embs]
        if self.use_gru:
            metapath_embs = self.embed_metapath_gru(metapath_sequences)
            combined_features.append(metapath_embs)
        if self.use_rule_stats:
            rule_stats = rule_stats.to(x.device, dtype=torch.float)
            combined_features.append(rule_stats)
        combined_vector = torch.cat(combined_features, dim=-1)
        score = F.relu(self.fc1(combined_vector))
        score = self.dropout(score)
        score = self.fc2(score)
        return score.squeeze(-1)


# =====================================================================================
# 2. Custom Dataset and Collate for Inference
# =====================================================================================
class InferenceDataset(Dataset):
    def __init__(self, root, metadata_df):
        super().__init__()
        self.root = root
        self.metadata_df = metadata_df

    def len(self):
        return len(self.metadata_df)

    def get(self, idx):
        row = self.metadata_df.iloc[idx]
        head_node_id = row['head']
        subgraph_path = os.path.join(self.root, f"{head_node_id}.pt")

        try:
            subgraph_data = torch.load(subgraph_path)
        except FileNotFoundError:
            return None, None

        return subgraph_data, row


def inference_collate_fn(batch):
    """
    Custom collate function for inference.
    """
    valid_samples = [item for item in batch if item[0] is not None]
    if not valid_samples:
        return None, None

    subgraph_data_list, metadata_rows = zip(*valid_samples)
    subgraph_batch = Batch.from_data_list(subgraph_data_list)
    metadata_batch_df = pd.DataFrame(list(metadata_rows))

    return subgraph_batch, metadata_batch_df


# =====================================================================================
# 3. Main Inference Script
# =====================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Score rules in a test file using the trained Head Subgraph + GRU model.")
    parser.add_argument("--dataset", help="Name of dataset", required=True)
    parser.add_argument("--test_file", type=str, required=False, help="Path to the test TSV file.")
    parser.add_argument("--num_rules", type=int, required=False,
                        help="Number of test rules - alternative to test file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model state_dict (.pt file).")
    parser.add_argument("--subgraph_dir", type=str, required=False,
                        help="Directory with preprocessed head subgraph files.")
    parser.add_argument("--num_relations", type=int, required=False, help="Total number of relations in the KG.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for inference.")
    parser.add_argument("--rel_emb_dim", type=int, default=128)
    parser.add_argument("--gnn_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability used by the model.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--num_gnn_layers", type=int, default=0, help="Number of GNN layers.")
    parser.add_argument("--split", help="test/valid", required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    if args.subgraph_dir is None:
        if args.dataset == 'wn18rr':
            args.subgraph_dir = f'sub_graphs/{args.dataset}_4_100_head_centric'
        elif args.dataset == 'FB15k-237':
            args.subgraph_dir = f'sub_graphs/{args.dataset}_1_100_head_centric'
        elif args.dataset == 'yago3-10':
            args.subgraph_dir = f'sub_graphs/{args.dataset}_1_100_head_centric'
        elif args.dataset == 'nations':
            args.subgraph_dir = f'sub_graphs/{args.dataset}_2_100_head_centric'
        else:
            raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.test_file is not None and args.num_rules is not None:
        raise ValueError('Test file and number of test rules cannot be specified at the same time')

    if args.test_file is None and args.num_rules is None:
        raise ValueError('Either test_file or num_test_rules must be specified')

    if args.num_rules:
        args.test_file = f'data/{args.dataset}/{args.split}_wilson_top_{args.num_rules}.tsv'

    # --- 1. Load the Trained Model ---
    print(f"Loading model from {args.model_path}...")
    # Instantiate the model with the same architecture as during training
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
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")

    # --- 2. Setup Inference DataLoader ---
    print(f"Loading test data from {args.test_file}...")
    test_df = pd.read_csv(args.test_file, sep='\t')
    dataset = InferenceDataset(root=args.subgraph_dir, metadata_df=test_df)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=inference_collate_fn)

    # --- 3. Run Batched Inference ---
    all_scores = torch.full((len(test_df),), float('-inf'))  # Initialize all scores to -inf

    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring rules in batches"):
            if batch is None or batch[0] is None:
                continue

            subgraph_batch, batch_metadata = batch
            subgraph_batch = subgraph_batch.to(device)

            # Get the original indices of the rows in this valid batch
            original_indices = batch_metadata.index.values

            # Prepare inputs for the model
            query_relations = torch.tensor(batch_metadata['relation'].values, dtype=torch.long).to(device)
            stats = torch.tensor(
                batch_metadata[['support', 'confidence', 'laplace_confidence', 'wilson_lower_bound']].values,
                dtype=torch.float).to(device)
            seqs = [torch.tensor([int(r) for r in str(s).split('|')]) for s in
                    batch_metadata['rule']]  # Assuming column is named 'rule'
            metapaths = pad_sequence(seqs, batch_first=True, padding_value=args.num_relations).to(device)

            # Get scores for the batch
            batch_scores = model(subgraph_batch, query_relations, metapath_sequences=metapaths, rule_stats=stats)

            # Place the scores back into the correct positions in the full results tensor
            all_scores[original_indices] = batch_scores.cpu()

    # --- 4. Add scores to DataFrame and Save ---
    test_df['score'] = all_scores.tolist()

    # Handle cases where a subgraph was missing by filling with a very low score
    test_df['score'].replace(float('-inf'), -999, inplace=True)

    output_filename = args.test_file.replace('.tsv', '_scored.tsv')
    print(f"\nSaving results with new 'score' column to: {output_filename}")
    test_df.to_csv(output_filename, sep='\t', index=False)
    print("Inference complete.")
