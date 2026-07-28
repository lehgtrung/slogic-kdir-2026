# SLogic

Code for the KDIR 2026 paper **SLogic: Subgraph-Informed Logical Rule Learning
for Knowledge Graph Completion**.

SLogic is a neuro-symbolic rule-ranking framework for knowledge graph
completion. It mines interpretable logical rules, extracts a subgraph around the
query head entity, and learns a query-dependent score for each candidate rule.

This repository is organized around the WN18RR experiment command used in the
paper:

```shell
EXTRACT_SUBGRAPHS=1 CORES=32 EPOCHS=15 DEVICE=cuda bash run_experiment wn18rr 50000
```

The command runs the following pipeline:

```text
sampled rule mining
-> head-centric subgraph extraction
-> locally applicable rule extraction
-> contrastive training data generation
-> R-GCN + GRU model training
-> valid/test candidate rule generation
-> neural rule scoring
-> final filtered ranking evaluation
```

## Files

- `run_experiment`: runs the full WN18RR experiment.
- `extract_rules_multi_cores.py`: mines cyclic/metapath rules from sampled training triples.
- `subgraph_extraction.py`: creates head-centric PyTorch Geometric subgraphs.
- `extract_local_applicable_rules_opt2.py`: finds locally applicable rules for training heads.
- `extract_training_data_optimized.py`: builds positive/negative rule-pair training data.
- `train_slogic_gru.py`: trains the SLogic R-GCN + GRU rule scorer.
- `extract_test_data_optimized.py`: builds validation/test candidate rule tables.
- `infer_slogic_gru.py`: scores candidate rules with the trained model.
- `run_score_parallel.py`: grounds scored rules and reports MRR/Hits@k.
- `utils.py`: shared loading, rule, graph, and checkpoint helpers.
- `datasets/wn18rr`: WN18RR train/validation/test files and dictionaries.

Generated files are written under `data/`, `sub_graphs/`, and `experiments/`.

## Installation

Use Python 3.10+.

Install PyTorch and PyTorch Geometric for your CUDA environment first, then:

```shell
pip install -r requirements.txt
```

## Run The WN18RR Experiment

```shell
EXTRACT_SUBGRAPHS=1 CORES=32 EPOCHS=15 DEVICE=cuda bash run_experiment wn18rr 50000
```

By default, the run uses seed `1024`. To use another seed, pass it as the third
argument:

```shell
EXTRACT_SUBGRAPHS=1 CORES=32 EPOCHS=15 DEVICE=cuda bash run_experiment wn18rr 50000 2048
```

For a small CPU smoke test:

```shell
EXTRACT_SUBGRAPHS=1 CORES=4 WORKERS=4 EPOCHS=1 DEVICE=cpu bash run_experiment wn18rr 1000
```

The main result file for the paper command is:

```text
data/wn18rr/experiments/samples_50000/result_samples_50000_seed_1024.txt
```

The full log is:

```text
data/wn18rr/experiments/samples_50000/run_samples_50000_seed_1024.log
```

## Citation

```bibtex
@inproceedings{slogic_kdir,
  title = {SLogic: Subgraph-Informed Logical Rule Learning for Knowledge Graph Completion},
  author = {Le, Trung Hoang and Son, Tran Cao and Ahmed, Ishtiaq and Cao, Huiping},
  booktitle = {Proceedings of the International Conference on Knowledge Discovery and Information Retrieval},
  year = {2026}
}
```
