# Across KG — ML Anomaly Detection Pipeline

Knowledge graph embedding and GNN based anomaly detection, benchmarked against rule-based SHACL+SPARQL detection. Companion repo to the [Across KG Quality Platform](https://github.com/faridmmdv/across-kg-quality-platform).

MSc thesis project, TU Chemnitz — *Quality Assessment for Federated University Knowledge Graphs Using Machine-Learning-Based Anomaly Detection: A Case Study on the Across Alliance.*

## Architecture

```
setup_data_loading.py
   └── load triples + course data, label anomalies, build train/test/val splits

transE_and_DistMult.py
   └── PyKEEN embedding baselines, trained on merged graph

RGCN.py / COMPGCN.py
   └── PyTorch Geometric GNNs, trained per university (TUC / Girona / Udine)

anomaly_detection.py
   └── Isolation Forest on embeddings, evaluated vs. SHACL+SPARQL + combined detector
```

Scripts run sequentially in one Colab session — each depends on variables created by the ones before it (`device`, `csv_df`, the train/test/val splits, the trained model results). Not standalone modules.

## Pipeline steps

- **Data loading** — `setup_data_loading.py` reads `triples_pykeen.tsv` and `course_instances.csv`, labels each course instance as anomalous using a fixed rule set (missing location, missing instructor email, missing description, ECTS outside 1–30, invalid date range, enrollment over 500), splits triples per university and builds a merged graph, then creates 80/10/10 train/test/validation splits
- **Embedding baselines** — `transE_and_DistMult.py` trains TransE and DistMult on the merged graph (PyKEEN, `embedding_dim=100`, 100 epochs, Adam, lr=0.001), reports MRR, Hits@10, mean rank
- **GNN models** — `RGCN.py` and `COMPGCN.py` train one model per university each (not merged), same training config, R-GCN with `num_layers=2`, CompGCN with `create_inverse_triples=True`
- **Anomaly detection** — `anomaly_detection.py` extracts entity embeddings, isolates course-instance vectors, runs Isolation Forest per model, compares precision/recall/F1 against `is_shacl_violation` (rule-based) and a combined detector (SHACL+SPARQL OR R-GCN), bootstrapped over 1000 resamples for confidence intervals, plus a per-university breakdown for R-GCN

## Stack

PyTorch, PyKEEN, PyTorch Geometric, scikit-learn (Isolation Forest, metrics, bootstrap resampling), pandas, numpy

## Setup

```bash
# in Google Colab, GPU runtime
# upload triples_pykeen.tsv and course_instances.csv to the working directory
# run scripts in order, as cells in the same session:

setup_data_loading.py
transE_and_DistMult.py
RGCN.py
COMPGCN.py
anomaly_detection.py
```

`triples_pykeen.tsv` and `course_instances.csv` are exported from the Django database in the main platform repo. Place both files in the same working directory as these scripts before running `setup_data_loading.py`.

## Repo structure

```
setup_data_loading.py     data loading, labeling, train/test/val splits
transE_and_DistMult.py    TransE + DistMult on merged graph
RGCN.py                   R-GCN per university
COMPGCN.py                CompGCN per university
anomaly_detection.py      Isolation Forest + evaluation vs. SHACL+SPARQL
```

## Notes

Trained and run in Google Colab with a GPU runtime, not part of the Django/React platform's runtime. Random seed fixed at 42 throughout for reproducibility.

## Author

Farid Mammadov — MSc Web Engineering, TU Chemnitz
