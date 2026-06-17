import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample

def get_embeddings(result):
    repr_layer = result.model.entity_representations[0]
    emb = repr_layer().detach().cpu().numpy()
    return emb, result.training.entity_to_id

def get_instance_embeddings(emb, entity_to_id, instance_ids, prefix='course-instance'):
    vectors, found = [], []
    for iid in instance_ids:
        key = f'{prefix}/{iid}'
        if key not in entity_to_id:
            continue
        vectors.append(emb[entity_to_id[key]])
        found.append(iid)
    return np.array(vectors), found

def run_isolation_forest(emb, found_ids, course_df, contamination):
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(emb)
    preds = (clf.predict(emb) == -1).astype(int)
    scores = clf.decision_function(emb)
    true_labels = course_df[course_df['instance_id'].isin(found_ids)]['label'].values
    return preds, scores, true_labels

def bootstrap_f1(predictions, ground_truth, n_iterations=1000):
    scores = []
    for _ in range(n_iterations):
        idx = resample(range(len(ground_truth)), random_state=None)
        scores.append(f1_score(ground_truth[idx], predictions[idx], zero_division=0))
    return np.array(scores)

def is_shacl_violation(row):
    if pd.isna(row['location']) or row['location'] == '':
        return 1
    if pd.isna(row['start_date']) or pd.isna(row['end_date']):
        return 1
    if pd.notna(row['start_date']) and pd.notna(row['end_date']) and row['end_date'] <= row['start_date']:
        return 1
    if pd.isna(row['ects_credits']) or not (1 <= row['ects_credits'] <= 30):
        return 1
    if pd.notna(row['enrollment']) and row['enrollment'] > 500:
        return 1
    if pd.isna(row['instructor_email']) or row['instructor_email'] == '':
        return 1
    return 0

transe_embeddings, transe_entity_to_id = get_embeddings(transe_result)
distmult_embeddings, distmult_entity_to_id = get_embeddings(distmult_result)

rgcn_tuc_embeddings, rgcn_tuc_entity_to_id = get_embeddings(rgcn_tuc)
rgcn_girona_embeddings, rgcn_girona_entity_to_id = get_embeddings(rgcn_girona)
rgcn_udine_embeddings, rgcn_udine_entity_to_id = get_embeddings(rgcn_udine)

compgcn_tuc_embeddings, compgcn_tuc_entity_to_id = get_embeddings(compgcn_tuc)
compgcn_girona_embeddings, compgcn_girona_entity_to_id = get_embeddings(compgcn_girona)
compgcn_udine_embeddings, compgcn_udine_entity_to_id = get_embeddings(compgcn_udine)

# get instance-level embeddings per university
tuc_instance_ids = csv_df[csv_df['provider'] == 'TU Chemnitz']['instance_id'].tolist()
girona_instance_ids = csv_df[csv_df['provider'] == 'University of Girona']['instance_id'].tolist()
udine_instance_ids = csv_df[csv_df['provider'] == 'University of Udine']['instance_id'].tolist()
all_instance_ids = csv_df['instance_id'].tolist()

transe_inst_emb, transe_found = get_instance_embeddings(transe_embeddings, transe_entity_to_id, all_instance_ids)
distmult_inst_emb, distmult_found = get_instance_embeddings(distmult_embeddings, distmult_entity_to_id, all_instance_ids)

rgcn_tuc_inst_emb, rgcn_tuc_found = get_instance_embeddings(rgcn_tuc_embeddings, rgcn_tuc_entity_to_id, tuc_instance_ids)
rgcn_girona_inst_emb, rgcn_girona_found = get_instance_embeddings(rgcn_girona_embeddings, rgcn_girona_entity_to_id, girona_instance_ids)
rgcn_udine_inst_emb, rgcn_udine_found = get_instance_embeddings(rgcn_udine_embeddings, rgcn_udine_entity_to_id, udine_instance_ids)

compgcn_tuc_inst_emb, compgcn_tuc_found = get_instance_embeddings(compgcn_tuc_embeddings, compgcn_tuc_entity_to_id, tuc_instance_ids)
compgcn_girona_inst_emb, compgcn_girona_found = get_instance_embeddings(compgcn_girona_embeddings, compgcn_girona_entity_to_id, girona_instance_ids)
compgcn_udine_inst_emb, compgcn_udine_found = get_instance_embeddings(compgcn_udine_embeddings, compgcn_udine_entity_to_id, udine_instance_ids)

transe_preds, transe_scores, transe_true = run_isolation_forest(transe_inst_emb, transe_found, csv_df, anomaly_ratio)
distmult_preds, distmult_scores, distmult_true = run_isolation_forest(distmult_inst_emb, distmult_found, csv_df, anomaly_ratio)

rgcn_tuc_preds, rgcn_tuc_scores, rgcn_tuc_true = run_isolation_forest(rgcn_tuc_inst_emb, rgcn_tuc_found, csv_df, anomaly_ratio)
rgcn_girona_preds, rgcn_girona_scores, rgcn_girona_true = run_isolation_forest(rgcn_girona_inst_emb, rgcn_girona_found, csv_df, anomaly_ratio)
rgcn_udine_preds, rgcn_udine_scores, rgcn_udine_true = run_isolation_forest(rgcn_udine_inst_emb, rgcn_udine_found, csv_df, anomaly_ratio)

compgcn_tuc_preds, compgcn_tuc_scores, compgcn_tuc_true = run_isolation_forest(compgcn_tuc_inst_emb, compgcn_tuc_found, csv_df, anomaly_ratio)
compgcn_girona_preds, compgcn_girona_scores, compgcn_girona_true = run_isolation_forest(compgcn_girona_inst_emb, compgcn_girona_found, csv_df, anomaly_ratio)
compgcn_udine_preds, compgcn_udine_scores, compgcn_udine_true = run_isolation_forest(compgcn_udine_inst_emb, compgcn_udine_found, csv_df, anomaly_ratio)

# SHACL+SPARQL
csv_found = csv_df[csv_df['instance_id'].isin(transe_found)].reset_index(drop=True)
shacl_predictions = csv_found.apply(is_shacl_violation, axis=1).values
ground_truth_found = csv_found['label'].values
shacl_precision = precision_score(ground_truth_found, shacl_predictions, zero_division=0)
shacl_recall = recall_score(ground_truth_found, shacl_predictions, zero_division=0)
shacl_f1 = f1_score(ground_truth_found, shacl_predictions, zero_division=0)

# combined
all_rgcn_preds = np.concatenate([rgcn_tuc_preds, rgcn_girona_preds, rgcn_udine_preds])
combined_predictions = ((shacl_predictions == 1) | (all_rgcn_preds == 1)).astype(int)
combined_precision = precision_score(ground_truth_found, combined_predictions, zero_division=0)
combined_recall = recall_score(ground_truth_found, combined_predictions, zero_division=0)
combined_f1 = f1_score(ground_truth_found, combined_predictions, zero_division=0)

all_models = {
    'TransE': (transe_preds,transe_true),
    'DistMult':(distmult_preds, distmult_true),
    'R-GCN TUC': (rgcn_tuc_preds, rgcn_tuc_true),
    'R-GCN Girona': (rgcn_girona_preds,rgcn_girona_true),
    'R-GCN Udine': (rgcn_udine_preds,  rgcn_udine_true),
    'CompGCN TUC':(compgcn_tuc_preds,  compgcn_tuc_true),
    'CompGCN Girona': (compgcn_girona_preds, compgcn_girona_true),
    'CompGCN Udine':(compgcn_udine_preds,compgcn_udine_true),
    'SHACL+SPARQL': (shacl_predictions,ground_truth_found),
    'Combined': (combined_predictions, ground_truth_found),
}

bootstrap_results = {}
for name, (preds, true_labels) in all_models.items():
    boot_scores = bootstrap_f1(preds, true_labels)
    low, high = np.percentile(boot_scores, [2.5, 97.5])
    bootstrap_results[name] = {
        'mean_f1': np.mean(boot_scores),
        'std': np.std(boot_scores),
        'ci_low': low,
        'ci_high': high,
    }

university_results = {}
universities = {
    'TU Chemnitz': (rgcn_tuc_preds,    rgcn_tuc_true,    rgcn_tuc_found),
    'University of Girona': (rgcn_girona_preds, rgcn_girona_true, rgcn_girona_found),
    'University of Udine':(rgcn_udine_preds,  rgcn_udine_true,  rgcn_udine_found),
}

for uni_name, (preds, true_labels, found) in universities.items():
    university_results[uni_name] = {
        'total': len(true_labels),
        'anomalies': int(true_labels.sum()),
        'precision': precision_score(true_labels, preds, zero_division=0),
        'recall': recall_score(true_labels, preds, zero_division=0),
        'f1': f1_score(true_labels, preds, zero_division=0),
    }