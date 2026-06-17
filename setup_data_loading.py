import torch
import pandas as pd
import numpy as np
from pykeen.triples import TriplesFactory

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using: {device}')

df_raw = pd.read_csv('triples_pykeen.tsv', sep='\t', header=None, names=['s', 'p', 'o'])
csv_df = pd.read_csv('course_instances.csv')
csv_df['start_date'] = pd.to_datetime(csv_df['start_date'], errors='coerce')
csv_df['end_date'] = pd.to_datetime(csv_df['end_date'], errors='coerce')

print(f'raw triples: {len(df_raw)}')


def is_anomaly(row):
    if pd.isna(row['location']) or row['location'] == '':
        return 1
    if pd.isna(row['instructor_email']) or row['instructor_email'] == '':
        return 1
    if pd.isna(row['description']) or row['description'] == '':
        return 1
    
    if pd.isna(row['ects_credits']) or not (1 <= row['ects_credits'] <= 30):
        return 1
    
    if pd.isna(row['start_date']) or pd.isna(row['end_date']):
        return 1
    if row['end_date'] <= row['start_date']:
        return 1

    if pd.notna(row['enrollment']) and row['enrollment'] > 500:
        return 1

    return 0

csv_df['label'] = csv_df.apply(is_anomaly, axis=1)
anomaly_ratio = csv_df['label'].mean()
print(f'total: {len(csv_df)},  anomalies: {csv_df["label"].sum()} ({anomaly_ratio*100:.1f}%)')


def shorten_uri(uri):
    if pd.isna(uri):
        return 'unknown'
    uri = str(uri).rstrip('/')
    parts = uri.split('/')
    if len(parts) >= 2 and parts[-2] in ('course-instance', 'course', 'person', 'place', 'university'):
        return f'{parts[-2]}/{parts[-1]}'
    return parts[-1].split('#')[-1]


def get_uni_triples(df, ids):
    uris = [f'https://example.org/resource/course-instance/{i}' for i in ids]
    mask = df['s'].isin(uris) | df['o'].isin(uris)
    out = df[mask].reset_index(drop=True).copy()
    for col in ['s', 'p', 'o']:
        out[col] = out[col].apply(shorten_uri)
    return out


tuc_ids = csv_df[csv_df['provider'] == 'TU Chemnitz']['instance_id'].tolist()
girona_ids = csv_df[csv_df['provider'] == 'University of Girona']['instance_id'].tolist()
udine_ids = csv_df[csv_df['provider'] == 'University of Udine']['instance_id'].tolist()

df_tuc = get_uni_triples(df_raw, tuc_ids)
df_girona = get_uni_triples(df_raw, girona_ids)
df_udine = get_uni_triples(df_raw, udine_ids)

tf_tuc = TriplesFactory.from_labeled_triples(df_tuc[['s','p','o']].values)
tf_girona = TriplesFactory.from_labeled_triples(df_girona[['s','p','o']].values)
tf_udine = TriplesFactory.from_labeled_triples(df_udine[['s','p','o']].values)
print(f'TUC: {tf_tuc.num_triples} triples, {tf_tuc.num_entities} entities')
print(f'Girona: {tf_girona.num_triples} triples, {tf_girona.num_entities} entities')
print(f'Udine: {tf_udine.num_triples} triples, {tf_udine.num_entities} entities')

tuc_train, tuc_test, tuc_val = tf_tuc.split([0.8, 0.1, 0.1], random_state=42)
girona_train, girona_test, girona_val = tf_girona.split([0.8, 0.1, 0.1], random_state=42)
udine_train, udine_test, udine_val = tf_udine.split([0.8, 0.1, 0.1], random_state=42)

df_merged = pd.concat([df_tuc, df_girona, df_udine]).reset_index(drop=True)
tf_merged = TriplesFactory.from_labeled_triples(df_merged[['s','p','o']].values)
train_m, test_m, val_m = tf_merged.split([0.8, 0.1, 0.1], random_state=42)

print(f'merged: {tf_merged.num_triples} triples, {tf_merged.num_entities} entities')
print(f'train/test/val: {train_m.num_triples} / {test_m.num_triples} / {val_m.num_triples}')