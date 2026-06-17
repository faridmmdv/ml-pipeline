from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

#  CompGCN TU Chemnitz 
tf_tuc_comp = TriplesFactory.from_labeled_triples(df_tuc[['s','p','o']].values, create_inverse_triples=True)
tuc_train_c, tuc_test_c, tuc_val_c = tf_tuc_comp.split([0.8, 0.1, 0.1], random_state=42)

compgcn_tuc = pipeline(
    training=tuc_train_c,
    testing=tuc_test_c,
    validation=tuc_val_c,
    model='CompGCN',
    random_seed=42,
    model_kwargs=dict(embedding_dim=100),
    training_kwargs=dict(num_epochs=100, use_tqdm_batch=False),
    optimizer='Adam',
    optimizer_kwargs=dict(lr=0.001),
    device=device,
)
cg_tuc_mrr= compgcn_tuc.get_metric('inverse_harmonic_mean_rank')
cg_tuc_hits10= compgcn_tuc.get_metric('hits_at_10')
cg_tuc_mr = compgcn_tuc.get_metric('mean_rank')
print(f'CompGCN TUC MRR={cg_tuc_mrr:.4f}  Hits@10={cg_tuc_hits10:.4f}  MR={cg_tuc_mr:.1f}')

#CompGCN University of Girona
tf_girona_comp = TriplesFactory.from_labeled_triples(df_girona[['s','p','o']].values, create_inverse_triples=True)
girona_train_c, girona_test_c, girona_val_c = tf_girona_comp.split([0.8, 0.1, 0.1], random_state=42)

compgcn_girona = pipeline(
    training=girona_train_c,
    testing=girona_test_c,
    validation=girona_val_c,
    model='CompGCN',
    random_seed=42,
    model_kwargs=dict(embedding_dim=100),
    training_kwargs=dict(num_epochs=100, use_tqdm_batch=False),
    optimizer='Adam',
    optimizer_kwargs=dict(lr=0.001),
    device=device,
)
cg_girona_mrr= compgcn_girona.get_metric('inverse_harmonic_mean_rank')
cg_girona_hits10= compgcn_girona.get_metric('hits_at_10')
cg_girona_mr= compgcn_girona.get_metric('mean_rank')
print(f'CompGCN Girona MRR={cg_girona_mrr:.4f}  Hits@10={cg_girona_hits10:.4f}  MR={cg_girona_mr:.1f}')

# ── CompGCN University of Udine 
tf_udine_comp = TriplesFactory.from_labeled_triples(df_udine[['s','p','o']].values, create_inverse_triples=True)
udine_train_c, udine_test_c, udine_val_c = tf_udine_comp.split([0.8, 0.1, 0.1], random_state=42)

compgcn_udine = pipeline(
    training=udine_train_c,
    testing=udine_test_c,
    validation=udine_val_c,
    model='CompGCN',
    random_seed=42,
    model_kwargs=dict(embedding_dim=100),
    training_kwargs=dict(num_epochs=100, use_tqdm_batch=False),
    optimizer='Adam',
    optimizer_kwargs=dict(lr=0.001),
    device=device,
)

cg_udine_mrr= compgcn_udine.get_metric('inverse_harmonic_mean_rank')
cg_udine_hits10= compgcn_udine.get_metric('hits_at_10')
cg_udine_mr = compgcn_udine.get_metric('mean_rank')
print(f'CompGCN Udine  MRR={cg_udine_mrr:.4f}  Hits@10={cg_udine_hits10:.4f}  MR={cg_udine_mr:.1f}')