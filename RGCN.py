from pykeen.pipeline import pipeline

#  R-GCN TU Chemnitz
rgcn_tuc = pipeline(
    training=tuc_train,
    testing=tuc_test,
    validation=tuc_val,
    model='RGCN',
    random_seed=42,
    model_kwargs=dict(embedding_dim=100, num_layers=2),
    training_kwargs=dict(num_epochs=100, use_tqdm_batch=False, sampler='schlichtkrull'),
    optimizer='Adam',
    optimizer_kwargs=dict(lr=0.001),
    device=device,
)
rg_tuc_mrr= rgcn_tuc.get_metric('inverse_harmonic_mean_rank')
rg_tuc_hits10=rgcn_tuc.get_metric('hits_at_10')
rg_tuc_mr= rgcn_tuc.get_metric('mean_rank')
print(f'R-GCN TUC MRR={rg_tuc_mrr:.4f}  Hits@10={rg_tuc_hits10:.4f}  MR={rg_tuc_mr:.1f}')

#R-GCN University of Girona 
rgcn_girona = pipeline(
    training=girona_train,
    testing=girona_test,
    validation=girona_val,
    model='RGCN',
    random_seed=42,
    model_kwargs=dict(embedding_dim=100, num_layers=2),
    training_kwargs=dict(num_epochs=100, use_tqdm_batch=False, sampler='schlichtkrull'),
    optimizer='Adam',
    optimizer_kwargs=dict(lr=0.001),
    device=device,
)
rg_girona_mrr = rgcn_girona.get_metric('inverse_harmonic_mean_rank')
rg_girona_hits10= rgcn_girona.get_metric('hits_at_10')
rg_girona_mr= rgcn_girona.get_metric('mean_rank')
print(f'R-GCN Girona MRR={rg_girona_mrr:.4f}  Hits@10={rg_girona_hits10:.4f}  MR={rg_girona_mr:.1f}')
# R-GCN University of Udine
rgcn_udine = pipeline(
    training=udine_train,
    testing=udine_test,
    validation=udine_val,
    model='RGCN',
    random_seed=42,
    model_kwargs=dict(embedding_dim=100, num_layers=2),
    training_kwargs=dict(num_epochs=100, use_tqdm_batch=False, sampler='schlichtkrull'),
    optimizer='Adam',
    optimizer_kwargs=dict(lr=0.001),
    device=device,
)
rg_udine_mrr= rgcn_udine.get_metric('inverse_harmonic_mean_rank')
rg_udine_hits10= rgcn_udine.get_metric('hits_at_10')
rg_udine_mr  = rgcn_udine.get_metric('mean_rank')
print(f'R-GCN Udine  MRR={rg_udine_mrr:.4f}  Hits@10={rg_udine_hits10:.4f}  MR={rg_udine_mr:.1f}')