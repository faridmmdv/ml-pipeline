from pykeen.pipeline import pipeline
# TransE
transe_result = pipeline(
    training=train_m,
    testing=test_m,
    validation=val_m,
    model='TransE',
    random_seed=42,
    model_kwargs=dict(embedding_dim=100),
    training_kwargs=dict(num_epochs=100, use_tqdm_batch=False),
    optimizer='Adam',
    optimizer_kwargs=dict(lr=0.001),
    device=device,
)
te_mrr= transe_result.get_metric('inverse_harmonic_mean_rank')
te_hits10= transe_result.get_metric('hits_at_10')
te_mr= transe_result.get_metric('mean_rank')
print(f'TransE MRR={te_mrr:.4f}  Hits@10={te_hits10:.4f}  MR={te_mr:.1f}')

#DISTMULT
distmult_result = pipeline(
    training=train_m,
    testing=test_m,
    validation=val_m,
    model='DistMult',
    random_seed=42,
    model_kwargs=dict(embedding_dim=100),
    training_kwargs=dict(num_epochs=100, use_tqdm_batch=False),
    optimizer='Adam',
    optimizer_kwargs=dict(lr=0.001),
    device=device,
)

dm_mrr= distmult_result.get_metric('inverse_harmonic_mean_rank')
dm_hits10= distmult_result.get_metric('hits_at_10')
dm_mr= distmult_result.get_metric('mean_rank')
print(f'DistMult  MRR={dm_mrr:.4f}  Hits@10={dm_hits10:.4f}  MR={dm_mr:.1f}')