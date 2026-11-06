"""Seed robustness (Section 3.6 / Table 7): every metric recomputed from the saved test predictions, so seeds 42, 43
and 44 are scored identically. Rows: benchmark ConvNeXtTiny (A0), A2b and A6 (seed 42 = the main runs).
Our binary task: A0 = binary-trained, A2b / A6 = 3-class output collapsed (argmax), as in the paper.
Output: results/revision/seeds_per_fold.csv, seeds_summary.csv
  python seed_analysis.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

NOV = Path(__file__).resolve().parent; CV = NOV.parent
PUB = CV / 'public' / 'results_public'; OUT = NOV / 'results' / 'revision'
TASKS = ['ours_3class', 'ours_binary', 'OCD3_2class', 'OCD2_2class', 'NDO_2class', 'NDO_3class', 'NDO_4class']
SEEDS = [42, 43, 44]


def bench_file(task, s, k):
    sub = '' if s == 42 else f'seeds/s{s}/'
    if task == 'ours_3class': d = CV / 'results' / sub / 'predictions'
    elif task == 'ours_binary': d = CV / 'results' / sub / 'binary' / 'predictions' if s != 42 else CV / 'results' / 'binary' / 'predictions'
    else: d = PUB / task / sub / 'predictions'
    return d / 'ConvNeXtTiny' / f'fold_{k:02d}_test_predictions.csv'


def novel_file(task, row, s, k):
    t = 'ours_3class' if task == 'ours_binary' else task
    return NOV / 'results' / t / (row if s == 42 else f'{row}_s{s}') / f'fold_{k:02d}' / 'test_predictions.csv'


def score(d, collapse=False):
    pc = [c for c in d.columns if c.startswith('p_')]
    order = d.groupby('y').label.first().sort_index().tolist()          # class index -> name
    cols = [f'p_{c}' for c in order]
    if not set(cols) <= set(d.columns): cols = pc                      # binary benchmark: p_non_cancer, p_cancer
    P = d[cols].values; y = d.y.values
    if collapse:                                                       # cancer vs non-cancer from the 3-class output
        m = order.index('malignant'); yb = (y == m).astype(int)
        return f1_score(yb, (P.argmax(1) == m).astype(int), average='macro'), roc_auc_score(yb, P[:, m])
    yp = P.argmax(1)
    auc = roc_auc_score(y, P[:, 1]) if P.shape[1] == 2 else roc_auc_score(y, P, multi_class='ovr', average='macro')
    return f1_score(y, yp, average='macro'), auc


rows = []
for t in TASKS:
    for model in ['A0', 'A2b', 'A6']:
        for s in SEEDS:
            for k in range(5):
                f = bench_file(t, s, k) if model == 'A0' else novel_file(t, model, s, k)
                if not f.exists(): continue
                f1, auc = score(pd.read_csv(f), collapse=(t == 'ours_binary' and model != 'A0'))
                rows.append(dict(task=t, model=model, seed=s, fold=k, f1_macro=100 * f1, roc_auc=100 * auc))
F = pd.DataFrame(rows); F.to_csv(OUT / 'seeds_per_fold.csv', index=False)
n = F.groupby(['task', 'model', 'seed']).size(); F = F.join(n.rename('n'), on=['task', 'model', 'seed']); F = F[F.n == 5]
S = F.groupby(['task', 'model', 'seed'])[['f1_macro', 'roc_auc']].mean().reset_index()
S.to_csv(OUT / 'seeds_summary.csv', index=False)
pd.set_option('display.width', 200)
print(S.pivot_table(index=['task', 'model'], columns='seed', values=['f1_macro', 'roc_auc']).round(1).reindex(TASKS, level=0))
