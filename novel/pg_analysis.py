"""Patient-grouped NDO (Section 3.7 / Table 8): macro F1 / ROC-AUC per fold from the saved test predictions, paired
corrected t-tests of A6 against A0 and A2b with Holm adjustment over the three pg tasks, factorial rows on NDO-4 pg,
and seed means (mean +- SD over seeds 42-44) from seeds_summary.csv.
  python pg_analysis.py
"""
import numpy as np, pandas as pd
from scipy import stats
from seed_analysis import NOV, PUB, OUT, score

CORR = 1 / 5 + 0.2 / 0.6
PG = ['NDO_4class_pg', 'NDO_3class_pg', 'NDO_2class_pg']


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p)
    adj[o] = np.minimum(1, np.maximum.accumulate(p[o] * (len(p) - np.arange(len(p)))))
    return adj


def nb(a, b):
    d = np.asarray(a) - np.asarray(b); t = d.mean() / np.sqrt(CORR * d.var(ddof=1))
    return d.mean(), int((d > 0).sum()), 2 * stats.t.sf(abs(t), len(d) - 1)


def fold_scores(task, row):
    f = [PUB / task / 'predictions' / 'ConvNeXtTiny' / f'fold_{k:02d}_test_predictions.csv' if row == 'A0' else
         NOV / 'results' / task / row / f'fold_{k:02d}' / 'test_predictions.csv' for k in range(5)]
    s = np.array([score(pd.read_csv(x)) for x in f]) * 100
    return s[:, 0], s[:, 1]


rows, tests = [], []
for t in PG:
    rws = ['A0', 'A2b', 'A4c', 'A4s', 'A2r', 'A5', 'A6'] if t == 'NDO_4class_pg' else ['A0', 'A2b', 'A6']
    F = {r: fold_scores(t, r) for r in rws}
    for r in rws:
        rows.append(dict(task=t, row=r, f1=F[r][0].mean(), f1_sd=F[r][0].std(ddof=1), auc=F[r][1].mean()))
    for c in ['A0', 'A2b'] + (['A5', 'A2r'] if t == 'NDO_4class_pg' else []):
        m, w, p = nb(F['A6'][0], F[c][0]); tests.append(dict(task=t, vs=c, diff=m, better=w, p=p))
R = pd.DataFrame(rows); T = pd.DataFrame(tests)
for c in ['A0', 'A2b']:
    i = T.vs == c; T.loc[i, 'p_holm'] = holm(T.loc[i, 'p'])
R.to_csv(OUT / 'pg_ndo.csv', index=False); T.to_csv(OUT / 'pg_ndo_tests.csv', index=False)
pd.set_option('display.width', 200)
print(R.round(1).to_string(index=False)); print(T.round(3).to_string(index=False))

S = pd.read_csv(OUT / 'seeds_summary.csv')
M = S.groupby(['task', 'model'])[['f1_macro', 'roc_auc']].agg(['mean', 'std']).round(1)
print(M.to_string())
W = S.pivot_table(index=['task', 'seed'], columns='model', values='f1_macro')
print('A6 > A2b seeds:', (W.A6 > W.A2b).groupby('task').sum().to_dict())
print('A6 > A0 seeds:', (W.A6 > W.A0).groupby('task').sum().to_dict())
print('A2b > A0 seeds:', (W.A2b > W.A0).groupby('task').sum().to_dict())
