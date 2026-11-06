"""Summarise the novel-design runs: per task x row mean / sd over the 5 test folds, compared with a reference row
(default A0 = benchmark ConvNeXtTiny) by the Nadeau-Bengio corrected resampled t-test (Holm within task).
Wilcoxon on 5 folds cannot reach p < 0.05, so only the count of improved folds is reported for it.

  python analyze_novel.py                 # all tasks / rows found in results/
  python analyze_novel.py --ref A1        # compare against A1 instead of A0
Output: results/summary/summary_<metric>.csv and results/summary/comparison.csv
"""
import argparse, json
import numpy as np, pandas as pd
from scipy import stats
from common_novel import RES, RESULTS, N_FOLDS, TASKS, public_out_dir

ap = argparse.ArgumentParser()
ap.add_argument('--ref', default='A0')
args = ap.parse_args()
METRICS = ['accuracy', 'f1_macro', 'roc_auc', 'balanced_accuracy', 'sensitivity', 'specificity']


def bench_rows(task):
    """A0 = the benchmark ConvNeXtTiny run of the same task (same folds)."""
    base = {'ours_3class': RESULTS / 'trained_models', 'ours_binary': RESULTS / 'binary' / 'trained_models'}.get(task)
    base = base or public_out_dir(task) / 'trained_models'
    out = []
    for f in sorted((base / 'ConvNeXtTiny').glob('fold_*/done.json')):
        j = json.load(open(f)); m = dict(j['test_metrics'])
        m.setdefault('roc_auc', m.get('roc_auc_ovr_macro'))
        out.append(dict(task=task, row='A0', fold=j['fold'], **{k: m.get(k) for k in METRICS}))
    return out


rows = []
for task in TASKS:
    rows += bench_rows(task)
    for f in sorted((RES / task).glob('*/fold_*/done.json')):
        j = json.load(open(f)); m = j['test_metrics']; mt = j.get('test_metrics_thr', {})
        r = dict(task=task, row=j['row'], fold=j['fold'], **{k: m.get(k) for k in METRICS})
        if mt: r.update(sens_thr=mt['sensitivity'], spec_thr=mt['specificity'], f1_macro_thr=mt['f1_macro'])
        r.update(ece=j['ece_test'], brier=j['brier_test'], tta=j['tta'], T=j['temperature'])
        rows.append(r)
df = pd.DataFrame(rows)
out = RES / 'summary'; out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / 'fold_metrics.csv', index=False)
cols = [c for c in METRICS + ['sens_thr', 'spec_thr', 'f1_macro_thr', 'ece', 'brier'] if c in df]
g = df.groupby(['task', 'row'])
summ = g[cols].agg(['mean', 'std']).round(4); summ['n_folds'] = g.size()
summ.to_csv(out / 'summary.csv'); print(summ.to_string())

# corrected resampled t-test vs the reference row (train 60 %, test 20 % of the data per fold)
corr = 1 / N_FOLDS + 0.2 / 0.6
comp = []
for task, d in df.groupby('task'):
    ref = d[d.row == args.ref].set_index('fold')
    if len(ref) < N_FOLDS: continue
    for row, dr in d[d.row != args.ref].groupby('row'):
        dr = dr.set_index('fold')
        if len(dr) < N_FOLDS: continue
        for met in ['f1_macro', 'roc_auc', 'balanced_accuracy']:
            diff = (dr[met] - ref[met]).dropna().values
            if len(diff) < N_FOLDS: continue
            sd = diff.std(ddof=1); t = diff.mean() / np.sqrt(corr * sd ** 2) if sd > 0 else np.inf * np.sign(diff.mean())
            p = 2 * stats.t.sf(abs(t), len(diff) - 1)
            comp.append(dict(task=task, row=row, ref=args.ref, metric=met, diff_mean=diff.mean(), folds_improved=int((diff > 0).sum()),
                             t_nb=t, p_nb=p))
comp = pd.DataFrame(comp)
if len(comp):
    comp['p_holm'] = comp.groupby(['task', 'metric']).p_nb.transform(
        lambda p: np.minimum(1, np.maximum.accumulate(np.sort(p.values) * (len(p) - np.arange(len(p))))[np.argsort(np.argsort(p.values))]))
    comp.round(4).to_csv(out / 'comparison.csv', index=False); print(comp.round(4).to_string())
