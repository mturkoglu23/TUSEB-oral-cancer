"""Analyses requested in the first internal review (2026-11-03), computed from the existing fold results and
prediction files only (no training). Output: results/revision/*.csv and a printed summary.

  1. paired corrected t-tests of A6 against the strong controls A2b, A1 and A2c (Holm over the 7 tasks per comparison)
  2. our data, cancer vs non-cancer from every 3-class model: argmax rule and P(malignant) > 0.5 rule
  3. our data, patient-level analysis (mean probability per patient, argmax) and per-malignant-patient detections
  4. patient-cluster bootstrap CI of image-level sensitivity / specificity (collapsed A6)
  5. class-wise recall A2b / A5 / A6 on every multi-class task (does cRW raise minority recall?)
  6. calibration (ECE, Brier) of A6; within-data-set exact duplicates (md5); NDO patients spanning folds
  7. inference cost (GFLOPs) of ConvNeXt-Tiny at 224 and 384 px
  python revision_analyses.py
"""
import json
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, recall_score
from common_novel import RES, RESULTS, N_FOLDS, TASKS, public_out_dir
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix

CORR = 1 / N_FOLDS + 0.2 / 0.6  # Nadeau-Bengio correction, as in report_novel.py


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p)
    adj[o] = np.minimum(1, np.maximum.accumulate(p[o] * (len(p) - np.arange(len(p)))))
    return adj


def nb_test(a, b):
    d = np.asarray(a) - np.asarray(b); sd = d.std(ddof=1)
    t = d.mean() / np.sqrt(CORR * sd ** 2) if sd > 0 else np.inf * np.sign(d.mean())
    return d.mean(), int((d > 0).sum()), 2 * stats.t.sf(abs(t), len(d) - 1)


def bin_metrics(y, p, yp):
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, yp), f1_macro=f1_score(y, yp, average='macro'), roc_auc=roc_auc_score(y, p),
                balanced_accuracy=balanced_accuracy_score(y, yp), mcc=matthews_corrcoef(y, yp),
                sensitivity=tp / (tp + fn), specificity=tn / (tn + fp))

OUT = RES / 'revision'; OUT.mkdir(parents=True, exist_ok=True)
ld = lambda p: json.load(open(p, encoding='utf-8'))
rng = np.random.default_rng(0)


def fold_metric(task, row):
    """per-fold test metrics of a novel row (ours_binary for 3-class rows = collapsed, see collapsed())"""
    out = []
    for k in range(N_FOLDS):
        j = ld(RES / task / row / f'fold_{k:02d}' / 'done.json'); t = j['test_metrics']
        out.append(dict(fold=k, f1_macro=t['f1_macro'], roc_auc=t.get('roc_auc', t.get('roc_auc_ovr_macro'))))
    return pd.DataFrame(out)


def preds(row, k, split='test'):
    if row == 'A0':
        d = pd.read_csv(RESULTS / 'predictions' / 'ConvNeXtTiny' / f'fold_{k:02d}_{split}_predictions.csv')
        return d.rename(columns={'patient_id': 'group'})
    return pd.read_csv(RES / 'ours_3class' / row / f'fold_{k:02d}' / f'{split}_predictions.csv')


def collapsed(row, rule='argmax'):
    out = []
    for k in range(N_FOLDS):
        d = preds(row, k); y = (d.label == 'malignant').astype(int).values; p = d.p_malignant.values
        yp = (d[['p_benign', 'p_malignant', 'p_normal']].values.argmax(1) == 1) if rule == 'argmax' else (p > 0.5)
        out.append(dict(fold=k, **bin_metrics(y, p, yp.astype(int))))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 1. A6 vs strong controls
rows = []
for ctl in ['A2b', 'A1', 'A2c', 'A5']:
    recs = []
    for t in TASKS:
        if t == 'ours_binary':
            a, b = collapsed('A6'), collapsed(ctl)
        else:
            a, b = fold_metric(t, 'A6'), fold_metric(t, ctl)
        for m in ['f1_macro', 'roc_auc']:
            d, nb, p = nb_test(a[m].values, b[m].values)
            recs.append(dict(control=ctl, task=t, metric=m, a6=100 * a[m].mean(), ctl=100 * b[m].mean(), diff=100 * d, folds_better=nb, p=p))
    r = pd.DataFrame(recs)
    for m in ['f1_macro', 'roc_auc']:
        s = r.metric == m; r.loc[s, 'p_holm'] = holm(r.loc[s, 'p'].values)
    rows.append(r)
S = pd.concat(rows); S.to_csv(OUT / 'a6_vs_controls.csv', index=False)
print('\n== 1. A6 vs controls (points; folds better; p / p_Holm)')
for (c, m), g in S.groupby(['control', 'metric'], sort=False):
    print(c, m, ' | '.join(f"{r.task}: {r.diff:+.1f} ({r.folds_better}/5, {r.p:.3f}/{r.p_holm:.3f})" for r in g.itertuples()))

# ------------------------------------------------------------------ 2. collapsed binary for every 3-class model
recs = []
for row in ['A0', 'A1', 'A2a', 'A2b', 'A2c', 'A4c', 'A5', 'A6']:
    for rule in ['argmax', 'p>0.5']:
        c = collapsed(row, rule)
        recs.append(dict(row=row, rule=rule, **{m: 100 * c[m].mean() for m in ['f1_macro', 'roc_auc', 'sensitivity', 'specificity', 'balanced_accuracy']}))
for row in ['A1', 'A2b', 'A5', 'A6']:  # binary-trained counterparts (same recipe)
    t = [ld(RES / 'ours_binary' / row / f'fold_{k:02d}' / 'done.json')['test_metrics'] for k in range(N_FOLDS)]
    recs.append(dict(row=row, rule='binary-trained', **{m: 100 * np.mean([x[m] for x in t]) for m in ['f1_macro', 'roc_auc', 'sensitivity', 'specificity', 'balanced_accuracy']}))
B = pd.DataFrame(recs); B.to_csv(OUT / 'ours_binary_collapsed.csv', index=False)
print('\n== 2. our data, cancer vs non-cancer (fold means, %)'); print(B.round(1).to_string(index=False))

# ------------------------------------------------------------------ 3./4. patient level and bootstrap
pool = {r: pd.concat([preds(r, k).assign(fold=k) for k in range(N_FOLDS)]) for r in ['A0', 'A1', 'A2b', 'A6']}
recs, per = [], []
for r, d in pool.items():
    d = d.assign(mal=(d.label == 'malignant').astype(int))
    g = d.groupby('group').agg(mal=('mal', 'max'), pb=('p_benign', 'mean'), pm=('p_malignant', 'mean'), pn=('p_normal', 'mean'),
                               n=('mal', 'size'), hit=('y_pred', lambda s: int((s == 1).sum())))
    pred = g[['pb', 'pm', 'pn']].values.argmax(1) == 1
    m = bin_metrics(g.mal.values, g.pm.values, pred.astype(int))
    recs.append(dict(row=r, level='patient (mean prob, argmax)', n_pos=int(g.mal.sum()), n_neg=int((1 - g.mal).sum()),
                     tp=int((pred & (g.mal == 1)).sum()), fp=int((pred & (g.mal == 0)).sum()), **{k: 100 * v for k, v in m.items()}))
    mg = g[g.mal == 1]
    per.append(pd.DataFrame(dict(row=r, patient=mg.index, images=mg.n, images_called_malignant=mg.hit, mean_p_malignant=mg.pm.round(3),
                                 detected_patient_level=pred[g.mal.values == 1])))
P = pd.DataFrame(recs); P.to_csv(OUT / 'ours_patient_level.csv', index=False)
PM = pd.concat(per); PM.to_csv(OUT / 'ours_malignant_patients.csv', index=False)
print('\n== 3. patient level (%)'); print(P.round(1).to_string(index=False))
print(PM.pivot(index='patient', columns='row', values='images_called_malignant').join(PM[PM.row == 'A6'].set_index('patient')[['images']]).to_string())

d = pool['A6']; d = d.assign(mal=(d.label == 'malignant').astype(int), call=(d.y_pred == 1).astype(int))
pats = d.group.unique(); idx = {p: np.where(d.group.values == p)[0] for p in pats}
bs = []
for _ in range(2000):
    s = np.concatenate([idx[p] for p in rng.choice(pats, len(pats))]); y, c = d.mal.values[s], d.call.values[s]
    bs.append((c[y == 1].mean(), 1 - c[y == 0].mean()))
bs = np.array(bs) * 100; y, c = d.mal.values, d.call.values
ci = dict(sens=100 * c[y == 1].mean(), sens_lo=np.percentile(bs[:, 0], 2.5), sens_hi=np.percentile(bs[:, 0], 97.5),
          spec=100 * (1 - c[y == 0].mean()), spec_lo=np.percentile(bs[:, 1], 2.5), spec_hi=np.percentile(bs[:, 1], 97.5))
pd.DataFrame([ci]).to_csv(OUT / 'ours_A6_sens_spec_ci.csv', index=False)
print('\n== 4. A6 collapsed, pooled image-level sens/spec with patient-cluster 95 % CI:', {k: round(v, 1) for k, v in ci.items()})

# ------------------------------------------------------------------ 5. class-wise recall A2b / A5 / A6
recs = []
for t in ['ours_3class', 'NDO_3class', 'NDO_4class', 'NDO_2class']:
    for row in ['A2b', 'A5', 'A6']:
        for k in range(N_FOLDS):
            d = pd.read_csv(RES / t / row / f'fold_{k:02d}' / 'test_predictions.csv')
            cls = [c[2:] for c in d.columns if c.startswith('p_')]
            r = recall_score(d.y, d.y_pred, labels=range(len(cls)), average=None, zero_division=0)
            recs += [dict(task=t, row=row, fold=k, cls=cl, recall=100 * v) for cl, v in zip(cls, r)]
R = pd.DataFrame(recs); R.to_csv(OUT / 'classwise_recall_A2b_A5_A6.csv', index=False)
print('\n== 5. class-wise recall (mean over folds, %)')
print(R.groupby(['task', 'cls', 'row']).recall.mean().unstack('row').round(1).to_string())

# ------------------------------------------------------------------ 6. calibration, duplicates, NDO patient overlap
cal = []
for t in TASKS:
    if t == 'ours_binary': continue
    j = [ld(RES / t / 'A6' / f'fold_{k:02d}' / 'done.json') for k in range(N_FOLDS)]
    cal.append(dict(task=t, ece=np.mean([x['ece_test'] for x in j]), brier=np.mean([x['brier_test'] for x in j]),
                    tta_folds=sum(x['tta'] for x in j)))
C = pd.DataFrame(cal); C.to_csv(OUT / 'a6_calibration.csv', index=False); print('\n== 6. A6 calibration'); print(C.round(3).to_string(index=False))
for t in ['OCD3_2class', 'OCD2_2class', 'NDO_4class']:
    s = pd.read_csv(public_out_dir(t) / 'fold_splits' / 'splits.csv', encoding='utf-8-sig')
    dup = s[s.md5.duplicated(keep=False)]
    print(f'{t}: {len(s)} images, {dup.md5.nunique()} md5 values occur more than once ({len(dup)} images); '
          f'cross-label duplicates: {(dup.groupby("md5").label.nunique() > 1).sum()}')
    if 'patient_id' in s and s.patient_id.notna().all():
        g = s.groupby('patient_id').fold.nunique()
        print(f'   patients {g.size}, in >1 fold {int((g > 1).sum())} ({(g > 1).mean():.0%}), images of those {int(s.patient_id.isin(g[g > 1].index).sum())}')

# ------------------------------------------------------------------ 7. inference cost
import torch, timm
from torch.utils.flop_counter import FlopCounterMode
m = timm.create_model('convnext_tiny', pretrained=False, num_classes=3).eval()
for px in (224, 384):
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(torch.zeros(1, 3, px, px))
    print(f'ConvNeXt-Tiny {px} px: {fc.get_total_flops() / 2e9:.2f} GMACs')
