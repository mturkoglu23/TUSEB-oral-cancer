"""Patient-level (cluster) bootstrap 95% confidence intervals for both experiments.

  python 07_bootstrap_ci.py

The out-of-fold test predictions of all N_FOLDS folds are pooled (each image is predicted exactly once, by the model that did not see its
patient). Patients - not images - are resampled with replacement, stratified by diagnosis (11 malignant / 35 benign / 46 normal patients
in every replicate), so all images of a patient stay together and every replicate contains cancer cases. Percentile intervals, B = 2000.
The same resampled patients are used for every model and rule within a replicate, which gives paired CIs for differences.

Two levels: image level (metrics over the images of the resampled patients) and patient level (one prediction per patient = mean
probability over its images). Outputs: results/bootstrap_ci/ (3-class) and results/binary/bootstrap_ci/ (binary).
"""
import numpy as np, pandas as pd
from scipy.stats import rankdata
from common import *

B = 2000
_to_csv = pd.DataFrame.to_csv
pd.DataFrame.to_csv = lambda self, *a, **k: _to_csv(self, *a, **{'encoding': 'utf-8-sig', **k})
O3 = RESULTS / 'bootstrap_ci'; OB = RESULTS / 'binary' / 'bootstrap_ci'
O3.mkdir(exist_ok=True); OB.mkdir(exist_ok=True)
MAL = CLASSES.index('malignant')


def load(root, name):
    return pd.concat([pd.read_csv(root / 'predictions' / name / f'fold_{k:02d}_test_predictions.csv') for k in range(N_FOLDS)]
                     ).sort_values('file').reset_index(drop=True)


def auc(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    return (rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def mcc_from_cm(cm):
    t = cm.sum(1); p = cm.sum(0); c = np.trace(cm); s = cm.sum()
    den = np.sqrt((s * s - (p * p).sum()) * (s * s - (t * t).sum()))
    return (c * s - (t * p).sum()) / den if den else 0.0


def m3(y, P):
    yp = P.argmax(1); K = 3
    cm = np.zeros((K, K)); np.add.at(cm, (y, yp), 1)
    tp = np.diag(cm); rec = tp / np.maximum(cm.sum(1), 1); pre = np.where(cm.sum(0) > 0, tp / np.maximum(cm.sum(0), 1), 0)
    f1 = np.where(pre + rec > 0, 2 * pre * rec / np.maximum(pre + rec, 1e-12), 0)
    return dict(accuracy=tp.sum() / cm.sum(), f1_macro=f1.mean(), balanced_accuracy=rec.mean(), mcc=mcc_from_cm(cm),
                roc_auc_ovr_macro=np.mean([auc((y == i).astype(int), P[:, i]) for i in range(K)]),
                malignant_recall=rec[MAL], malignant_f1=f1[MAL])


def m2(y, s, yp, lab):
    tp = ((y == 1) & (yp == 1)).sum(); fn = ((y == 1) & (yp == 0)).sum(); fp = ((y == 0) & (yp == 1)).sum(); tn = ((y == 0) & (yp == 0)).sum()
    se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1); pr = tp / (tp + fp) if tp + fp else 0.0
    return dict(roc_auc=auc(y, s), sensitivity=se, specificity=sp, balanced_accuracy=(se + sp) / 2,
                f1=2 * pr * se / (pr + se) if pr + se else 0.0, mcc=mcc_from_cm(np.array([[tn, fp], [fn, tp]], float)),
                benign_fpr=yp[lab == 'benign'].mean(), normal_fpr=yp[lab == 'normal'].mean())


# ---------------- data (same image order for every model / experiment)
ref = load(RESULTS, list(MODELS)[0])
files, pid, lab = ref.file.values, ref.patient_id.values, ref.label.values
y3 = ref.y.values; yb = (lab == 'malignant').astype(int)
pats = pd.Series(lab).groupby(pid).first()  # patient -> diagnosis
pat_ids = pats.index.values; pat_lab = pats.values; pat_y3 = np.array([CLASSES.index(l) for l in pat_lab]); pat_yb = (pat_lab == 'malignant').astype(int)
img_of = {p: np.where(pid == p)[0] for p in pat_ids}
P3, PB = {}, {}
for m in MODELS:
    a = load(RESULTS, m); b = load(RESULTS / 'binary', m)
    assert (a.file.values == files).all() and (b.file.values == files).all()
    P3[m] = a[[f'p_{c}' for c in CLASSES]].values; PB[m] = b.p_cancer.values
# patient-level probabilities (mean over a patient's images)
Q3 = {m: np.stack([P3[m][img_of[p]].mean(0) for p in pat_ids]) for m in MODELS}
QB = {m: np.array([PB[m][img_of[p]].mean() for p in pat_ids]) for m in MODELS}


def evaluate(img_idx, pat_idx):
    """All metrics for all models / rules on one (re)sample. Returns {(experiment, rule, level, model): metrics}."""
    out = {}
    yi, li, yib = y3[img_idx], lab[img_idx], yb[img_idx]
    yp_, lp_, ypb = pat_y3[pat_idx], pat_lab[pat_idx], pat_yb[pat_idx]
    for m in MODELS:
        Pi = P3[m][img_idx]; Qi = Q3[m][pat_idx]
        out[('3class', '3class', 'image', m)] = m3(yi, Pi)
        out[('3class', '3class', 'patient', m)] = m3(yp_, Qi)
        s = PB[m][img_idx]; out[('binary', 'binary_thr0.5', 'image', m)] = m2(yib, s, (s >= .5).astype(int), li)
        s = QB[m][pat_idx]; out[('binary', 'binary_thr0.5', 'patient', m)] = m2(ypb, s, (s >= .5).astype(int), lp_)
        out[('binary', 'from_3class', 'image', m)] = m2(yib, Pi[:, MAL], (Pi.argmax(1) == MAL).astype(int), li)
        out[('binary', 'from_3class', 'patient', m)] = m2(ypb, Qi[:, MAL], (Qi.argmax(1) == MAL).astype(int), lp_)
    return out


point = evaluate(np.arange(len(files)), np.arange(len(pat_ids)))
rng = np.random.default_rng(SEED)
strata = [np.where(pat_lab == c)[0] for c in CLASSES]
reps = []
for b in range(B):
    pi = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
    ii = np.concatenate([img_of[pat_ids[j]] for j in pi])
    reps.append(evaluate(ii, pi))
    if (b + 1) % 500 == 0: print('bootstrap', b + 1, '/', B, flush=True)


def ci_rows(keys):
    rows = []
    for key in keys:
        for met, v in point[key].items():
            bs = np.array([r[key][met] for r in reps], float)
            rows.append(dict(experiment=key[0], rule=key[1], level=key[2], model=key[3], metric=met, point=v,
                             ci_low=np.nanpercentile(bs, 2.5), ci_high=np.nanpercentile(bs, 97.5), boot_sd=np.nanstd(bs, ddof=1)))
    return pd.DataFrame(rows)


def diff_rows(pairs, metrics, label):
    """Paired bootstrap CI of a - b (same resampled patients). p = two-sided bootstrap p (2 x share of replicates on the other side of 0)."""
    rows = []
    for ka, kb in pairs:
        for met in metrics:
            d = np.array([r[ka][met] - r[kb][met] for r in reps], float); d = d[np.isfinite(d)]
            rows.append(dict(comparison=label(ka, kb), level=ka[2], metric=met, diff=point[ka][met] - point[kb][met],
                             ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
                             p_boot=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
    return pd.DataFrame(rows)


def fmt(r, pct=True):
    f = 100 if pct else 1; d = 1 if pct else 2
    return f"{r.point * f:.{d}f} [{r.ci_low * f:.{d}f}, {r.ci_high * f:.{d}f}]"


def pretty(ci, rule, level, metrics, names, order):
    t = ci[(ci.rule == rule) & (ci.level == level) & ci.metric.isin(metrics)].copy()
    t['v'] = [fmt(r, r.metric != 'mcc') for r in t.itertuples()]
    t = t.pivot(index='model', columns='metric', values='v').loc[order, metrics]; t.columns = names
    return t


# ---------------- 3-class
k3 = [k for k in point if k[0] == '3class']
ci3 = ci_rows(k3); ci3.to_csv(O3 / 'ci_3class.csv', index=False)
order3 = ci3[(ci3.level == 'image') & (ci3.metric == 'f1_macro')].sort_values('point', ascending=False).model.tolist()
met3 = ['accuracy', 'f1_macro', 'roc_auc_ovr_macro', 'mcc', 'malignant_recall', 'malignant_f1']
nm3 = ['Accuracy (%)', 'Macro F1 (%)', 'AUC OvR macro (%)', 'MCC', 'Malignant recall (%)', 'Malignant F1 (%)']
md3 = []
for level in ('image', 'patient'):
    t = pretty(ci3, '3class', level, met3, nm3, order3); t.to_csv(O3 / f'ci_3class_{level}_level.csv')
    md3.append(f'### {level.capitalize()} level\n\n' + t.to_markdown())
best3 = {lv: ci3[(ci3.level == lv) & (ci3.metric == 'f1_macro')].sort_values('point').model.iloc[-1] for lv in ('image', 'patient')}
d3 = pd.concat([diff_rows([(('3class', '3class', lv, best3[lv]), ('3class', '3class', lv, m)) for m in MODELS if m != best3[lv]],
                          ['f1_macro', 'roc_auc_ovr_macro', 'malignant_recall'], lambda a, b: f'{a[3]} - {b[3]}') for lv in ('image', 'patient')])
d3.to_csv(O3 / 'diff_vs_best_3class.csv', index=False)

# ---------------- binary
kb = [k for k in point if k[0] == 'binary']
cib = ci_rows(kb); cib.to_csv(OB / 'ci_binary.csv', index=False)
orderb = cib[(cib.rule == 'binary_thr0.5') & (cib.level == 'image') & (cib.metric == 'roc_auc')].sort_values('point', ascending=False).model.tolist()
metb = ['roc_auc', 'sensitivity', 'specificity', 'balanced_accuracy', 'mcc', 'benign_fpr', 'normal_fpr']
nmb = ['ROC-AUC (%)', 'Sensitivity (%)', 'Specificity (%)', 'Balanced acc. (%)', 'MCC', 'Benign → cancer (%)', 'Normal → cancer (%)']
mdb = []
for rule in ('binary_thr0.5', 'from_3class'):
    for level in ('image', 'patient'):
        t = pretty(cib, rule, level, metb, nmb, orderb); t.to_csv(OB / f'ci_{rule}_{level}_level.csv')
        mdb.append(f'### {rule}, {level} level\n\n' + t.to_markdown())
dbin = diff_rows([(('binary', 'binary_thr0.5', lv, m), ('binary', 'from_3class', lv, m)) for lv in ('image', 'patient') for m in MODELS],
                 ['roc_auc', 'sensitivity', 'specificity', 'balanced_accuracy', 'normal_fpr', 'benign_fpr'], lambda a, b: f'{a[3]}: binary - 3class')
dbin.to_csv(OB / 'diff_binary_minus_3class.csv', index=False)
bestb = {lv: cib[(cib.rule == 'binary_thr0.5') & (cib.level == lv) & (cib.metric == 'roc_auc')].sort_values('point').model.iloc[-1] for lv in ('image', 'patient')}
dvb = pd.concat([diff_rows([(('binary', 'binary_thr0.5', lv, bestb[lv]), ('binary', 'binary_thr0.5', lv, m)) for m in MODELS if m != bestb[lv]],
                           ['roc_auc', 'sensitivity'], lambda a, b: f'{a[3]} - {b[3]}') for lv in ('image', 'patient')])
dvb.to_csv(OB / 'diff_vs_best_binary.csv', index=False)


def diff_md(d, metric):
    t = d[d.metric == metric].copy()
    t['diff (pp) [95% CI]'] = [f'{r.diff * 100:+.1f} [{r.ci_low * 100:+.1f}, {r.ci_high * 100:+.1f}]' for r in t.itertuples()]
    t['CI excludes 0'] = np.where((t.ci_low > 0) | (t.ci_high < 0), 'yes', 'no')
    return t[['comparison', 'level', 'diff (pp) [95% CI]', 'p_boot', 'CI excludes 0']].round(3).to_markdown(index=False)


head = (f'Patient-level cluster bootstrap, B = {B}, stratified by diagnosis (11 malignant / 35 benign / 46 normal patients per replicate), '
        'percentile 95% CI, pooled out-of-fold test predictions. Format: point estimate [2.5%, 97.5%]. '
        'Differences are paired (same resampled patients); p_boot is not corrected for multiple comparisons.\n\n')
(O3 / 'BOOTSTRAP_CI_3class.md').write_text(head + '\n\n'.join(md3) + f'\n\n### Paired differences vs best model (macro F1; image level: {best3["image"]}, '
                                           f'patient level: {best3["patient"]})\n\nMacro F1:\n\n' + diff_md(d3, 'f1_macro') +
                                           '\n\nROC-AUC:\n\n' + diff_md(d3, 'roc_auc_ovr_macro') + '\n', encoding='utf-8')
(OB / 'BOOTSTRAP_CI_binary.md').write_text(head + '\n\n'.join(mdb) + '\n\n### Binary-trained minus 3-class-collapsed (paired)\n\nROC-AUC:\n\n' +
                                           diff_md(dbin, 'roc_auc') + '\n\nNormal flagged as cancer:\n\n' + diff_md(dbin, 'normal_fpr') +
                                           '\n\nSpecificity:\n\n' + diff_md(dbin, 'specificity') + '\n\nSensitivity:\n\n' + diff_md(dbin, 'sensitivity') +
                                           f'\n\n### Paired differences vs best binary model (ROC-AUC; image: {bestb["image"]}, patient: {bestb["patient"]})\n\n' +
                                           diff_md(dvb, 'roc_auc') + '\n', encoding='utf-8')
print('done')
