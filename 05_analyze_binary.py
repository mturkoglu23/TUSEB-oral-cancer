"""Binary experiment: cancer (malignant) vs non-cancer (benign + normal) on the same patient-grouped folds.

  python 05_analyze_binary.py [smoke]

Everything is recomputed from the saved predictions in results/binary/predictions. Three decision rules are reported:
  * binary model, threshold 0.5 (primary, pre-specified)
  * binary model, threshold chosen on that fold's VALIDATION predictions (Youden J) - test fold never used
  * the 3-class models from step 2 collapsed to binary (score = p_malignant, decision = argmax == malignant)
The non-cancer class is broken down into its benign and normal subgroups (false-positive rate of each).
"""
import json, itertools, warnings, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from common import *
from metrics import binary_metrics

warnings.filterwarnings('ignore', category=RuntimeWarning)
_to_csv = pd.DataFrame.to_csv  # utf-8 with BOM so Excel shows ± correctly
pd.DataFrame.to_csv = lambda self, *a, **k: _to_csv(self, *a, **{'encoding': 'utf-8-sig', **k})
R = RESULTS / 'binary' / ('smoke' if 'smoke' in sys.argv[1:] else '')
for s in ['fold_metrics', 'confusion_matrices', 'roc_curves', 'comparison_plots', 'statistical_tests', 'tables']:
    (R / s).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight', 'figure.dpi': 100})
GCOL = {'benign': '#009E73', 'normal': '#0072B2', 'cancer': '#D55E00'}
NICE = dict(accuracy='Accuracy', sensitivity='Sensitivity (recall, cancer)', specificity='Specificity', precision='Precision (PPV)',
            npv='NPV', f1='F1-score (cancer)', f1_macro='F1-score (macro)', balanced_accuracy='Balanced accuracy',
            roc_auc='ROC-AUC', pr_auc='PR-AUC (average precision)', cohen_kappa="Cohen's kappa", mcc='MCC',
            benign_fpr='Benign flagged as cancer', normal_fpr='Normal flagged as cancer', benign_vs_malignant_auc='AUC benign vs malignant')
PCT = [c for c in NICE if c not in ('cohen_kappa', 'mcc')]
RULES = {'binary_thr0.5': 'Binary model, threshold 0.5', 'binary_valthr': 'Binary model, validation-selected threshold',
         'from_3class': '3-class model collapsed to binary'}


def save(fig, path):
    fig.savefig(str(path) + '.png', dpi=400); fig.savefig(str(path) + '.pdf'); plt.close(fig)


def youden_threshold(y, p):
    if not 0 < y.sum() < len(y): return 0.5
    fpr, tpr, thr = roc_curve(y, p); j = tpr - fpr
    return float(min(max(thr[np.argmax(j)], 1e-6), 1.0))


def msd(x, pct):
    x = np.asarray(x, float); f = 100 if pct else 1; n = np.isfinite(x).sum()
    return f'{np.nanmean(x) * f:.2f} ± {np.nanstd(x, ddof=1) * f:.2f}' + ('' if n == len(x) else f' (n={n})')


# ---------------- collect
rows, comp, preds = [], [], {}
for name in MODELS:
    for k in range(N_FOLDS):
        d = R / 'trained_models' / name / f'fold_{k:02d}'
        st = 'complete' if (d / 'done.json').exists() else 'failed' if (d / 'error.txt').exists() else 'not run'
        comp.append(dict(model=name, fold=k, status=st))
        if st != 'complete': continue
        info = json.load(open(d / 'done.json'))
        te = pd.read_csv(R / 'predictions' / name / f'fold_{k:02d}_test_predictions.csv')
        va = pd.read_csv(R / 'predictions' / name / f'fold_{k:02d}_val_predictions.csv')
        thr = youden_threshold(va.y.values, va.p_cancer.values)
        common_ = dict(model=name, fold=k, best_phase=info['best_phase'], best_epoch=info['best_global_epoch'],
                       epochs_run=info['epochs_run'], train_time_s=info['train_time_s'], latency_ms=info['latency_ms_bs1_median'],
                       params_M=info['params_total'] / 1e6)
        for rule, t in (('binary_thr0.5', 0.5), ('binary_valthr', thr)):
            met = binary_metrics(te.y.values, te.p_cancer.values, te.label.values, thr=t)
            rows.append(dict(rule=rule, **common_, **{kk: v for kk, v in met.items() if kk != 'confusion_matrix'}))
        f3 = RESULTS / 'predictions' / name / f'fold_{k:02d}_test_predictions.csv'
        if f3.exists():  # same images, same fold
            t3 = pd.read_csv(f3); assert list(t3.file) == list(te.file)
            met = binary_metrics((t3.label == 'malignant').astype(int).values, t3.p_malignant.values, t3.label.values,
                                 yp=(t3.y_pred == CLASSES.index('malignant')).values)
            rows.append(dict(rule='from_3class', **common_, **{kk: v for kk, v in met.items() if kk != 'confusion_matrix'}))
            te['p3_malignant'] = t3.p_malignant.values; te['y3_pred_cancer'] = (t3.y_pred == CLASSES.index('malignant')).astype(int).values
        preds[(name, k)] = te.assign(fold=k, val_threshold=thr)
comp = pd.DataFrame(comp)
cp = comp.pivot(index='model', columns='fold', values='status').reindex(list(MODELS))
cp['n_complete'] = (cp == 'complete').sum(1); cp.to_csv(R / 'tables' / 'completeness.csv')
fm = pd.DataFrame(rows)
if fm.empty: raise SystemExit('no completed folds yet')
fm.to_csv(R / 'fold_metrics' / 'all_models_fold_metrics.csv', index=False)
models = [m for m in MODELS if m in set(fm.model)]
full = [m for m in models if cp.loc[m, 'n_complete'] == N_FOLDS]
B = fm[fm.rule == 'binary_thr0.5']

# ---------------- summary tables
summ, num = [], []
for (rule, m), g in fm.groupby(['rule', 'model']):
    summ.append(dict(Rule=rule, Model=m, folds=len(g), **{NICE[c] + (' (%)' if c in PCT else ''): msd(g[c], c in PCT) for c in NICE}))
    num.append(dict(rule=rule, model=m, folds=len(g), **{f'{c}_mean': g[c].mean() for c in NICE}, **{f'{c}_sd': g[c].std(ddof=1) for c in NICE},
                    params_M=g.params_M.iloc[0], latency_ms=g.latency_ms.median(), train_time_min=g.train_time_s.mean() / 60,
                    threshold_mean=g.threshold.mean()))
summ = pd.DataFrame(summ); num = pd.DataFrame(num)
nb = num[num.rule == 'binary_thr0.5'].set_index('model')
order = nb.sort_values('roc_auc_mean', ascending=False).index.tolist()  # primary ranking metric: ROC-AUC (threshold-free)
summ['_o'] = summ.Model.map({m: i for i, m in enumerate(order)}); summ = summ.sort_values(['Rule', '_o']).drop(columns='_o')
summ.to_csv(R / 'tables' / 'summary_mean_sd_all_metrics.csv', index=False)
num.to_csv(R / 'tables' / 'summary_numeric.csv', index=False)
main_cols = ['folds', 'Accuracy (%)', 'Sensitivity (recall, cancer) (%)', 'Specificity (%)', 'Precision (PPV) (%)', 'F1-score (cancer) (%)',
             'Balanced accuracy (%)', 'ROC-AUC (%)', 'PR-AUC (average precision) (%)', 'MCC']
short = ['Folds', 'Accuracy (%)', 'Sensitivity (%)', 'Specificity (%)', 'Precision (%)', 'F1 cancer (%)', 'Balanced acc. (%)',
         'ROC-AUC (%)', 'PR-AUC (%)', 'MCC']
for rule in RULES:
    t = summ[summ.Rule == rule].set_index('Model')
    if t.empty: continue
    t = t.loc[[m for m in order if m in t.index], main_cols]; t.columns = short
    t.to_csv(R / 'tables' / f'main_table_{rule}.csv')
    (R / 'tables' / f'main_table_{rule}.md').write_text(t.to_markdown(), encoding='utf-8')
    (R / 'tables' / f'main_table_{rule}.tex').write_text(t.to_latex(escape=True), encoding='utf-8')
sub_cols = ['Specificity (%)', 'Benign flagged as cancer (%)', 'Normal flagged as cancer (%)', 'AUC benign vs malignant (%)']
sg = summ[summ.Rule == 'binary_thr0.5'].set_index('Model').loc[order, sub_cols]
sg.to_csv(R / 'tables' / 'subgroup_false_positive_rates.csv'); (R / 'tables' / 'subgroup_false_positive_rates.md').write_text(sg.to_markdown(), encoding='utf-8')

# pooled out-of-fold and patient level (each patient appears in exactly one test fold)
oof, pat = [], []
for m in order:
    P = pd.concat([preds[(m, k)] for k in range(N_FOLDS) if (m, k) in preds])
    o = binary_metrics(P.y.values, P.p_cancer.values, P.label.values); o.pop('confusion_matrix'); oof.append(dict(model=m, n=len(P), **o))
    G = P.groupby('patient_id').agg(y=('y', 'first'), label=('label', 'first'), p_cancer=('p_cancer', 'mean'))
    q = binary_metrics(G.y.values, G.p_cancer.values, G.label.values); q.pop('confusion_matrix')
    pat.append(dict(model=m, n_patients=len(G), cancer_patients_detected=f"{q['tp']}/{q['tp'] + q['fn']}",
                    benign_patients_flagged=f"{int(round(q['benign_fpr'] * q['benign_n']))}/{q['benign_n']}",
                    normal_patients_flagged=f"{int(round(q['normal_fpr'] * q['normal_n']))}/{q['normal_n']}", **q))
pd.DataFrame(oof).set_index('model').to_csv(R / 'tables' / 'pooled_oof_metrics.csv')
pd.DataFrame(pat).set_index('model').to_csv(R / 'tables' / 'patient_level_oof_metrics.csv')

# ---------------- statistics
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(run, 1)
    return adj
def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0: return 0.0
    r = stats.rankdata(np.abs(d)); return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())
def corrected_t(d, n_tr, n_te):
    """Nadeau & Bengio (2003) corrected resampled t-test for CV score differences."""
    k = len(d); v = np.var(d, ddof=1)
    if v == 0: return np.nan
    t = d.mean() / np.sqrt((1 / k + n_te / n_tr) * v); return float(2 * stats.t.sf(abs(t), k - 1))
n_te = B.groupby('fold').n_test.first().mean(); n_tr = n_te * (N_FOLDS - 2)
st_rows, pw_rows = [], []
if len(full) >= 3:
    for met in ['roc_auc', 'mcc', 'balanced_accuracy', 'f1', 'sensitivity', 'pr_auc']:
        M = B[B.model.isin(full)].pivot(index='fold', columns='model', values=met)[full].dropna()
        chi, p = stats.friedmanchisquare(*[M[c] for c in full]); W = chi / (len(M) * (len(full) - 1))
        ranks = M.rank(axis=1, ascending=False).mean().sort_values(); best = ranks.index[0]
        st_rows.append(dict(metric=met, n_folds=len(M), n_models=len(full), friedman_chi2=chi, p_value=p, kendalls_W=W,
                            mean_ranks='; '.join(f'{a}={b:.2f}' for a, b in ranks.items())))
        res = []
        for a, b in itertools.combinations(full, 2):
            d = (M[a] - M[b]).values
            res.append(dict(metric=met, model_a=a, model_b=b, mean_diff=d.mean(), median_diff=np.median(d),
                            wilcoxon_p=stats.wilcoxon(M[a], M[b], zero_method='zsplit').pvalue if np.any(d != 0) else 1.0,
                            rank_biserial=rank_biserial(d), corrected_t_p=corrected_t(d, n_tr, n_te), vs_best=best in (a, b)))
        res = pd.DataFrame(res)
        res['wilcoxon_p_holm_all_pairs'] = holm(res.wilcoxon_p.values)
        res['corrected_t_p_holm_all_pairs'] = holm(res.corrected_t_p.fillna(1).values)
        vb = res.vs_best.values
        res['wilcoxon_p_holm_vs_best'] = np.nan; res.loc[vb, 'wilcoxon_p_holm_vs_best'] = holm(res.wilcoxon_p.values[vb])
        res['corrected_t_p_holm_vs_best'] = np.nan; res.loc[vb, 'corrected_t_p_holm_vs_best'] = holm(res.corrected_t_p.fillna(1).values[vb])
        pw_rows.append(res)
    pd.DataFrame(st_rows).to_csv(R / 'statistical_tests' / 'friedman_tests.csv', index=False)
    pd.concat(pw_rows).to_csv(R / 'statistical_tests' / 'pairwise_tests.csv', index=False)
# binary-trained vs 3-class-collapsed, per model (paired over folds), Holm across the models per metric
cmp_rows = []
for met in ['roc_auc', 'sensitivity', 'specificity', 'mcc', 'balanced_accuracy']:
    rr = []
    for m in full:
        a = fm[(fm.rule == 'binary_thr0.5') & (fm.model == m)].set_index('fold')[met]
        b = fm[(fm.rule == 'from_3class') & (fm.model == m)].set_index('fold')[met]
        j = a.index.intersection(b.index); d = (a[j] - b[j]).values
        if len(j) < N_FOLDS: continue
        rr.append(dict(metric=met, model=m, binary_mean=a[j].mean(), from_3class_mean=b[j].mean(), mean_diff=d.mean(),
                       wilcoxon_p=stats.wilcoxon(a[j], b[j], zero_method='zsplit').pvalue if np.any(d != 0) else 1.0,
                       rank_biserial=rank_biserial(d)))
    if rr:
        rr = pd.DataFrame(rr); rr['wilcoxon_p_holm'] = holm(rr.wilcoxon_p.values); cmp_rows.append(rr)
if cmp_rows: pd.concat(cmp_rows).to_csv(R / 'statistical_tests' / 'binary_vs_3class_collapsed.csv', index=False)

# ---------------- figures
def bar(metric, fname, ylab):
    g = nb.loc[order]; fig, ax = plt.subplots(figsize=(7.2, 3.2)); x = np.arange(len(order))
    ax.bar(x, g[f'{metric}_mean'] * 100, yerr=g[f'{metric}_sd'] * 100, color='#0072B2', alpha=.85, capsize=3, error_kw=dict(lw=.8), width=.65)
    for k, m in enumerate(order):
        v = B[B.model == m][metric].values * 100
        ax.scatter(np.full(len(v), k) + np.random.RandomState(k).uniform(-.18, .18, len(v)), v, s=6, color='k', alpha=.5, zorder=3, lw=0)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=40, ha='right'); ax.set_ylabel(ylab); ax.set_ylim(0, 105)
    ax.set_title(f'Cancer vs non-cancer: {NICE[metric]} (mean ± SD; dots = folds)')
    ax.yaxis.grid(True, lw=.4, alpha=.5); ax.set_axisbelow(True); save(fig, R / 'comparison_plots' / fname)
bar('roc_auc', 'roc_auc_comparison', 'ROC-AUC (%)')
bar('mcc', 'mcc_comparison', 'MCC (×100)')
bar('balanced_accuracy', 'balanced_accuracy_comparison', 'Balanced accuracy (%)')

# sensitivity / specificity and subgroup false-positive rates
g = nb.loc[order]; y_ = np.arange(len(order))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
for i, (c, col, lab) in enumerate([('sensitivity', GCOL['cancer'], 'Sensitivity'), ('specificity', '#555', 'Specificity')]):
    axs[0].barh(y_ + (i - .5) * .38, g[f'{c}_mean'] * 100, .38, xerr=g[f'{c}_sd'] * 100, color=col, label=lab, error_kw=dict(lw=.5, capsize=1.5))
for i, (c, col, lab) in enumerate([('benign_fpr', GCOL['benign'], 'Benign → cancer'), ('normal_fpr', GCOL['normal'], 'Normal → cancer')]):
    axs[1].barh(y_ + (i - .5) * .38, g[f'{c}_mean'] * 100, .38, xerr=g[f'{c}_sd'] * 100, color=col, label=lab, error_kw=dict(lw=.5, capsize=1.5))
axs[0].set_title('Sensitivity / specificity (%)'); axs[1].set_title('False-positive rate by subgroup (%)')
for ax in axs: ax.set_xlim(0, 105); ax.xaxis.grid(True, lw=.4, alpha=.5); ax.set_axisbelow(True); ax.legend(fontsize=7, frameon=False, loc='lower right')
axs[0].set_yticks(y_); axs[0].set_yticklabels(order); axs[0].invert_yaxis()
fig.suptitle(f'Cancer vs non-cancer, threshold 0.5 (mean ± SD over {N_FOLDS} folds)'); fig.tight_layout(); save(fig, R / 'comparison_plots' / 'sensitivity_specificity_subgroups')

# P(cancer) distribution by true subgroup, pooled out-of-fold
fig, axs = plt.subplots(3, 4, figsize=(7.2, 5.6), sharex=True, sharey=True)
for ax in axs.ravel()[len(order):]: ax.axis('off')
for ax, m in zip(axs.ravel(), order):
    P = pd.concat([preds[(m, k)] for k in range(N_FOLDS) if (m, k) in preds])
    data = [P[P.label == 'normal'].p_cancer, P[P.label == 'benign'].p_cancer, P[P.label == 'malignant'].p_cancer]
    bp = ax.boxplot(data, patch_artist=True, widths=.6, medianprops=dict(color='k', lw=1), flierprops=dict(markersize=2))
    for b_, c in zip(bp['boxes'], [GCOL['normal'], GCOL['benign'], GCOL['cancer']]): b_.set(facecolor=c, alpha=.6, lw=.6)
    ax.axhline(.5, color='k', lw=.5, ls=':'); ax.set_title(m, fontsize=8); ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['Nor', 'Ben', 'Mal'], fontsize=7)
fig.supylabel('Predicted P(cancer)'); fig.suptitle('Predicted cancer probability by true subgroup (pooled test folds)'); fig.tight_layout()
save(fig, R / 'comparison_plots' / 'p_cancer_by_subgroup')

# binary-trained vs 3-class collapsed (AUC, sensitivity)
if cmp_rows:
    C = pd.concat(cmp_rows).set_index(['metric', 'model'])
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.4), sharey=True)
    for ax, met in zip(axs, ['roc_auc', 'sensitivity', 'specificity']):
        c = C.loc[met].reindex(order).dropna()
        for i, (m, r) in enumerate(c.iterrows()):
            ax.plot([r.from_3class_mean * 100, r.binary_mean * 100], [i, i], color='#bbb', lw=1, zorder=1)
        ax.scatter(c.from_3class_mean * 100, range(len(c)), color='#999', s=14, label='3-class collapsed', zorder=2)
        ax.scatter(c.binary_mean * 100, range(len(c)), color=GCOL['cancer'], s=14, label='Binary-trained', zorder=3)
        ax.set_title(NICE[met].split(' (')[0] + ' (%)'); ax.xaxis.grid(True, lw=.4, alpha=.5); ax.set_axisbelow(True)
    axs[0].set_yticks(range(len(c))); axs[0].set_yticklabels(c.index); axs[0].invert_yaxis(); axs[2].legend(fontsize=7, frameon=False, loc='lower left')
    fig.suptitle('Binary-trained models vs 3-class models collapsed to cancer / non-cancer (fold means)'); fig.tight_layout()
    save(fig, R / 'comparison_plots' / 'binary_vs_3class_collapsed')

# ROC and PR curves (pooled out-of-fold)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.5))
cmap = plt.get_cmap('tab20')
for i, m in enumerate(order):
    P = pd.concat([preds[(m, k)] for k in range(N_FOLDS) if (m, k) in preds])
    f_, t_, _ = roc_curve(P.y, P.p_cancer); pr_, rc_, _ = precision_recall_curve(P.y, P.p_cancer)
    o = [r for r in oof if r['model'] == m][0]
    axs[0].plot(f_, t_, lw=.9, color=cmap(i), label=f"{m} ({o['roc_auc']:.3f})"); axs[1].plot(rc_, pr_, lw=.9, color=cmap(i), label=f"{m} ({o['pr_auc']:.3f})")
axs[0].plot([0, 1], [0, 1], 'k:', lw=.6); axs[1].axhline(P.y.mean(), color='k', ls=':', lw=.6)  # prevalence = no-skill precision
axs[0].set(xlabel='False-positive rate', ylabel='True-positive rate', title='ROC (pooled test folds)')
axs[1].set(xlabel='Recall (sensitivity)', ylabel='Precision', title='Precision–recall (pooled test folds)')
for ax in axs: ax.legend(fontsize=5.5, frameon=False, loc='lower right' if ax is axs[0] else 'upper right'); ax.grid(True, lw=.3, alpha=.5)
fig.tight_layout(); save(fig, R / 'roc_curves' / 'roc_pr_all_models')

# confusion matrices: true 3-class subgroup (rows) x predicted binary (cols), pooled out-of-fold
fig, axs = plt.subplots(3, 4, figsize=(7.2, 6.0))
for ax in axs.ravel()[len(order):]: ax.axis('off')
for ax, m in zip(axs.ravel(), order):
    P = pd.concat([preds[(m, k)] for k in range(N_FOLDS) if (m, k) in preds])
    cm = np.array([[((P.label == g_) & (P.y_pred == j)).sum() for j in (0, 1)] for g_ in ('normal', 'benign', 'malignant')])
    cmn = cm / cm.sum(1, keepdims=True); ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(3):
        for j in range(2): ax.text(j, i, f'{cm[i, j]}\n{cmn[i, j] * 100:.0f}%', ha='center', va='center', fontsize=6, color='w' if cmn[i, j] > .55 else 'k')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Non-cancer', 'Cancer'], fontsize=6.5); ax.set_yticks(range(3)); ax.set_yticklabels(['Nor', 'Ben', 'Mal'], fontsize=7)
    ax.set_title(m, fontsize=8); ax.spines[:].set_visible(False)
fig.suptitle('True subgroup × predicted class, threshold 0.5 (pooled test folds)'); fig.tight_layout(); save(fig, R / 'confusion_matrices' / 'all_models_subgroup_confusion')

# malignant patients: mean predicted cancer probability over all models (binary model and 3-class model), and the
# mean P(cancer) of the non-cancer images in the same test fold -> shows which patients drive the fold-to-fold spread
AP = pd.concat([v.assign(model=m) for (m, k), v in preds.items()])
if 'p3_malignant' in AP:
    mp = AP[AP.label == 'malignant'].groupby(['fold', 'patient_id']).agg(n_images=('file', 'nunique'), binary_p_cancer=('p_cancer', 'mean'),
                                                                         three_class_p_malignant=('p3_malignant', 'mean'))
    nc = AP[AP.label != 'malignant'].groupby(['fold', 'label']).p_cancer.mean().unstack().add_prefix('binary_p_cancer_')
    mp.join(nc, on='fold').to_csv(R / 'tables' / 'malignant_patients_mean_probability.csv')
    from sklearn.metrics import roc_auc_score
    lv = []
    for m in order:
        Q = AP[AP.model == m]; Gp = Q.groupby('patient_id').agg(y=('y', 'first'), pb=('p_cancer', 'mean'), p3=('p3_malignant', 'mean'))
        lv.append(dict(model=m, pooled_image_auc_binary=roc_auc_score(Q.y, Q.p_cancer), pooled_image_auc_3class=roc_auc_score(Q.y, Q.p3_malignant),
                       patient_auc_binary=roc_auc_score(Gp.y, Gp.pb), patient_auc_3class=roc_auc_score(Gp.y, Gp.p3)))
    pd.DataFrame(lv).set_index('model').to_csv(R / 'tables' / 'pooled_and_patient_auc_binary_vs_3class.csv')

print('complete runs:', int(cp.n_complete.sum()), '/', len(MODELS) * N_FOLDS)
print(pd.read_csv(R / 'tables' / 'main_table_binary_thr0.5.csv', encoding='utf-8-sig').to_string(index=False))
print(sg.to_string())
