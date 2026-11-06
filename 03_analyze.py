"""Tables, figures and statistics from saved fold outputs only (no retraining).

  python 03_analyze.py            # uses results/
Only (model, fold) pairs with done.json are used; completeness is reported in tables/completeness.csv.
"""
import json, itertools, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc as sk_auc, confusion_matrix
from common import *
from metrics import fold_metrics

warnings.filterwarnings('ignore', category=RuntimeWarning)
_to_csv = pd.DataFrame.to_csv  # utf-8 with BOM so Excel shows ± correctly
pd.DataFrame.to_csv = lambda self, *a, **k: _to_csv(self, *a, **{'encoding': 'utf-8-sig', **k})
import sys
R = RESULTS / 'smoke' if 'smoke' in sys.argv[1:] else RESULTS
for s in ['fold_metrics', 'confusion_matrices', 'roc_curves', 'training_curves', 'comparison_plots', 'statistical_tests', 'tables']:
    (R / s).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight', 'figure.dpi': 100})
OKABE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000']
CCOL = {'benign': '#009E73', 'malignant': '#D55E00', 'normal': '#0072B2'}
NICE = dict(accuracy='Accuracy', precision_macro='Precision (macro)', recall_macro='Recall (macro)', f1_macro='F1-score (macro)',
            f1_weighted='F1-score (weighted)', specificity_macro='Specificity (macro)', roc_auc_ovr_macro='ROC-AUC (OvR macro)',
            balanced_accuracy='Balanced accuracy', cohen_kappa="Cohen's kappa", mcc='MCC')
PCT = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted', 'specificity_macro', 'roc_auc_ovr_macro', 'balanced_accuracy']


def save(fig, path):
    fig.savefig(str(path) + '.png', dpi=400); fig.savefig(str(path) + '.pdf'); plt.close(fig)


# ---------------- collect
rows, comp, logs, preds = [], [], {}, {}
for name in MODELS:
    for k in range(N_FOLDS):
        d = R / 'trained_models' / name / f'fold_{k:02d}'
        st = 'complete' if (d / 'done.json').exists() else 'failed' if (d / 'error.txt').exists() else 'not run'
        comp.append(dict(model=name, fold=k, status=st))
        if st != 'complete': continue
        info = json.load(open(d / 'done.json'))
        p = pd.read_csv(R / 'predictions' / name / f'fold_{k:02d}_test_predictions.csv')
        met = fold_metrics(p.y.values, p[[f'p_{c}' for c in CLASSES]].values)  # recomputed from saved predictions
        rows.append(dict(model=name, fold=k, **{kk: v for kk, v in met.items() if kk != 'confusion_matrix'},
                         confusion_matrix=json.dumps(met['confusion_matrix']), best_phase=info['best_phase'],
                         best_epoch=info['best_global_epoch'], epochs_run=info['epochs_run'], best_val_loss=info['best_val_loss'],
                         best_val_f1_macro=info['best_val_f1_macro'], train_time_s=info['train_time_s'],
                         latency_ms=info['latency_ms_bs1_median'], params_M=info['params_total'] / 1e6, input_size=info['input_size']))
        logs[(name, k)] = pd.read_csv(d / 'training_log.csv'); preds[(name, k)] = p.assign(fold=k)
comp = pd.DataFrame(comp)
cp = comp.pivot(index='model', columns='fold', values='status').reindex(list(MODELS))
cp['n_complete'] = (cp == 'complete').sum(1); cp.to_csv(R / 'tables' / 'completeness.csv')
fm = pd.DataFrame(rows)
if fm.empty: raise SystemExit('no completed folds yet')
fm.to_csv(R / 'fold_metrics' / 'all_models_fold_metrics.csv', index=False)
for name, g in fm.groupby('model'): g.to_csv(R / 'fold_metrics' / f'{name}_fold_metrics.csv', index=False)
models = [m for m in MODELS if m in set(fm.model)]
full = [m for m in models if cp.loc[m, 'n_complete'] == N_FOLDS]

# ---------------- summary tables (mean +- SD over folds; SD = sample SD, ddof=1)
def msd(x, pct):
    x = np.asarray(x, float); f = 100 if pct else 1; n = np.isfinite(x).sum()
    return f'{np.nanmean(x) * f:.2f} ± {np.nanstd(x, ddof=1) * f:.2f}' + ('' if n == len(x) else f' (n={n})')
cols = list(NICE)
summ, num = [], []
for m in models:
    g = fm[fm.model == m]
    summ.append(dict(Model=m, folds=len(g), **{NICE[c] + (' (%)' if c in PCT else ''): msd(g[c], c in PCT) for c in cols},
                     **{'Params (M)': f"{g.params_M.iloc[0]:.2f}", 'Input': int(g.input_size.iloc[0]),
                        'Train time / fold (min)': msd(g.train_time_s / 60, False), 'Latency bs1 (ms)': f"{g.latency_ms.median():.2f}"}))
    num.append(dict(model=m, folds=len(g), **{f'{c}_mean': g[c].mean() for c in cols}, **{f'{c}_sd': g[c].std(ddof=1) for c in cols},
                    params_M=g.params_M.iloc[0], latency_ms=g.latency_ms.median(), train_time_min=g.train_time_s.mean() / 60,
                    auc_folds_undefined=int(g.roc_auc_ovr_macro.isna().sum())))
summ = pd.DataFrame(summ); num = pd.DataFrame(num)
order = num.sort_values('f1_macro_mean', ascending=False).model.tolist()
summ.set_index('Model').loc[order].to_csv(R / 'tables' / 'summary_mean_sd_all_metrics.csv')
num.set_index('model').loc[order].to_csv(R / 'tables' / 'summary_numeric.csv')
main = summ.set_index('Model').loc[order][['folds', 'Accuracy (%)', 'Precision (macro) (%)', 'Recall (macro) (%)', 'F1-score (macro) (%)',
                                           'ROC-AUC (OvR macro) (%)', 'MCC']]
main.columns = ['Folds', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-score (%)', 'AUC (%)', 'MCC']
main.to_csv(R / 'tables' / 'main_table.csv')
(R / 'tables' / 'main_table.md').write_text(main.to_markdown(), encoding='utf-8')
(R / 'tables' / 'main_table.tex').write_text(main.to_latex(escape=True), encoding='utf-8')
(R / 'tables' / 'summary_mean_sd_all_metrics.md').write_text(summ.set_index('Model').loc[order].to_markdown(), encoding='utf-8')
auc_na = fm[fm.roc_auc_ovr_macro.isna()][['model', 'fold', 'auc_note']]
auc_na.to_csv(R / 'tables' / 'auc_undefined_folds.csv', index=False)

# per-class (mean +- SD over folds) and pooled out-of-fold
pc = []
for m in models:
    g = fm[fm.model == m]
    for c in CLASSES:
        pc.append(dict(model=m, cls=c, **{f'{s}': msd(g[f'{c}_{s}'], True) for s in ['precision', 'recall', 'f1', 'specificity', 'auc']}))
pd.DataFrame(pc).to_csv(R / 'tables' / 'per_class_mean_sd.csv', index=False)
oof, pat = [], []
for m in models:
    P = pd.concat([preds[(m, k)] for k in range(N_FOLDS) if (m, k) in preds])
    pr = P[[f'p_{c}' for c in CLASSES]].values
    o = fold_metrics(P.y.values, pr); o.pop('confusion_matrix'); oof.append(dict(model=m, n=len(P), **o))
    # patient-level: mean probability over a patient's images (supplementary; each patient is in exactly one test fold)
    G = P.groupby('patient_id').agg(y=('y', 'first'), **{f'p_{c}': (f'p_{c}', 'mean') for c in CLASSES})
    q = fold_metrics(G.y.values, G[[f'p_{c}' for c in CLASSES]].values); q.pop('confusion_matrix'); pat.append(dict(model=m, n_patients=len(G), **q))
pd.DataFrame(oof).set_index('model').loc[order].to_csv(R / 'tables' / 'pooled_oof_metrics.csv')
pd.DataFrame(pat).set_index('model').loc[order].to_csv(R / 'tables' / 'patient_level_oof_metrics.csv')

# ---------------- statistics (folds as matched blocks; only models with all N_FOLDS folds)
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
st_rows, pw_rows = [], []
n_tr, n_te = fm.groupby('fold').n_test.first().mean() * (N_FOLDS - 2), fm.groupby('fold').n_test.first().mean()
if len(full) >= 3:
    for met in ['f1_macro', 'accuracy', 'roc_auc_ovr_macro', 'mcc', 'balanced_accuracy']:
        M = fm[fm.model.isin(full)].pivot(index='fold', columns='model', values=met)[full].dropna()
        chi, p = stats.friedmanchisquare(*[M[c] for c in full])
        W = chi / (len(M) * (len(full) - 1))
        ranks = M.rank(axis=1, ascending=False).mean().sort_values()
        st_rows.append(dict(metric=met, n_folds=len(M), n_models=len(full), friedman_chi2=chi, p_value=p, kendalls_W=W,
                            mean_ranks='; '.join(f'{a}={b:.2f}' for a, b in ranks.items())))
        ranks.rename('mean_rank').to_csv(R / 'statistical_tests' / f'mean_ranks_{met}.csv')
        best = ranks.index[0]
        prs = list(itertools.combinations(full, 2))
        res = []
        for a, b in prs:
            d = (M[a] - M[b]).values
            pw = stats.wilcoxon(M[a], M[b], zero_method='zsplit').pvalue if np.any(d != 0) else 1.0
            res.append(dict(metric=met, model_a=a, model_b=b, mean_diff=d.mean(), median_diff=np.median(d),
                            wilcoxon_p=pw, rank_biserial=rank_biserial(d), corrected_t_p=corrected_t(d, n_tr, n_te), vs_best=best in (a, b)))
        res = pd.DataFrame(res)
        res['wilcoxon_p_holm_all_pairs'] = holm(res.wilcoxon_p.values)
        res['corrected_t_p_holm_all_pairs'] = holm(res.corrected_t_p.fillna(1).values)
        vb = res.vs_best.values
        res['wilcoxon_p_holm_vs_best'] = np.nan; res.loc[vb, 'wilcoxon_p_holm_vs_best'] = holm(res.wilcoxon_p.values[vb])
        res['corrected_t_p_holm_vs_best'] = np.nan; res.loc[vb, 'corrected_t_p_holm_vs_best'] = holm(res.corrected_t_p.fillna(1).values[vb])
        pw_rows.append(res)
    pd.DataFrame(st_rows).to_csv(R / 'statistical_tests' / 'friedman_tests.csv', index=False)
    pd.concat(pw_rows).to_csv(R / 'statistical_tests' / 'pairwise_tests.csv', index=False)

# ---------------- figures
def bar(metric, fname, ylab):
    g = num.set_index('model').loc[order]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(order)); ax.bar(x, g[f'{metric}_mean'] * 100, yerr=g[f'{metric}_sd'] * 100, color='#0072B2', alpha=.85,
                                      capsize=3, error_kw=dict(lw=.8), width=.65)
    for k, m in enumerate(order):  # individual folds
        v = fm[fm.model == m][metric].values * 100
        ax.scatter(np.full(len(v), k) + np.random.RandomState(k).uniform(-.18, .18, len(v)), v, s=6, color='k', alpha=.5, zorder=3, lw=0)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=40, ha='right'); ax.set_ylabel(ylab); ax.set_ylim(0, 105)
    ax.set_title(f'{NICE[metric]} across {N_FOLDS} patient-grouped test folds (mean ± SD; dots = folds)')
    ax.yaxis.grid(True, lw=.4, alpha=.5); ax.set_axisbelow(True); save(fig, R / 'comparison_plots' / fname)
bar('accuracy', 'accuracy_comparison', 'Accuracy (%)')
bar('f1_macro', 'f1_macro_comparison', 'Macro F1-score (%)')
bar('roc_auc_ovr_macro', 'roc_auc_comparison', 'ROC-AUC, OvR macro (%)')
bar('mcc', 'mcc_comparison', 'MCC (×100)')

fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.2), sharex=True)
for ax, met in zip(axs.ravel(), ['accuracy', 'f1_macro', 'roc_auc_ovr_macro', 'mcc']):
    data = [fm[fm.model == m][met].dropna().values for m in order]
    bp = ax.boxplot(data, patch_artist=True, widths=.6, medianprops=dict(color='k', lw=1), flierprops=dict(markersize=3))
    for b in bp['boxes']: b.set(facecolor='#56B4E9', alpha=.6, lw=.7)
    ax.set_title(NICE[met]); ax.yaxis.grid(True, lw=.4, alpha=.5); ax.set_axisbelow(True)
for ax in axs[1]: ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=55, ha='right')
fig.suptitle(f'Fold-wise test performance ({N_FOLDS} folds per model)', y=1.0); fig.tight_layout(); save(fig, R / 'comparison_plots' / 'foldwise_boxplots')

# per-class P/R/F1 (mean over folds)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.4), sharey=True)
for ax, s in zip(axs, ['precision', 'recall', 'f1']):
    w = .26
    for i, c in enumerate(CLASSES):
        mu = [fm[fm.model == m][f'{c}_{s}'].mean() * 100 for m in order]; sd = [fm[fm.model == m][f'{c}_{s}'].std(ddof=1) * 100 for m in order]
        ax.barh(np.arange(len(order)) + (i - 1) * w, mu, w, xerr=sd, color=CCOL[c], label=c.capitalize(), error_kw=dict(lw=.5, capsize=1.5))
    ax.set_title({'f1': 'F1-score'}.get(s, s.capitalize()) + ' (%)'); ax.set_xlim(0, 105); ax.xaxis.grid(True, lw=.4, alpha=.5); ax.set_axisbelow(True)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(order); axs[0].invert_yaxis()
axs[2].legend(loc='lower right', fontsize=7, frameon=False); fig.suptitle('Class-wise performance (mean ± SD over folds)'); fig.tight_layout()
save(fig, R / 'comparison_plots' / 'per_class_precision_recall_f1')

# efficiency
g = num.set_index('model').loc[order]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
for ax, col, lab, lg in [(axs[0], 'params_M', 'Parameters (millions, log scale)', True), (axs[1], 'latency_ms', 'Inference latency, batch 1 (ms)', False)]:
    ax.errorbar(g[col], g.f1_macro_mean * 100, yerr=g.f1_macro_sd * 100, fmt='o', ms=4, color='#0072B2', ecolor='#999', elinewidth=.7, capsize=2)
    for m, r in g.iterrows(): ax.annotate(m, (r[col], r.f1_macro_mean * 100), fontsize=6.5, xytext=(3, 3), textcoords='offset points')
    if lg: ax.set_xscale('log')
    ax.set_xlabel(lab); ax.set_ylabel('Macro F1-score (%)'); ax.grid(True, lw=.4, alpha=.5)
fig.suptitle('Performance versus model complexity'); fig.tight_layout(); save(fig, R / 'comparison_plots' / 'performance_vs_complexity')

# confusion matrices (pooled out-of-fold)
n = len(order); cols_ = 4; rows_ = int(np.ceil(n / cols_))
fig, axs = plt.subplots(rows_, cols_, figsize=(7.2, 1.95 * rows_ + .3))
for ax in np.ravel(axs)[n:]: ax.axis('off')
for ax, m in zip(np.ravel(axs), order):
    P = pd.concat([preds[(m, k)] for k in range(N_FOLDS) if (m, k) in preds])
    cm = confusion_matrix(P.y, P.y_pred, labels=range(3)); cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3): ax.text(j, i, f'{cm[i, j]}\n{cmn[i, j] * 100:.0f}%', ha='center', va='center', fontsize=6, color='w' if cmn[i, j] > .55 else 'k')
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(['Ben', 'Mal', 'Nor'], fontsize=7); ax.set_yticklabels(['Ben', 'Mal', 'Nor'], fontsize=7)
    ax.set_title(m, fontsize=8); ax.spines[:].set_visible(False)
    # individual figure too
    f2, a2 = plt.subplots(figsize=(3.2, 2.9)); im = a2.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3): a2.text(j, i, f'{cm[i, j]}\n({cmn[i, j] * 100:.1f}%)', ha='center', va='center', fontsize=8, color='w' if cmn[i, j] > .55 else 'k')
    a2.set_xticks(range(3)); a2.set_yticks(range(3)); a2.set_xticklabels([c.capitalize() for c in CLASSES]); a2.set_yticklabels([c.capitalize() for c in CLASSES])
    a2.set_xlabel('Predicted'); a2.set_ylabel('True'); a2.set_title(f'{m}: out-of-fold confusion matrix'); a2.spines[:].set_visible(False)
    f2.colorbar(im, fraction=.046, pad=.04, label='Row-normalised'); save(f2, R / 'confusion_matrices' / f'{m}_oof_confusion_matrix')
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(R / 'confusion_matrices' / f'{m}_oof_confusion_matrix.csv')
fig.text(.5, -.01, 'Predicted', ha='center'); fig.text(-.01, .5, 'True', va='center', rotation=90)
fig.suptitle('Out-of-fold confusion matrices (counts and row %)'); fig.tight_layout(); save(fig, R / 'confusion_matrices' / 'all_models_oof_confusion_matrices')

# ROC curves (pooled out-of-fold, one-vs-rest)
grid = np.linspace(0, 1, 201)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for ci, (ax, c) in enumerate(zip(axs, CLASSES)):
    for mi, m in enumerate(order):
        P = pd.concat([preds[(m, k)] for k in range(N_FOLDS) if (m, k) in preds])
        fpr, tpr, _ = roc_curve(P.y == ci, P[f'p_{c}'])
        ax.plot(fpr, tpr, lw=.9, color=plt.cm.tab20(mi / max(len(order) - 1, 1)), label=f'{m} ({sk_auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k:', lw=.6); ax.set_title(f'{c.capitalize()} vs rest'); ax.set_xlabel('False positive rate'); ax.set_aspect('equal')
    ax.legend(fontsize=4.6, loc='lower right', frameon=False)
axs[0].set_ylabel('True positive rate'); fig.suptitle('Out-of-fold ROC curves (AUC in legend)'); fig.tight_layout()
save(fig, R / 'roc_curves' / 'all_models_oof_roc_per_class')
for m in order:
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for ci, c in enumerate(CLASSES):
        tprs = []
        for k in range(N_FOLDS):
            if (m, k) not in preds: continue
            P = preds[(m, k)]; yb = P.y == ci
            if yb.all() or not yb.any(): continue
            fpr, tpr, _ = roc_curve(yb, P[f'p_{c}']); tprs.append(np.interp(grid, fpr, tpr)); tprs[-1][0] = 0
        mu, sd = np.mean(tprs, 0), np.std(tprs, 0, ddof=1)
        ax.plot(grid, mu, color=CCOL[c], lw=1.2, label=f'{c.capitalize()} (AUC {np.trapz(mu, grid):.3f})')
        ax.fill_between(grid, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=CCOL[c], alpha=.12, lw=0)
    ax.plot([0, 1], [0, 1], 'k:', lw=.6); ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); ax.set_aspect('equal')
    ax.set_title(f'{m}: fold-averaged OvR ROC (±1 SD)'); ax.legend(fontsize=7, frameon=False, loc='lower right')
    save(fig, R / 'roc_curves' / f'{m}_foldmean_roc')

# training curves
for m in order:
    fig, axs = plt.subplots(2, 5, figsize=(7.2, 3.6), sharey=False)
    for k, ax in enumerate(axs.ravel()):
        if (m, k) not in logs: ax.axis('off'); continue
        L = logs[(m, k)]; e = L.global_epoch
        ax.plot(e, L.train_loss, color='#0072B2', lw=.8, label='Train loss'); ax.plot(e, L.val_loss, color='#D55E00', lw=.8, label='Val loss')
        a2 = ax.twinx(); a2.plot(e, L.train_acc, color='#0072B2', lw=.6, ls='--'); a2.plot(e, L.val_acc, color='#D55E00', lw=.6, ls='--')
        a2.set_ylim(0, 1); a2.tick_params(labelsize=5); ax.tick_params(labelsize=5)
        ft = L[L.phase == 'finetune'].global_epoch.min()
        if np.isfinite(ft): ax.axvline(ft - .5, color='grey', lw=.5, ls=':')
        b = L[L.best].global_epoch.max(); ax.axvline(b, color='k', lw=.5)
        ax.set_title(f'Fold {k}', fontsize=7)
    axs[0, 0].legend(fontsize=5, frameon=False)
    fig.suptitle(f'{m}: loss (solid, left axis) and accuracy (dashed, right axis); dotted = start of fine-tuning, black = selected epoch', fontsize=7)
    fig.tight_layout(); save(fig, R / 'training_curves' / f'{m}_training_curves')

print(cp.to_string()); print(main.to_string())
if st_rows: print(pd.DataFrame(st_rows)[['metric', 'friedman_chi2', 'p_value', 'kendalls_W']].to_string())
