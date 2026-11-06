"""Tables, figures and statistics for the public datasets, from saved fold outputs only (no retraining).

  python 03_analyze_public.py                 # every dataset that has finished folds
  python 03_analyze_public.py NDO_4class      # one dataset
Only (model, fold) pairs with done.json are used; completeness is reported in tables/completeness.csv.
Cross-dataset overview: results_public/overview/.
"""
import json, itertools, sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc as sk_auc, confusion_matrix
from common_public import *
from metrics_public import fold_metrics

warnings.filterwarnings('ignore', category=RuntimeWarning)
_to_csv = pd.DataFrame.to_csv  # utf-8 with BOM so Excel shows ± correctly
pd.DataFrame.to_csv = lambda self, *a, **k: _to_csv(self, *a, **{'encoding': 'utf-8-sig', **k})
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight', 'figure.dpi': 100})
OKABE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
PCT = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted', 'specificity_macro', 'roc_auc', 'balanced_accuracy',
       'sensitivity', 'specificity', 'pr_auc']
NICE = dict(accuracy='Accuracy', precision_macro='Precision (macro)', recall_macro='Recall (macro)', f1_macro='F1-score (macro)',
            f1_weighted='F1-score (weighted)', specificity_macro='Specificity (macro)', roc_auc='ROC-AUC',
            balanced_accuracy='Balanced accuracy', cohen_kappa="Cohen's kappa", mcc='MCC', sensitivity='Sensitivity',
            specificity='Specificity', pr_auc='PR-AUC')


def save(fig, path):
    fig.savefig(str(path) + '.png', dpi=400); fig.savefig(str(path) + '.pdf'); plt.close(fig)


def msd(x, pct):
    x = np.asarray(x, float); f = 100 if pct else 1; n = np.isfinite(x).sum()
    return f'{np.nanmean(x) * f:.2f} ± {np.nanstd(x, ddof=1) * f:.2f}' + ('' if n == len(x) else f' (n={n})')


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
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


def analyze(DSN):
    cfg = DATASETS[DSN]; C = cfg['classes']; K = cfg['k']; R = out_dir(DSN); binary = len(C) == 2
    for s in ['fold_metrics', 'confusion_matrices', 'roc_curves', 'comparison_plots', 'statistical_tests', 'tables']:
        (R / s).mkdir(parents=True, exist_ok=True)
    rows, comp, preds = [], [], {}
    for name in MODELS:
        for k in range(K):
            d = R / 'trained_models' / name / f'fold_{k:02d}'
            st = 'complete' if (d / 'done.json').exists() else 'failed' if (d / 'error.txt').exists() else 'not run'
            comp.append(dict(model=name, fold=k, status=st))
            if st != 'complete': continue
            info = json.load(open(d / 'done.json'))
            p = pd.read_csv(R / 'predictions' / name / f'fold_{k:02d}_test_predictions.csv')
            met = fold_metrics(p.y.values, p[[f'p_{c}' for c in C]].values, C)  # recomputed from saved predictions
            rows.append(dict(model=name, fold=k, **{kk: v for kk, v in met.items() if kk != 'confusion_matrix'},
                             confusion_matrix=json.dumps(met['confusion_matrix']), best_phase=info['best_phase'],
                             best_epoch=info['best_global_epoch'], epochs_run=info['epochs_run'], n_train=info['n_train'],
                             train_time_s=info['train_time_s'], latency_ms=info['latency_ms_bs1_median'],
                             params_M=info['params_total'] / 1e6, input_size=info['input_size']))
            preds[(name, k)] = p.assign(fold=k)
    comp = pd.DataFrame(comp)
    cp = comp.pivot(index='model', columns='fold', values='status').reindex(list(MODELS))
    cp['n_complete'] = (cp == 'complete').sum(1); cp.to_csv(R / 'tables' / 'completeness.csv')
    fm = pd.DataFrame(rows)
    if fm.empty: print(DSN, 'no completed folds yet'); return None
    fm.to_csv(R / 'fold_metrics' / 'all_models_fold_metrics.csv', index=False)
    models = [m for m in MODELS if m in set(fm.model)]
    full = [m for m in models if cp.loc[m, 'n_complete'] == K]

    cols = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted', 'specificity_macro', 'roc_auc',
            'balanced_accuracy', 'cohen_kappa', 'mcc'] + (['sensitivity', 'specificity', 'pr_auc'] if binary else [])
    summ, num = [], []
    for m in models:
        g = fm[fm.model == m]
        summ.append(dict(Model=m, folds=len(g), **{NICE[c] + (' (%)' if c in PCT else ''): msd(g[c], c in PCT) for c in cols},
                         **{'Params (M)': f"{g.params_M.iloc[0]:.2f}", 'Input': int(g.input_size.iloc[0]),
                            'Train time / fold (min)': msd(g.train_time_s / 60, False), 'Latency bs1 (ms)': f"{g.latency_ms.median():.2f}"}))
        num.append(dict(model=m, folds=len(g), **{f'{c}_mean': g[c].mean() for c in cols}, **{f'{c}_sd': g[c].std(ddof=1) for c in cols},
                        params_M=g.params_M.iloc[0], latency_ms=g.latency_ms.median(), train_time_min=g.train_time_s.mean() / 60))
    summ = pd.DataFrame(summ); num = pd.DataFrame(num)
    order = num.sort_values('f1_macro_mean', ascending=False).model.tolist()
    summ.set_index('Model').loc[order].to_csv(R / 'tables' / 'summary_mean_sd_all_metrics.csv')
    num.set_index('model').loc[order].to_csv(R / 'tables' / 'summary_numeric.csv')
    mc = ['folds', 'Accuracy (%)', 'Precision (macro) (%)', 'Recall (macro) (%)', 'F1-score (macro) (%)', 'ROC-AUC (%)', 'MCC']
    mn = ['Folds', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-score (%)', 'AUC (%)', 'MCC']
    if binary:
        mc += ['Sensitivity (%)', 'Specificity (%)']; mn += [f'Sensitivity, {C[1]} (%)', 'Specificity (%)']
    main = summ.set_index('Model').loc[order][mc]; main.columns = mn
    main.to_csv(R / 'tables' / 'main_table.csv')
    (R / 'tables' / 'main_table.md').write_text(main.to_markdown(), encoding='utf-8')
    (R / 'tables' / 'main_table.tex').write_text(main.to_latex(escape=True), encoding='utf-8')

    pc = []
    for m in models:
        g = fm[fm.model == m]
        for c in C:
            pc.append(dict(model=m, cls=c, **{s: msd(g[f'{c}_{s}'], True) for s in ['precision', 'recall', 'f1', 'specificity', 'auc']}))
    pc = pd.DataFrame(pc); pc.to_csv(R / 'tables' / 'per_class_mean_sd.csv', index=False)
    pcw = pc.pivot(index='model', columns='cls', values='recall').loc[order][C]
    (R / 'tables' / 'per_class_recall.md').write_text(pcw.to_markdown(), encoding='utf-8')

    oof, pat, sub = [], [], []
    for m in models:
        P = pd.concat([preds[(m, k)] for k in range(K) if (m, k) in preds])
        o = fold_metrics(P.y.values, P[[f'p_{c}' for c in C]].values, C); o.pop('confusion_matrix'); oof.append(dict(model=m, n=len(P), **o))
        if True:  # image-level splits as published; no patient-level aggregation
            if DSN == 'NDO_2class':  # OPMD folder = OPMD + OCA: recall of each original subgroup
                for oc, gg in P[P.label == 'OPMD'].groupby('orig_class'):
                    sub.append(dict(model=m, orig_class=oc, n=len(gg), recall_as_OPMD=(gg.y_pred == 1).mean()))
    pd.DataFrame(oof).set_index('model').loc[order].to_csv(R / 'tables' / 'pooled_oof_metrics.csv')
    if pat: pd.DataFrame(pat).set_index('model').loc[order].to_csv(R / 'tables' / 'patient_level_oof_metrics.csv')
    if sub:
        sb = pd.DataFrame(sub).pivot(index='model', columns='orig_class', values='recall_as_OPMD').loc[order]
        sb.to_csv(R / 'tables' / 'subgroup_recall_OPMD_vs_OCA.csv'); (R / 'tables' / 'subgroup_recall_OPMD_vs_OCA.md').write_text((sb * 100).round(1).to_markdown(), encoding='utf-8')

    # statistics: folds as matched blocks, only models with all K folds
    st_rows, pw_rows = [], []
    n_te = fm.groupby('fold').n_test.first().mean(); n_tr = fm.n_train.mean()
    if len(full) >= 3:
        for met in ['f1_macro', 'accuracy', 'roc_auc', 'mcc', 'balanced_accuracy']:
            M = fm[fm.model.isin(full)].pivot(index='fold', columns='model', values=met)[full].dropna()
            chi, p = stats.friedmanchisquare(*[M[c] for c in full]); W = chi / (len(M) * (len(full) - 1))
            ranks = M.rank(axis=1, ascending=False).mean().sort_values()
            st_rows.append(dict(metric=met, n_folds=len(M), n_models=len(full), friedman_chi2=chi, p_value=p, kendalls_W=W,
                                mean_ranks='; '.join(f'{a}={b:.2f}' for a, b in ranks.items())))
            best = ranks.index[0]; res = []
            for a, b in itertools.combinations(full, 2):
                d = (M[a] - M[b]).values
                pw = stats.wilcoxon(M[a], M[b], zero_method='zsplit').pvalue if np.any(d != 0) else 1.0
                res.append(dict(metric=met, model_a=a, model_b=b, mean_diff=d.mean(), wilcoxon_p=pw, rank_biserial=rank_biserial(d),
                                corrected_t_p=corrected_t(d, n_tr, n_te), vs_best=best in (a, b)))
            res = pd.DataFrame(res)
            res['wilcoxon_p_holm_all_pairs'] = holm(res.wilcoxon_p.values)
            res['corrected_t_p_holm_all_pairs'] = holm(res.corrected_t_p.fillna(1).values)
            vb = res.vs_best.values
            res['wilcoxon_p_holm_vs_best'] = np.nan; res.loc[vb, 'wilcoxon_p_holm_vs_best'] = holm(res.wilcoxon_p.values[vb])
            res['corrected_t_p_holm_vs_best'] = np.nan; res.loc[vb, 'corrected_t_p_holm_vs_best'] = holm(res.corrected_t_p.fillna(1).values[vb])
            pw_rows.append(res)
        pd.DataFrame(st_rows).to_csv(R / 'statistical_tests' / 'friedman_tests.csv', index=False)
        pd.concat(pw_rows).to_csv(R / 'statistical_tests' / 'pairwise_tests.csv', index=False)

    # figures
    unit = 'image-level stratified'
    for metric, fname, ylab in [('f1_macro', 'f1_macro_comparison', 'Macro F1-score (%)'), ('roc_auc', 'roc_auc_comparison', 'ROC-AUC (%)'),
                                ('accuracy', 'accuracy_comparison', 'Accuracy (%)'), ('mcc', 'mcc_comparison', 'MCC (×100)')]:
        g = num.set_index('model').loc[order]
        fig, ax = plt.subplots(figsize=(7.2, 3.2)); x = np.arange(len(order))
        ax.bar(x, g[f'{metric}_mean'] * 100, yerr=g[f'{metric}_sd'] * 100, color='#0072B2', alpha=.85, capsize=3, error_kw=dict(lw=.8), width=.65)
        for i, m in enumerate(order):
            v = fm[fm.model == m][metric].values * 100
            ax.scatter(np.full(len(v), i) + np.random.RandomState(i).uniform(-.18, .18, len(v)), v, s=6, color='k', alpha=.5, zorder=3, lw=0)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=40, ha='right'); ax.set_ylabel(ylab); ax.set_ylim(0, 105)
        ax.set_title(f'{DSN}: {NICE[metric]} over {K} {unit} test folds (mean ± SD; dots = folds)', fontsize=9)
        ax.yaxis.grid(True, lw=.4, alpha=.5); ax.set_axisbelow(True); save(fig, R / 'comparison_plots' / fname)

    nC = len(C); short = [c.replace('_lesions', '').replace('NON CANCER', 'Non-ca').replace('CANCER', 'Cancer')[:7] for c in C]
    n = len(order); cols_ = 4; rows_ = int(np.ceil(n / cols_))
    fig, axs = plt.subplots(rows_, cols_, figsize=(7.2, 1.95 * rows_ + .3), squeeze=False)
    for ax in np.ravel(axs)[n:]: ax.axis('off')
    for ax, m in zip(np.ravel(axs), order):
        P = pd.concat([preds[(m, k)] for k in range(K) if (m, k) in preds])
        cm = confusion_matrix(P.y, P.y_pred, labels=range(nC)); cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
        for i in range(nC):
            for j in range(nC): ax.text(j, i, f'{cm[i, j]}\n{cmn[i, j] * 100:.0f}%', ha='center', va='center', fontsize=5.5, color='w' if cmn[i, j] > .55 else 'k')
        ax.set_xticks(range(nC)); ax.set_yticks(range(nC)); ax.set_xticklabels(short, fontsize=6); ax.set_yticklabels(short, fontsize=6)
        ax.set_title(m, fontsize=8); ax.spines[:].set_visible(False)
        pd.DataFrame(cm, index=C, columns=C).to_csv(R / 'confusion_matrices' / f'{m}_oof_confusion_matrix.csv')
    fig.text(.5, -.01, 'Predicted', ha='center'); fig.text(-.01, .5, 'True', va='center', rotation=90)
    fig.suptitle(f'{DSN}: out-of-fold confusion matrices (counts and row %)'); fig.tight_layout(); save(fig, R / 'confusion_matrices' / 'all_models_oof_confusion_matrices')

    show = C[1:] if binary else C
    fig, axs = plt.subplots(1, len(show), figsize=(min(7.2, 2.9 * len(show) + .4), 2.9), sharey=True, squeeze=False)
    for ax, c in zip(axs[0], show):
        ci = C.index(c)
        for mi, m in enumerate(order):
            P = pd.concat([preds[(m, k)] for k in range(K) if (m, k) in preds])
            fpr, tpr, _ = roc_curve(P.y == ci, P[f'p_{c}'])
            ax.plot(fpr, tpr, lw=.9, color=plt.cm.tab20(mi / max(len(order) - 1, 1)), label=f'{m} ({sk_auc(fpr, tpr):.3f})')
        ax.plot([0, 1], [0, 1], 'k:', lw=.6); ax.set_title(f'{c} vs rest', fontsize=8); ax.set_xlabel('False positive rate'); ax.set_aspect('equal')
        ax.legend(fontsize=4.6, loc='lower right', frameon=False)
    axs[0, 0].set_ylabel('True positive rate'); fig.suptitle(f'{DSN}: out-of-fold ROC curves (AUC in legend)'); fig.tight_layout()
    save(fig, R / 'roc_curves' / 'all_models_oof_roc')
    print(f'== {DSN}'); print(cp.n_complete.to_string()); print(main.to_string())
    if st_rows: print(pd.DataFrame(st_rows)[['metric', 'friedman_chi2', 'p_value', 'kendalls_W']].to_string())
    return num.assign(dataset=DSN)


allnum = [r for r in (analyze(d) for d in (sys.argv[1:] or PUBLISHED)) if r is not None]
if len(sys.argv) == 1 and allnum:
    O = RES / 'overview'; O.mkdir(parents=True, exist_ok=True)
    A = pd.concat(allnum)
    for met in ['f1_macro', 'roc_auc', 'accuracy', 'mcc']:
        t = A.pivot(index='model', columns='dataset', values=f'{met}_mean').reindex(index=list(MODELS), columns=[d for d in PUBLISHED if d in set(A.dataset)])
        sd = A.pivot(index='model', columns='dataset', values=f'{met}_sd').reindex(index=t.index, columns=t.columns)
        f = 1 if met == 'mcc' else 100
        cell = (t * f).round(2).astype(str) + ' ± ' + (sd * f).round(2).astype(str)
        cell.to_csv(O / f'models_x_datasets_{met}.csv'); (O / f'models_x_datasets_{met}.md').write_text(cell.to_markdown(), encoding='utf-8')
    print(pd.read_csv(O / 'models_x_datasets_f1_macro.csv', encoding='utf-8-sig').to_string())
