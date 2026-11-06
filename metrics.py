"""Test-set metrics for one fold (3-class)."""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score, roc_auc_score,
                             balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix)
from common import CLASSES


def fold_metrics(y, p):
    yp = p.argmax(1); K = len(CLASSES)
    cm = confusion_matrix(y, yp, labels=range(K))
    pr, rc, f1, sup = precision_recall_fscore_support(y, yp, labels=range(K), zero_division=0)
    spec = [(cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]) / max(cm.sum() - cm[i].sum(), 1) for i in range(K)]
    present = sorted(set(y))
    if len(present) == K:
        auc, auc_note = float(roc_auc_score(y, p, multi_class='ovr', average='macro', labels=range(K))), ''
        auc_cls = [float(roc_auc_score(y == i, p[:, i])) for i in range(K)]
    else:
        auc, auc_note = float('nan'), f'classes present in test fold: {[CLASSES[i] for i in present]}'
        auc_cls = [float(roc_auc_score(y == i, p[:, i])) if 0 < (y == i).sum() < len(y) else float('nan') for i in range(K)]
    out = dict(accuracy=accuracy_score(y, yp), precision_macro=float(pr.mean()), recall_macro=float(rc.mean()),
               f1_macro=f1_score(y, yp, average='macro', labels=range(K), zero_division=0),
               f1_weighted=f1_score(y, yp, average='weighted', labels=range(K), zero_division=0),
               specificity_macro=float(np.mean(spec)), roc_auc_ovr_macro=auc, auc_note=auc_note,
               balanced_accuracy=balanced_accuracy_score(y, yp), cohen_kappa=cohen_kappa_score(y, yp, labels=range(K)),
               mcc=matthews_corrcoef(y, yp), confusion_matrix=cm.tolist(), n_test=int(len(y)))
    for i, c in enumerate(CLASSES):
        out.update({f'{c}_precision': float(pr[i]), f'{c}_recall': float(rc[i]), f'{c}_f1': float(f1[i]),
                    f'{c}_specificity': float(spec[i]), f'{c}_auc': auc_cls[i], f'{c}_support': int(sup[i])})
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in out.items()}


def binary_metrics(y, p, labels, thr=0.5, yp=None):
    """Cancer (1) vs non-cancer (0). p = P(cancer). labels = original 3-class labels, used for the
    benign / normal subgroup breakdown of the non-cancer class. yp overrides the thresholded decision
    (used for the 3-class model's argmax == malignant)."""
    from sklearn.metrics import average_precision_score
    y = np.asarray(y).astype(int); p = np.asarray(p, float); labels = np.asarray(labels)
    yp = (p >= thr).astype(int) if yp is None else np.asarray(yp).astype(int)
    cm = confusion_matrix(y, yp, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    both = 0 < y.sum() < len(y)
    out = dict(threshold=float(thr), accuracy=accuracy_score(y, yp),
               sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
               precision=tp / (tp + fp) if tp + fp else 0.0, npv=tn / (tn + fn) if tn + fn else 0.0,
               f1=f1_score(y, yp, zero_division=0),
               f1_macro=f1_score(y, yp, average='macro', labels=[0, 1], zero_division=0),
               balanced_accuracy=balanced_accuracy_score(y, yp),
               roc_auc=float(roc_auc_score(y, p)) if both else float('nan'),
               pr_auc=float(average_precision_score(y, p)) if both else float('nan'),
               cohen_kappa=cohen_kappa_score(y, yp, labels=[0, 1]), mcc=matthews_corrcoef(y, yp),
               tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), confusion_matrix=cm.tolist(), n_test=int(len(y)))
    for g in ('benign', 'normal'):  # how often each non-cancer subgroup is flagged as cancer
        m = labels == g
        out[f'{g}_n'] = int(m.sum())
        out[f'{g}_fpr'] = float(yp[m].mean()) if m.any() else float('nan')
        out[f'{g}_mean_p_cancer'] = float(p[m].mean()) if m.any() else float('nan')
    out['benign_vs_malignant_auc'] = float(roc_auc_score(y[labels != 'normal'], p[labels != 'normal'])) \
        if both and (labels == 'benign').any() else float('nan')
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in out.items()}
