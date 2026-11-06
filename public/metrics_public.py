"""Test-set metrics for one fold, for any class list. For 2 classes, index 1 is the positive (disease) class."""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score, roc_auc_score, average_precision_score,
                             balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix)


def fold_metrics(y, p, classes):
    y = np.asarray(y).astype(int); p = np.asarray(p, float); K = len(classes); yp = p.argmax(1)
    cm = confusion_matrix(y, yp, labels=range(K))
    pr, rc, f1, sup = precision_recall_fscore_support(y, yp, labels=range(K), zero_division=0)
    spec = [(cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]) / max(cm.sum() - cm[i].sum(), 1) for i in range(K)]
    present = set(y.tolist())
    auc_cls = [float(roc_auc_score(y == i, p[:, i])) if 0 < (y == i).sum() < len(y) else float('nan') for i in range(K)]
    if K == 2:
        auc = auc_cls[1]
    else:
        auc = float(roc_auc_score(y, p, multi_class='ovr', average='macro', labels=range(K))) if len(present) == K else float('nan')
    out = dict(accuracy=accuracy_score(y, yp), precision_macro=float(pr.mean()), recall_macro=float(rc.mean()),
               f1_macro=f1_score(y, yp, average='macro', labels=range(K), zero_division=0),
               f1_weighted=f1_score(y, yp, average='weighted', labels=range(K), zero_division=0),
               specificity_macro=float(np.mean(spec)), roc_auc=auc,
               balanced_accuracy=balanced_accuracy_score(y, yp), cohen_kappa=cohen_kappa_score(y, yp, labels=range(K)),
               mcc=matthews_corrcoef(y, yp), confusion_matrix=cm.tolist(), n_test=int(len(y)))
    if K == 2:  # positive class = index 1
        tn, fp, fn, tp = cm.ravel()
        out.update(sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                   precision=tp / (tp + fp) if tp + fp else 0.0, npv=tn / (tn + fn) if tn + fn else 0.0,
                   f1=float(f1[1]), pr_auc=float(average_precision_score(y, p[:, 1])) if 0 < y.sum() < len(y) else float('nan'))
    for i, c in enumerate(classes):
        out.update({f'{c}_precision': float(pr[i]), f'{c}_recall': float(rc[i]), f'{c}_f1': float(f1[i]),
                    f'{c}_specificity': float(spec[i]), f'{c}_auc': auc_cls[i], f'{c}_support': int(sup[i])})
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in out.items()}
