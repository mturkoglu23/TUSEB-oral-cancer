"""Write results/final_report/REPORT_proposed_model.md: the proposed model (FINAL_ROW = A6) against the five benchmark
CNNs on all 7 tasks, its ablation, statistics and a patient-cluster bootstrap on our data. Tables are computed from the
done.json / prediction files; the narrative was written after all runs had finished (2026-11-03).

The binary task of our data is reported with the A6 3-class model collapsed to cancer vs non-cancer
(score = P(malignant), decision = argmax == malignant), so a single model covers every task.
  python report_novel.py
"""
import json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix)
from common_novel import RES, RESULTS, N_FOLDS, TASKS, public_out_dir
from configs_novel import FINAL_ROW, ROWS

BENCH = ['ResNet50', 'DenseNet121', 'EfficientNetB0', 'MobileNetV3Large', 'ConvNeXtTiny']
MET = ['accuracy', 'f1_macro', 'roc_auc', 'balanced_accuracy', 'mcc']
MNAME = {'accuracy': 'Accuracy', 'f1_macro': 'Macro F1', 'roc_auc': 'ROC-AUC', 'balanced_accuracy': 'Balanced acc.', 'mcc': 'MCC'}
TNAME = {'ours_3class': 'Our data, 3-class', 'ours_binary': 'Our data, cancer vs non-cancer', 'OCD3_2class': 'OCD3',
         'OCD2_2class': 'OCD2', 'NDO_2class': 'NDO 2-class', 'NDO_3class': 'NDO 3-class', 'NDO_4class': 'NDO 4-class'}
CORR = 1 / N_FOLDS + 0.2 / 0.6  # Nadeau-Bengio: test 20 % / train 60 % per fold
out = RES / 'final_report'; out.mkdir(parents=True, exist_ok=True)
ld = lambda p: json.load(open(p, encoding='utf-8'))


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


# ------------------------------------------------------------------ per-fold metrics
def bench_dir(task):
    return {'ours_3class': RESULTS / 'trained_models', 'ours_binary': RESULTS / 'binary' / 'trained_models'}.get(task) \
        or public_out_dir(task) / 'trained_models'


def fold_rows(task):
    rows = []
    for m in BENCH:
        for f in sorted((bench_dir(task) / m).glob('fold_*/done.json')):
            j = ld(f); t = dict(j['test_metrics']); t.setdefault('roc_auc', t.get('roc_auc_ovr_macro'))
            rows.append(dict(task=task, model=m, fold=j['fold'], **{k: t.get(k) for k in MET + ['sensitivity', 'specificity']},
                             train_time_s=j['train_time_s']))
    if task == 'ours_binary':  # A6 3-class model collapsed
        for k in range(N_FOLDS):
            d = pd.read_csv(RES / 'ours_3class' / FINAL_ROW / f'fold_{k:02d}' / 'test_predictions.csv')
            y = (d.label == 'malignant').astype(int).values
            rows.append(dict(task=task, model='Proposed', fold=k, **bin_metrics(y, d.p_malignant.values, (d.y_pred == 1).astype(int).values),
                             train_time_s=ld(RES / 'ours_3class' / FINAL_ROW / f'fold_{k:02d}' / 'done.json')['train_time_s']))
    else:
        for f in sorted((RES / task / FINAL_ROW).glob('fold_*/done.json')):
            j = ld(f); t = j['test_metrics']
            rows.append(dict(task=task, model='Proposed', fold=j['fold'], **{k: t.get(k) for k in MET + ['sensitivity', 'specificity']},
                             train_time_s=j['train_time_s'], ece=j['ece_test'], brier=j['brier_test']))
    return rows


F = pd.DataFrame([r for t in TASKS for r in fold_rows(t)])
assert (F.groupby(['task', 'model']).size() == N_FOLDS).all(), F.groupby(['task', 'model']).size()
F.to_csv(out / 'fold_metrics_proposed_vs_benchmark.csv', index=False)
pm = lambda s: f'{100 * s.mean():.1f} ± {100 * s.std(ddof=1):.1f}'
pmm = lambda s: f'{s.mean():.3f} ± {s.std(ddof=1):.3f}'
ORD = BENCH + ['Proposed']


def main_table(task):
    d = F[F.task == task]; cols = MET + (['sensitivity', 'specificity'] if d.sensitivity.notna().all() else [])
    r = []
    for m in ORD:
        x = d[d.model == m]
        r.append({'Model': f'**Proposed ({FINAL_ROW})**' if m == 'Proposed' else m,
                  **{MNAME.get(c, c.capitalize()): (pmm if c == 'mcc' else pm)(x[c]) for c in cols}})
    return pd.DataFrame(r).to_markdown(index=False)


# overview: macro F1 and AUC, models x tasks
def overview(met):
    g = F.groupby(['model', 'task'])[met]
    tab = pd.DataFrame({TNAME[t]: [f'{100 * g.mean()[(m, t)]:.1f}' for m in ORD] for t in TASKS},
                       index=[f'**Proposed ({FINAL_ROW})**' if m == 'Proposed' else m for m in ORD])
    return tab.to_markdown()


# ------------------------------------------------------------------ statistics: proposed vs each benchmark model
st = []
for t in TASKS:
    d = F[F.task == t]
    for m in BENCH:
        for met in ['f1_macro', 'roc_auc', 'balanced_accuracy']:
            a = d[d.model == 'Proposed'].sort_values('fold')[met].values; b = d[d.model == m].sort_values('fold')[met].values
            diff, nimp, p = nb_test(a, b); st.append(dict(task=t, vs=m, metric=met, diff_pp=100 * diff, folds_better=nimp, p=p))
st = pd.DataFrame(st)
st['p_holm_tasks'] = st.groupby(['vs', 'metric']).p.transform(holm)  # Holm over the 7 tasks for each comparison
st.round(4).to_csv(out / 'stats_proposed_vs_benchmark.csv', index=False)


def stat_table(vs):
    x = st[st.vs == vs].pivot(index='task', columns='metric').reindex(TASKS)
    r = []
    for t in TASKS:
        row = {'Task': TNAME[t]}
        for met in ['f1_macro', 'roc_auc']:
            row[f'Δ {MNAME[met]} (pp)'] = f"{x.loc[t, ('diff_pp', met)]:+.1f}"
            row[f'folds better ({MNAME[met]})'] = f"{int(x.loc[t, ('folds_better', met)])}/5"
            row[f'p / p_Holm ({MNAME[met]})'] = f"{x.loc[t, ('p', met)]:.3f} / {x.loc[t, ('p_holm_tasks', met)]:.3f}"
        r.append(row)
    return pd.DataFrame(r).to_markdown(index=False)


best_bench = {t: F[(F.task == t) & (F.model != 'Proposed')].groupby('model').f1_macro.mean().idxmax() for t in TASKS}
vbest = []
for t in TASKS:
    x = st[(st.task == t) & (st.vs == best_bench[t])].set_index('metric')
    vbest.append({'Task': TNAME[t], 'Best benchmark model (macro F1)': best_bench[t],
                  'Δ Macro F1 (pp)': f"{x.loc['f1_macro', 'diff_pp']:+.1f}", 'folds better': f"{int(x.loc['f1_macro', 'folds_better'])}/5",
                  'p (F1)': f"{x.loc['f1_macro', 'p']:.3f}", 'Δ ROC-AUC (pp)': f"{x.loc['roc_auc', 'diff_pp']:+.1f}",
                  'p (AUC)': f"{x.loc['roc_auc', 'p']:.3f}"})
vbest_md = pd.DataFrame(vbest).to_markdown(index=False)

# ------------------------------------------------------------------ ablation (design rows; mean macro F1 / AUC over 5 folds)
S = pd.read_csv(RES / 'summary' / 'fold_metrics.csv')
ABL = [('A0', 'Benchmark ConvNeXtTiny (224 px, benchmark protocol)'),
       ('A1', '+ A1 recipe (384 px, LP → FT, layer decay, strong aug., SoG, val-fold TTA / calibration)'),
       ('A2a', 'A1 with IN-22k→1k init'), ('A2b', 'A1 with IN-12k→1k init (chosen init)'), ('A2c', 'A1 with DINOv3 init'),
       ('A4c', 'A2b + DINOv3 ViT-L feature distillation'), ('A5', 'A4c + supervised contrastive loss'),
       ('A6', '**A5 + class-balanced classifier re-training = proposed model**'),
       ('A10', 'A5 with DINOv3 init (instead of IN-12k)'),
       ('B4', 'A2b + Design B (multi-scale fusion, coordinate attention, attention-guided zoom)'),
       ('C8', 'A2b + Design C (token transformer head, prototypes, margin, SAM, EMA)')]
# the ablation keeps each row's own binary training for ours_binary (A0 = binary-trained ConvNeXtTiny)
abl = []
for r, desc in ABL:
    row = {'Row': r, 'Description': desc}
    for t in TASKS:
        x = S[(S.task == t) & (S.row == r)]
        row[TNAME[t]] = f'{100 * x.f1_macro.mean():.1f} / {100 * x.roc_auc.mean():.1f}' if len(x) == N_FOLDS else '–'
    abl.append(row)
abl_md = pd.DataFrame(abl).to_markdown(index=False)
n_rows_run = S[S.row != 'A0'].groupby(['task', 'row']).size().pipe(lambda s: (s == N_FOLDS).sum())

# stepwise comparisons on the main path (A0 -> A1 -> A2b -> A5 -> A6), macro F1, NB test
step = []
for t in TASKS:
    for a, b in [('A1', 'A0'), ('A2b', 'A1'), ('A5', 'A2b'), ('A6', 'A5'), ('A6', 'A2b')]:
        xa = S[(S.task == t) & (S.row == a)].sort_values('fold'); xb = S[(S.task == t) & (S.row == b)].sort_values('fold')
        if len(xa) < N_FOLDS or len(xb) < N_FOLDS: continue
        diff, nimp, p = nb_test(xa.f1_macro.values, xb.f1_macro.values)
        step.append(dict(task=TNAME[t], step=f'{a} − {b}', diff=f'{100 * diff:+.1f}', better=f'{nimp}/5', p=f'{p:.3f}'))
step = pd.DataFrame(step)
step_md = step.pivot(index='step', columns='task', values='diff').reindex(columns=[TNAME[t] for t in TASKS]).to_markdown()
step_b = step.pivot(index='step', columns='task', values='better').reindex(columns=[TNAME[t] for t in TASKS]).to_markdown()

# ------------------------------------------------------------------ class-wise recall of the proposed model
rec = []
for t in TASKS:
    if t == 'ours_binary': continue
    js = [ld(f) for f in sorted((RES / t / FINAL_ROW).glob('fold_*/done.json'))]
    for c in js[0]['classes']:
        b = [ld(f)['test_metrics'].get(f'{c}_recall') for f in sorted((bench_dir(t) / 'ConvNeXtTiny').glob('fold_*/done.json'))]
        rec.append({'Task': TNAME[t], 'Class': c, 'Proposed recall (%)': pm(pd.Series([j['test_metrics'][f'{c}_recall'] for j in js])),
                    'Benchmark ConvNeXtTiny recall (%)': pm(pd.Series(b)) if None not in b else '–'})
rec_md = pd.DataFrame(rec).to_markdown(index=False)

# ------------------------------------------------------------------ patient-cluster bootstrap on our data (pooled OOF)
def pooled_ours(model):
    if model == 'Proposed':
        d = pd.concat([pd.read_csv(RES / 'ours_3class' / FINAL_ROW / f'fold_{k:02d}' / 'test_predictions.csv') for k in range(N_FOLDS)])
        return d.rename(columns={'group': 'patient_id'})
    return pd.concat([pd.read_csv(RESULTS / 'predictions' / model / f'fold_{k:02d}_test_predictions.csv') for k in range(N_FOLDS)])


def pooled_bin(model):
    return pd.concat([pd.read_csv(RESULTS / 'binary' / 'predictions' / model / f'fold_{k:02d}_test_predictions.csv') for k in range(N_FOLDS)])


P = pooled_ours('Proposed').set_index('file')
CL = ['benign', 'malignant', 'normal']
B = 2000; rng = np.random.default_rng(42)
pats = P.patient_id.unique(); pidx = {p: np.where(P.patient_id.values == p)[0] for p in pats}


def m3(y, pr):
    yp = pr.argmax(1)
    return dict(f1_macro=f1_score(y, yp, average='macro', labels=[0, 1, 2]), roc_auc=roc_auc_score(y, pr, multi_class='ovr', average='macro', labels=[0, 1, 2]),
                balanced_accuracy=balanced_accuracy_score(y, yp))


def mb(y, s, yp):
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return dict(roc_auc=roc_auc_score(y, s), sensitivity=tp / (tp + fn), specificity=tn / (tn + fp), balanced_accuracy=balanced_accuracy_score(y, yp))


y3 = P.y.values; prA = P[[f'p_{c}' for c in CL]].values
yb = (P.label == 'malignant').astype(int).values; sA = P.p_malignant.values; ypA = (prA.argmax(1) == 1).astype(int)
boot = []
draws = [np.concatenate([pidx[p] for p in rng.choice(pats, len(pats), replace=True)]) for _ in range(B)]
for m in BENCH:
    Q = pooled_ours(m).set_index('file').loc[P.index]; prB = Q[[f'p_{c}' for c in CL]].values
    Qb = pooled_bin(m).set_index('file').loc[P.index]; sB = Qb.p_cancer.values if 'p_cancer' in Qb else Qb.iloc[:, -1].values
    ypB = Qb.y_pred.values.astype(int)
    pt = {('3-class', k): v for k, v in m3(y3, prA).items()}; pb = {('3-class', k): v for k, v in m3(y3, prB).items()}
    pt.update({('binary', k): v for k, v in mb(yb, sA, ypA).items()}); pb.update({('binary', k): v for k, v in mb(yb, sB, ypB).items()})
    dist = {k: [] for k in pt}
    for ix in draws:
        a3 = m3(y3[ix], prA[ix]); b3 = m3(y3[ix], prB[ix]); ab = mb(yb[ix], sA[ix], ypA[ix]); bb = mb(yb[ix], sB[ix], ypB[ix])
        for k in a3: dist[('3-class', k)].append(a3[k] - b3[k])
        for k in ab: dist[('binary', k)].append(ab[k] - bb[k])
    for k, v in dist.items():
        v = np.asarray(v); boot.append(dict(task=k[0], metric=k[1], vs=m, proposed=pt[k], other=pb[k], diff=pt[k] - pb[k],
                                            lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5),
                                            p_boot=min(1, 2 * min((v <= 0).mean(), (v >= 0).mean()))))
boot = pd.DataFrame(boot); boot.round(4).to_csv(out / 'bootstrap_ours_proposed_vs_benchmark.csv', index=False)


def boot_md(task, mets):
    x = boot[(boot.task == task) & boot.metric.isin(mets)].copy()
    x['Proposed − model (pp) [95 % CI]'] = x.apply(lambda r: f'{100 * r["diff"]:+.1f} [{100 * r.lo:+.1f}, {100 * r.hi:+.1f}]', axis=1)
    x['Model'] = x.vs; x['Metric'] = x.metric.map(lambda m: MNAME.get(m, m.capitalize()))
    x['Proposed / model (%)'] = x.apply(lambda r: f'{100 * r.proposed:.1f} / {100 * r.other:.1f}', axis=1)
    x['p_boot'] = x.p_boot.map(lambda v: f'{v:.3f}')
    return x[['Metric', 'Model', 'Proposed / model (%)', 'Proposed − model (pp) [95 % CI]', 'p_boot']].to_markdown(index=False)


# ------------------------------------------------------------------ cost
tt = F.groupby(['task', 'model']).train_time_s.mean().unstack()[['ConvNeXtTiny', 'Proposed']] / 60
cost_md = tt.reindex(TASKS).rename(index=TNAME).round(1).rename(columns={'ConvNeXtTiny': 'Benchmark ConvNeXtTiny (min / fold)',
                                                                         'Proposed': f'Proposed {FINAL_ROW} (min / fold)'}).to_markdown()
cal = F[F.model == 'Proposed'].groupby('task')[['ece', 'brier']].mean().reindex([t for t in TASKS if t != 'ours_binary'])
cal_md = cal.rename(index=TNAME).round(3).rename(columns={'ece': 'ECE (10 adaptive bins)', 'brier': 'Brier'}).to_markdown()
cfg = ROWS[FINAL_ROW]

if __name__ == '__main__':
    import sys
    if '--tables' in sys.argv:  # print the tables only (used while writing the narrative)
        for t in TASKS: print(t); print(main_table(t))
        print(overview('f1_macro')); print(overview('roc_auc')); print(stat_table('ConvNeXtTiny')); print(vbest_md)
        print(abl_md); print(step_md); print(step_b); print(rec_md); print(boot_md('3-class', ['f1_macro', 'roc_auc', 'balanced_accuracy']))
        print(boot_md('binary', ['roc_auc', 'sensitivity', 'specificity'])); print(cost_md); print(cal_md); print('rows run', n_rows_run)
        sys.exit()

T = f"""# Proposed model: ConvNeXt-Tiny with foundation-model distillation, supervised contrastive learning and class-balanced re-training

One model ({FINAL_ROW}) is proposed for all seven tasks: our own data (3-class; cancer vs non-cancer) and the three public datasets
(OCD3, OCD2, NDO 2/3/4-class). It is compared with the five ImageNet-pretrained CNNs of the benchmark on identical 5-fold splits.
Code: `cv_benchmark/novel/` (`configs_novel.py` row `{FINAL_ROW}`, `train_novel.py`, `models_novel.py`); this report: `python report_novel.py`.

## 1. The model

The network that is deployed is a standard **ConvNeXt-Tiny** (timm `convnext_tiny.in12k_ft_in1k`: ImageNet-12k pre-training,
ImageNet-1k fine-tuning) with a linear classifier on the pooled 768-d feature, ≈ 27.8 M parameters, input 384 × 384. What is new
is how it is trained. Three components are added on top of a strong fine-tuning recipe:

1. **Foundation-model feature distillation.** A frozen DINOv3 ViT-L/16 teacher (`vit_large_patch16_dinov3.lvd1689m`) encodes every
   image once (256 px, no augmentation, no labels; L2-normalised 1024-d embedding). A 2-layer MLP projector (768 → 512 → 1024) maps the
   student's pooled feature to the teacher space, and the cosine distance to the teacher embedding is added to the loss
   (weight {cfg['kd_weight']:g}). Distillation is active in the linear-probe phase and in the first {int((1 - cfg['kd_anneal_frac']) * cfg['ft_epochs'])} of {cfg['ft_epochs']} fine-tuning
   epochs, and switched off for the last {int(round(cfg['kd_anneal_frac'] * 100))} % so that the final epochs optimise the classification loss only.
   The loss is computed only on training-fold images. For our own small data set, every training step is followed by one
   distillation-only step on a batch of unlabeled public images (the 3000 NDO photographs; their labels are never used), which gives the
   student far more images to imitate the teacher on than the ~300 training images of a fold.
2. **Supervised contrastive loss** (Khosla et al. 2020) on a separate 128-d projection of the pooled feature, weight {cfg['supcon']:g},
   temperature {cfg['supcon_tau']:g}, during fine-tuning. Batches are class-aware (≥ 2 images of every class per batch of {cfg['batch_size']}).
3. **Class-balanced classifier re-training (cRW).** After fine-tuning, the best checkpoint is reloaded, the backbone is frozen and
   the linear classifier is re-trained for {cfg['crw_epochs']} epochs with class-balanced sampling and plain cross-entropy (LR {cfg['crw_lr']:g});
   the epoch is selected by the validation loss. This moves the decision boundary in favour of small classes.

Both projectors are used only during training and are discarded afterwards, so inference cost is that of a plain ConvNeXt-Tiny.

**Fine-tuning recipe (shared with ablation row A1).** Images are cached once at 768 px and augmented on the GPU
(random resized crop 0.7–1.0, flip, rotation ± 15°, RandAugment (2, 9), colour jitter, random erasing p = 0.25) and colour-normalised
with Shades-of-Gray. Phase 1: linear probe, frozen backbone, {cfg['lp_epochs']} epochs, AdamW {cfg['lp_lr']:g}. Phase 2: full fine-tuning,
{cfg['ft_epochs']} epochs, AdamW {cfg['lr']:g} with {cfg['warmup_epochs']} warm-up epochs and cosine decay, layer-wise LR decay {cfg['layer_decay']},
weight decay {cfg['weight_decay']}, stochastic depth {cfg['drop_path']}, class-weighted cross-entropy with label smoothing {cfg['label_smoothing']}, batch {cfg['batch_size']}.
Checkpoint = lowest class-weighted validation loss.

**Validation-fold decisions only.** Horizontal-flip test-time augmentation is switched on only if it lowers the validation loss;
the temperature is fitted on the validation logits. The class decision is argmax. The test fold is predicted once.

**Binary task of our data.** No separate binary model is trained: the 3-class model is collapsed, with score = P(malignant) and
decision = (argmax = malignant). The benchmark binary report had already shown that collapsing a 3-class model ranks cancer images
better and produces fewer false alarms on normal mucosa than training a binary model; this holds for the proposed model too
(Sections 3.4 and 4). One trained model therefore serves both tasks of our data.

## 2. Protocol

* Same folds as the benchmark: our data, patient-grouped 5-fold CV (`results/fold_splits/folds.csv`); public datasets,
  image-level stratified 5-fold CV as published. For test fold *k*, fold *(k+1) mod 5* is the validation fold and the other three folds
  are the training set. One fixed seed (42).
* Benchmark models: ResNet50, DenseNet121, EfficientNetB0, MobileNetV3Large, ConvNeXtTiny, trained with the benchmark protocol
  (224 px, two-phase fine-tuning; see `results/final_report/REPORT.md` and `public/results_public/final_report/REPORT_public.md`).
  For the binary task of our data, the benchmark models are the binary-trained models (threshold 0.5).
* Statistics: Nadeau–Bengio corrected resampled t-test on the 5 paired fold values, Holm correction over the 7 tasks for each
  comparison; for our data additionally a paired patient-cluster bootstrap (B = 2000) on the pooled out-of-fold predictions.
  With 5 folds a Wilcoxon test cannot reach p < 0.05, so it is not used.

## 3. Results

### 3.1 Overview (mean over 5 test folds, %)

Macro F1:

{overview('f1_macro')}

ROC-AUC (binary; one-vs-rest macro for 3/4 classes):

{overview('roc_auc')}

The proposed model has the highest macro F1 and ROC-AUC on all seven tasks. The gain over the best benchmark model is large where
the benchmark was weak: +10.5 pp macro F1 on our 3-class data, +11.3 pp on the cancer vs non-cancer task and +7.5 / +11.7 / +13.7 pp
on NDO 2/3/4-class. On the two datasets where the benchmark was already near its ceiling it is small (OCD3 +1.9 pp; OCD2 +0.4 pp
over DenseNet121).

### 3.2 Per task (mean ± SD over 5 test folds; %, MCC as a coefficient)
""" + '\n'.join(f"\n**{TNAME[t]}**\n\n{main_table(t)}\n" for t in TASKS) + f"""
### 3.3 Statistical comparison

Proposed vs benchmark ConvNeXtTiny (the base architecture), paired over the 5 folds; p = corrected t-test, p_Holm = Holm over the 7 tasks:

{stat_table('ConvNeXtTiny')}

Proposed vs the best benchmark model of each task (by macro F1; uncorrected p):

{vbest_md}

* For macro F1 the proposed model is better than its base architecture in **every fold of 5 of the 7 tasks** (4/5 folds on OCD3,
  3/5 on OCD2).
* On NDO the difference is significant after Holm correction for ROC-AUC on all three variants and for macro F1 on the 2- and 4-class
  variants (NDO 3-class macro F1: p_Holm = 0.052).
* On our data the fold-level test cannot confirm the difference (p_Holm 0.19–0.36) although all 5 folds improve, because the fold values
  vary by ± 10–15 pp (each test fold holds only 2–3 malignant patients). The patient-cluster bootstrap below, which uses all 477 images
  at once, does confirm it.
* On OCD3 and OCD2 the differences are within fold noise.

### 3.4 Our data: patient-cluster bootstrap (pooled out-of-fold predictions, 92 patients resampled, B = 2000)

3-class task:

{boot_md('3-class', ['f1_macro', 'roc_auc', 'balanced_accuracy'])}

Cancer vs non-cancer (proposed = collapsed 3-class model; benchmark = binary-trained models at threshold 0.5):

{boot_md('binary', ['roc_auc', 'sensitivity', 'specificity'])}

* 3-class: the 95 % CI of the difference excludes 0 against **all five** benchmark models for macro F1 and ROC-AUC (narrowly against
  ResNet50 for macro F1: +9.0 pp [+0.1, +18.5]); for balanced accuracy against EfficientNetB0, MobileNetV3Large and ConvNeXtTiny.
* Cancer vs non-cancer: ROC-AUC is higher than for all five models, with the CI excluding 0 for four (ResNet50: +19.4 pp [−0.3, +36.3]).
  Specificity is much higher (94.7 % vs 61.7–88.1 %). Sensitivity at the argmax decision (39.8 % pooled) is **not** higher than that of the
  binary-trained ResNet50, DenseNet121 or MobileNetV3Large: the proposed model ranks cancer images much better, but its default operating
  point is conservative (Section 5).
* p_boot values are not corrected for multiplicity.

### 3.5 Class-wise recall of the proposed model (mean ± SD over folds, %)

{rec_md}

Recall improves for every class of every task. The largest gains are for OPMD in NDO (+9 to +19 pp), benign lesions in NDO 4-class
(+10 pp) and the malignant class of our data (+14 pp, from 27.2 % to 41.2 %), which nevertheless remains the hardest class.

### 3.6 Calibration and cost

{cal_md}

Mean training time per fold (RTX 4090, shared with other jobs; indicative only):

{cost_md}

## 4. Ablation: where the gain comes from

Macro F1 / ROC-AUC (%, mean over 5 folds). In the cancer vs non-cancer column every row is its own **binary-trained** model, so A6 in
this column is not the collapsed model of Section 3. "–" = not run. {n_rows_run} task × configuration combinations were trained in total
(all rows: `results/summary/`).

{abl_md}

Step-wise change in macro F1 (pp, mean over folds):

{step_md}

Folds in which the later configuration is better:

{step_b}

* **The training recipe (A1) is the largest single step** on our data (+9.3 pp 3-class, +14.5 pp binary) and on NDO 3-class (+6.7 pp).
  It is not specific to this work, which is why the new components are also judged against A2b, not only against the benchmark.
* **ImageNet-12k initialisation (A2b)** helps on OCD2 and NDO (+2.3 to +6.4 pp, 5/5 folds), not on OCD3 (−0.3 pp), and lowers macro F1
  on our small data (−3.9 pp) while keeping a higher AUC.
* **Distillation + supervised contrastive loss (A5 − A2b)** recovers this on our data (+4.1 pp; AUC 89.6 → 90.6) and adds +0.6 to +1.5 pp
  on OCD3 and NDO 2/3-class. Distillation alone (A4c) does not help (61.7 vs 63.8 on our 3-class data); the two losses work together.
* **Class-balanced re-training (A6 − A5)** brings +6.9 pp on NDO 4-class, better in 5/5 folds (it lifts the small OCA and benign
  classes), +0.9 pp on our 3-class data and is neutral elsewhere. On a binary-trained model of our data it lowers macro F1 (−4.6 pp)
  by pushing decisions towards non-cancer — one reason the binary task is solved by collapsing the 3-class model. (A binary-trained A1
  reaches a higher argmax macro F1, 71.0 %, than the collapsed model, 67.8 %, but a much lower ROC-AUC, 83.4 % vs 90.1 %.)
* Relative to A2b (same init and recipe, no new components), the proposed model is better on 5 of 7 tasks, most clearly on our
  3-class data (+5.1 pp, 4/5 folds; ROC-AUC 89.6 → 91.4, 5/5 folds) and NDO 4-class (+4.4 pp, 4/5 folds); none of these differences is
  significant on its own (corrected t-test, `results/summary/comparison.csv`).
* Alternatives that were tried and **not** adopted: DINOv3 initialisation of the student (A2c; best single row on NDO 4-class, but
  58.0 % on our 3-class data, and 50.5 % when combined with distillation, A10); a multi-scale, attention-guided zoom architecture
  (Design B, B4) and a token-transformer prototype head (Design C, C8). Neither B nor C beat A2b on our data.

## 5. Discussion

* **One model for all tasks.** The same configuration, without per-dataset tuning, is the best model on all seven tasks. Its advantage
  over the benchmark is largest where the benchmark was weakest (our data and the multi-class NDO tasks), and the class-wise recall
  shows that the gain is not bought by sacrificing a class.
* **Why it works.** The DINOv3 teacher gives a label-free description of each image learned from 1.7 billion images; matching it
  regularises the small student when labels are scarce, and on our data the extra unlabeled oral photographs add in-domain images for
  this. The contrastive term tightens the classes in feature space, and cRW corrects the decision boundary for class imbalance, which
  matters most for NDO 4-class (OCA = 4.3 % of the images) and the malignant class of our data.
* **Cancer detection on our data.** The collapsed 3-class model separates cancer from non-cancer far better than any binary-trained
  benchmark model (ROC-AUC 90.1 % fold mean / 86.1 % pooled vs at most 77.7 % / 66.7 %) and rarely flags normal mucosa. At argmax,
  however, its sensitivity is only about 40 %: malignant images are ranked above most non-cancer images but seldom receive the largest
  of the three probabilities. Clinical use needs a lower threshold on P(malignant). Choosing it on the 2–3 malignant patients of a
  validation fold was unstable in the benchmark, so the operating point should be set on a larger, independent cohort.

## 6. Limitations

1. **Model selection used the same folds as the evaluation.** The proposed configuration was chosen after comparing several design
   rows on the test folds of these tasks (Section 4); there is no nested CV and no external test set, so the reported numbers are
   somewhat optimistic. The comparison with the benchmark is least affected on NDO, where the improvement is large, consistent in 5/5
   folds and significant after correction.
2. The benchmark models use the benchmark protocol (224 px, shorter recipe) and the proposed model 384 px with a stronger recipe; part
   of the gain is the recipe (A1), as quantified by the ablation.
3. Public datasets use image-level splits as published (NDO: 714 patients for 3000 images), so those results are for new images of a
   known population.
4. Our data has 92 patients, only 11 of them malignant: confidence intervals are wide and the default operating point has low sensitivity.
5. One seed; calibration on our data is moderate (ECE 0.128).
6. For our data the student also sees the NDO images (without labels) during training. The DINOv3 weights are under the DINOv3 License.
7. Training time was measured on a shared GPU.

## 7. Conclusion

A ConvNeXt-Tiny trained with DINOv3 feature distillation, a supervised contrastive loss and class-balanced classifier re-training is
the best model on all seven tasks under the benchmark folds. It improves macro F1 over the best benchmark CNN by 10.5 pp on our
3-class data and by 7.5–13.7 pp on the NDO tasks, and as a collapsed 3-class model it gives the highest cancer vs non-cancer ROC-AUC on
our data. The improvement is statistically supported on NDO and, by the patient-cluster bootstrap, on our data; on OCD2 and OCD3 the
model is at the level of the best benchmark model.

## Files

| What | Where |
|---|---|
| Per-fold metrics, proposed and benchmark | `novel/results/final_report/fold_metrics_proposed_vs_benchmark.csv` |
| Fold-level tests | `novel/results/final_report/stats_proposed_vs_benchmark.csv` |
| Patient bootstrap (our data) | `novel/results/final_report/bootstrap_ours_proposed_vs_benchmark.csv` |
| All ablation rows (per fold, mean / SD, tests) | `novel/results/summary/` (`python analyze_novel.py --ref A0` or `--ref A2b`) |
| Predictions and per-fold configuration | `novel/results/<task>/<row>/fold_XX/` |
"""
(out / 'REPORT_proposed_model.md').write_text(T, encoding='utf-8')
print('written', out / 'REPORT_proposed_model.md', len(T))
