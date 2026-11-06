"""Write results/final_report/REPORT.md. Tables are read from the saved result files; the narrative
was written after inspecting the completed run (5 models x 5 folds, seed 42, finished 2026-11-02)."""
import json
import pandas as pd
from common import *

R = RESULTS; out = R / 'final_report'; out.mkdir(exist_ok=True)
def md(df, **k): return df.to_markdown(**k)
def rd(p, **k): return pd.read_csv(p, encoding='utf-8-sig', **k)

audit = json.load(open(R / 'data_audit' / 'audit_report.json'))
cls = rd(R / 'data_audit' / 'class_distribution.csv', index_col=0)
sub = rd(R / 'data_audit' / 'subtype_distribution.csv')
split = rd(R / 'fold_splits' / 'split_summary.csv')
main = rd(R / 'tables' / 'main_table.csv', index_col=0)
full = rd(R / 'tables' / 'summary_mean_sd_all_metrics.csv', index_col=0)
num = rd(R / 'tables' / 'summary_numeric.csv', index_col=0)
comp = rd(R / 'tables' / 'completeness.csv', index_col=0)
fm = rd(R / 'fold_metrics' / 'all_models_fold_metrics.csv')
oof = rd(R / 'tables' / 'pooled_oof_metrics.csv', index_col=0)
pat = rd(R / 'tables' / 'patient_level_oof_metrics.csv', index_col=0)
fr = rd(R / 'statistical_tests' / 'friedman_tests.csv')
pw = rd(R / 'statistical_tests' / 'pairwise_tests.csv')
order = list(main.index)

te = split[split.role == 'test'][['test_fold', 'val_fold', 'images', 'patients'] + [f'{c}_{t}' for c in CLASSES for t in ('img', 'pat')]]
pc = (fm.groupby('model')[[f'{c}_{s}' for c in CLASSES for s in ('precision', 'recall', 'f1')]].mean() * 100).round(1).loc[order]
pc.columns = [c.replace('_', ' ') for c in pc.columns]
eff = num[['params_M', 'latency_ms', 'train_time_min']].round(2).loc[order]
eff.columns = ['Parameters (M)', 'Latency, batch 1 (ms, shared GPU)', 'Training time per fold (min)']
eff.insert(0, 'Input size', [MODELS[m][1] for m in order])
eff.insert(0, 'timm weights', [MODELS[m][0] for m in order])
sel = fm.groupby('model').agg(mean_epochs_run=('epochs_run', 'mean'), mean_selected_epoch=('best_epoch', 'mean'),
                              folds_selected_in_head_phase=('best_phase', lambda x: int((x == 'head').sum()))).round(1).loc[order]
oofc = oof[['n', 'accuracy', 'f1_macro', 'roc_auc_ovr_macro', 'mcc', 'malignant_recall', 'malignant_precision', 'malignant_f1']].copy()
oofc.iloc[:, 1:] = (oofc.iloc[:, 1:] * 100).round(1); oofc['mcc'] = (oof['mcc']).round(3)
patc = pat[['n_patients', 'accuracy', 'f1_macro', 'roc_auc_ovr_macro', 'mcc', 'benign_recall', 'malignant_recall', 'normal_recall']].copy()
patc.iloc[:, 1:] = (patc.iloc[:, 1:] * 100).round(1); patc['mcc'] = pat['mcc'].round(3)
frc = fr[['metric', 'n_folds', 'n_models', 'friedman_chi2', 'p_value', 'kendalls_W']].copy()
frc['friedman_chi2'] = frc.friedman_chi2.round(2); frc['p_value'] = frc.p_value.map(lambda p: f'{p:.2e}'); frc['kendalls_W'] = frc.kendalls_W.round(3)
ranks = fr[['metric', 'mean_ranks']]
def vsbest(metric):
    x = pw[(pw.metric == metric) & pw.vs_best].copy()
    x['comparison'] = x.model_a + ' vs ' + x.model_b
    x = x[['comparison', 'mean_diff', 'wilcoxon_p', 'wilcoxon_p_holm_vs_best', 'rank_biserial', 'corrected_t_p', 'corrected_t_p_holm_vs_best']]
    x['mean_diff'] = (x.mean_diff * 100).round(2)
    for c in x.columns[2:]: x[c] = x[c].round(4)
    return x.rename(columns={'mean_diff': 'mean diff (pp)'})
n_sig_all = {m: int((pw[pw.metric == m].wilcoxon_p_holm_all_pairs < .05).sum()) for m in fr.metric}
n_pairs = int((pw.metric == 'f1_macro').sum())
bs_md = (R / 'bootstrap_ci' / 'BOOTSTRAP_CI_3class.md')
bs_md = bs_md.read_text(encoding='utf-8').replace('### ', '#### ') if bs_md.exists() else '_not computed yet: run `python 07_bootstrap_ci.py`_'

T = f"""# Patient-grouped 5-fold comparison of ImageNet-pretrained CNNs for intra-oral photograph classification (benign / malignant / normal)

All numbers below come from completed runs. Completeness: **{int((comp.n_complete == N_FOLDS).sum())}/{len(MODELS)} models with all {N_FOLDS} folds**
({int(comp.n_complete.sum())}/{len(MODELS) * N_FOLDS} model-fold runs; no failed runs). Every metric is computed on the held-out test fold only.
Source files: `results/` (paths given per section). Code: `cv_benchmark/*.py`.

## 1. Dataset Description

*our_data_clean* (TÜSEB oral cancer project): anonymised clinical intra-oral photographs, JPEG, EXIF removed,
organised as `<label>/<patient_id>/`. {audit['manifest_rows']} images from {int(cls.patients.sum())} patients, 3 classes.

{md(cls)}

Diagnostic sub-types (from source folder names; *belirtilmemis* = not specified):

{md(sub, index=False)}

Image sizes vary widely (see `data_audit/image_size_summary.csv`). Malignant comprises 9 SCC patients, 1 malignant melanoma patient
(6 images) and 1 unspecified malignant collection. Benign lesions are histopathology folders of the clinic.

## 2. Data Leakage Prevention

* **Patient grouping.** During curation, images were grouped per patient from patient folders; folderless images (mainly *normal*)
  were grouped by EXIF capture session (≤180 s) and image similarity; same-patient groups found visually were merged
  (over-merging is safe for leakage, under-merging is not). Exact duplicates (MD5) and within-patient near-duplicates (perceptual hash)
  were removed.
* **Audit before training** (`data_audit/audit_report.json`): manifest = files on disk ({audit['manifest_matches_disk']}), no patient with
  >1 label, {audit['duplicate_file_md5']} duplicate files, {audit['cross_patient_near_duplicate_pairs']} near-duplicate pairs across patients
  (32×32 difference hash + grey-level correlation incl. horizontal flip; highest cross-patient correlation {audit['cross_patient_max_gray_corr']}).
* **Split at patient level**; train, validation and test never share a patient (asserted in code for every fold).
* Augmentation only in training; class weights computed on the training split only; normalisation uses the fixed ImageNet
  statistics of each pretrained model (nothing estimated from our data).
* Early stopping, LR scheduling and checkpoint selection use the validation fold only; the test fold is predicted **once**
  with the selected weights; class decision = argmax (no threshold tuning).
* Each fold starts again from the ImageNet weights; nothing is carried across folds.

Residual risks: patient groups for folderless *normal* images are estimates; one malignant group (M004) may contain more than one person
(kept as one group, which is conservative).

## 3. Experimental Setup

* PyTorch {json.load(open(next((R / 'trained_models').glob('*/fold_00/done.json'))))['torch']}, timm {json.load(open(next((R / 'trained_models').glob('*/fold_00/done.json'))))['timm']}, NVIDIA RTX 4090 (24 GB), Windows 11, bfloat16 autocast.
* The project's earlier scripts are Keras-based; TensorFlow is not installed and has no native-Windows GPU support after 2.10,
  so the same architectures were taken from timm (ImageNet-1k weights; Keras-ported weights for EfficientNetB0).
* The GPU was shared with other training jobs during the run; training times and latencies are therefore indicative only.
* **One fixed seed, 42**, for every model and every fold (Python, NumPy, PyTorch, data-loader generator); cuDNN deterministic mode.

## 4. 5-Fold Cross-Validation Strategy

Split method: **{audit.get('split_method', 'see data_audit/audit_report.json')}**.
Every test and validation fold contains all three classes (folds of `StratifiedGroupKFold` missing a class:
{audit['sklearn_StratifiedGroupKFold_folds_missing_a_class'] or 'none'}).

For test fold *k*, the validation fold is *(k+1) mod 5* and the remaining 3 folds form the training set
(≈60 / 20 / 20 % of patients). The same folds were used for every model (`fold_splits/folds.csv`, per-fold file lists
`fold_splits/fold_XX_files.csv`).

{md(te, index=False)}

With 11 malignant patients, each test fold contains 2–3 malignant patients (9–37 images), so fold-level malignant metrics
still depend on individual patients.

## 5. Pre-trained Model Architectures

{md(eff)}

Input size is 224×224 for all five models.
Head: global average pooling → dropout 0.2 → linear layer (3 classes) (timm default head).

## 6. Training and Fine-Tuning Strategy

| Item | Setting (identical for all models) |
|---|---|
| Phase 1 (head only) | backbone frozen (BatchNorm in eval mode), AdamW lr 1e-3, max 20 epochs, early-stopping patience 5 |
| Phase 2 (full fine-tuning) | all layers trainable, AdamW lr 1e-5, max 100 epochs, early-stopping patience 12; starts from the best phase-1 weights |
| Weight decay / batch | 1e-4 / 16 (drop_last) |
| LR scheduler | ReduceLROnPlateau on validation loss (factor 0.5, patience 4, min 1e-7) in both phases |
| Loss | cross-entropy with 'balanced' class weights from the training split |
| Checkpoint / early stopping | lowest validation loss across both phases (ModelCheckpoint equivalent) |
| Augmentation (train only) | RandomResizedCrop (scale 0.7–1.0, ratio 0.85–1.18), horizontal flip, rotation ±15°, ColorJitter (0.15, 0.15, 0.10, 0.02) |
| Evaluation transform | whole image resized to input size, model-specific ImageNet mean/std |

Deviations from the requested starting protocol: patience for the short head phase is 5 (not 12); best weights are stored in fp16
to save disk space (predictions were made with the fp32 weights before saving).

Epochs run and where the selected checkpoint came from:

{md(sel)}

In 1–3 of the 5 folds per model the lowest validation loss was reached in the head-only phase (EfficientNetB0 3/5, DenseNet121 2/5),
i.e. full fine-tuning did not lower validation loss further; with 86–108 validation images, validation loss is noisy (see `training_curves/`).

## 7. Performance Evaluation

Per fold (test fold only): accuracy, macro precision / recall / F1, weighted F1, macro specificity, ROC-AUC (one-vs-rest, macro),
balanced accuracy, Cohen's κ, MCC, class-wise precision / recall / F1 / specificity / AUC, confusion matrix
(`fold_metrics/`, `predictions/` with true labels, predicted classes and class probabilities). AUC was defined in all
{len(fm)} model-fold runs (every test fold contains all classes). Mean ± SD use the sample SD (ddof = 1) over 5 folds.
Supplementary: pooled out-of-fold metrics and patient-level metrics (mean probability over a patient's images).

## 8. Comparative Results

### 8.1 Main table (mean ± SD over 5 test folds, sorted by macro F1)

{md(main)}

All metrics: `tables/summary_mean_sd_all_metrics.csv`.

{md(full[['Balanced accuracy (%)', 'F1-score (weighted) (%)', 'Specificity (macro) (%)', "Cohen's kappa"]].loc[order])}

### 8.2 Class-wise performance (mean over folds, %)

{md(pc)}

### 8.3 Pooled out-of-fold (all 477 test predictions; %)

{md(oofc)}

### 8.4 Patient-level (92 patients, mean probability per patient; %)

{md(patc)}

### 8.5 Patient-level bootstrap 95% confidence intervals

Fold mean ± SD describes how much results vary between splits, but it is not a confidence interval. Pooled out-of-fold predictions
were therefore resampled by **patient** (cluster bootstrap, B = 2000, stratified 11 malignant / 35 benign / 46 normal patients per
replicate; `07_bootstrap_ci.py`). Full tables are in the Appendix and in `bootstrap_ci/`. Main points:

* The intervals are wide. For ResNet50 (best pooled image-level macro F1), macro F1 is 62.3 % [52.9, 71.3] and malignant recall is
  43.4 % [21.9, 71.4]. At patient level, its malignant recall is 54.5 % [27.3, 81.8], i.e. 6/11 patients.
* Paired differences against the best model: the 95 % CI excludes 0 only against MobileNetV3Large (image-level macro F1
  +11.8 pp [+4.5, +18.2], AUC +8.1 pp [+1.3, +13.6]; patient-level macro F1 +12.1 pp [+0.2, +23.1] for ConvNeXtTiny).
  ResNet50, DenseNet121 and ConvNeXtTiny cannot be separated, and EfficientNetB0 is not significantly worse either
  (macro F1 +8.3 pp [−0.6, +17.6]).

Figures: `comparison_plots/` (accuracy, F1, AUC, MCC bar charts; fold-wise box plots; class-wise P/R/F1; performance vs parameters
and latency), `confusion_matrices/`, `roc_curves/`, `training_curves/` (PNG 400 dpi + PDF).

## 9. Statistical Analysis

Folds are matched blocks (same splits for all models). Friedman test over the 5 models (5 folds):

{md(frc, index=False)}

Mean ranks (1 = best):

{md(ranks, index=False)}

Post-hoc: two-sided Wilcoxon signed-rank tests (effect size: matched-pairs rank-biserial correlation) and the Nadeau–Bengio corrected
resampled t-test, which inflates the variance to account for overlapping training sets in CV. Holm correction was applied
(a) over all {n_pairs} pairs and (b) over the 4 comparisons against the best-ranked model.
Over all pairs, the number of Holm-significant Wilcoxon differences is {', '.join(f'{k}: {v}' for k, v in n_sig_all.items())}.
With 5 folds the smallest attainable two-sided Wilcoxon p is 0.0625, so **no Wilcoxon test can reach p < 0.05 at all**;
the corrected t-test and the patient bootstrap (§8.5) are the informative tests here.

Macro F1, best-ranked model vs others:

{md(vsbest('f1_macro'), index=False)}

ROC-AUC, best-ranked model vs others:

{md(vsbest('roc_auc_ovr_macro'), index=False)}

Full results: `statistical_tests/pairwise_tests.csv`, `friedman_tests.csv`, `mean_ranks_*.csv`.

**Limitations of the tests.** CV fold scores are not independent (training sets overlap by ~2/3), so Friedman and Wilcoxon
p-values are optimistic; the corrected t-test partly addresses this. Five folds give low power, and every fold contains only
2–3 malignant patients, so fold scores are noisy.

## 10. Discussion

**Overall.** Only accuracy differs between the models in the Friedman test (p = 0.019, Kendall's W = 0.59); macro F1 (p = 0.47),
AUC (p = 0.28), MCC and balanced accuracy do not. No pairwise difference is significant after Holm correction with either test.
The pattern is a split into two groups rather than a single winner:

* *Top group* — ConvNeXtTiny, DenseNet121, ResNet50: macro F1 57.4–58.5 %, AUC 81.0–83.3 %, MCC 0.43–0.47. Differences inside this
  group are ≤ 1.1 pp in macro F1, far smaller than the fold SD (≈ 9–13 pp).
* *Lightweight group* — EfficientNetB0, MobileNetV3Large: macro F1 49.4–51.8 %, AUC 73.9–74.7 %. Under the patient bootstrap, only
  MobileNetV3Large is significantly worse than the best model. Both were trained with exactly the same protocol; their lower performance
  may reflect sensitivity to the shared low fine-tuning LR and small batch rather than an intrinsic limit (not tested here, by design of a
  fixed protocol).

**Which model is best?** Judged on macro F1, class-wise performance, AUC and stability rather than accuracy:

* **ConvNeXtTiny** has the highest mean macro F1 (58.5 ± 12.9 %), accuracy (69.3 %) and MCC (0.47), the best mean rank for macro F1, and the
  best patient-level macro F1 (66.5 %). Its weakness is the malignant class: the lowest malignant recall of the top group
  (fold mean 27.2 %; pooled 23/83 images; 4/11 patients). Its accuracy advantage comes from benign and normal images.
* **ResNet50** has the highest AUC (83.3 ± 5.0 %), the most stable accuracy (SD 0.8 pp), the best pooled image-level macro F1 (62.3 %),
  and detects 6/11 malignant patients.
* **DenseNet121** has the highest malignant F1 (fold mean 38.5 %; pooled 44.6 %), the best mean rank for balanced accuracy and also
  detects 6/11 malignant patients.

Statistically these three cannot be separated. For the clinically critical malignant class, **ResNet50** and **DenseNet121** offer a better
balance than ConvNeXtTiny; ConvNeXtTiny is best on the aggregate metrics. A claim of superiority of any one of them is not supported by these data.

**Class-wise behaviour.** Normal mucosa (recall 56.5–78.2 %) and benign lesions (58.8–77.1 %) are recognised far better than malignant lesions,
which are the bottleneck for every model (fold-mean F1 25.7–38.5 %, recall 27.2–46.7 %). In the pooled confusion matrices, misclassified
malignant images are predicted as *benign* more often than as *normal* by every model except DenseNet121
(benign: 25–51 % of the 83 malignant images; normal: 7–30 %), i.e. benign-vs-malignant discrimination is the main difficulty.

**Practical vs statistical significance.** The ~6–9 pp macro-F1 gap between the top and the lightweight group is practically relevant;
the ≤ 1.1 pp gaps within the top group are not distinguishable from fold-to-fold noise.

## 11. Limitations

1. Only 11 malignant patients (83 images); each test fold has 2–3 malignant patients, so malignant metrics are dominated by individual patients.
2. Single centre, no external test set; results may not transfer to other cameras or populations.
3. Patient groups for folderless normal images are estimated from EXIF session and similarity; one malignant group may be mixed.
4. Malignant class is heterogeneous (SCC + one melanoma + unspecified); benign subtypes are partly unspecified.
5. Five folds give little statistical power (minimum Wilcoxon p = 0.0625).
6. One fixed training protocol and one seed (42); no per-model hyper-parameter tuning; no repeated CV. The bootstrap CIs (§8.5)
   capture patient sampling uncertainty but not training-seed variability.
7. Image-level metrics weight patients with many images more (e.g. one malignant patient has 24 images); patient-level results are given as a supplement.
8. The GPU was shared with other jobs; latency and training time are indicative only.
9. timm/PyTorch weights differ slightly from Keras Applications weights used in the earlier project scripts.

## 12. Conclusion

Under a leakage-controlled, patient-grouped 5-fold protocol with one fixed seed, ImageNet-pretrained ConvNeXtTiny, DenseNet121 and ResNet50
reach macro F1 ≈ 57–59 % and OvR AUC ≈ 81–83 % for benign / malignant / normal intra-oral photographs, with large fold-to-fold variability
(SD 9–13 pp). The lightweight EfficientNetB0 and MobileNetV3Large perform worse under the same protocol. Malignant lesions are the main source of
error, and the available 11 malignant patients are too few for a statistically confirmed ranking of the leading models.
More malignant patients (or a validated external set) are the main requirement before clinical claims can be made.

## Appendix: patient-level bootstrap 95% CIs (3-class)

{bs_md}

## Reproducibility

`python 01_audit_and_splits.py` → `python 02_train_cv.py` (resumable) → `python 03_analyze.py` → `python 07_bootstrap_ci.py` → `python 04_report.py`.
Configuration: `common.py`; per-run configuration and environment: `trained_models/<model>/fold_XX/done.json`; training logs:
`training_log.csv`; best weights: `best_weights_fp16.pt`; test predictions: `predictions/<model>/fold_XX_test_predictions.csv`.
"""
(out / 'REPORT.md').write_text(T, encoding='utf-8')
print('written', out / 'REPORT.md', len(T))
