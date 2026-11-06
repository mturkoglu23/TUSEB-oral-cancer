"""Write results/binary/final_report/REPORT_binary.md from the tables produced by 05_analyze_binary.py.
Tables are read from the CSV/MD files; the narrative was written after the 25/25 runs (5 models x 5 folds, seed 42) had finished."""
import pandas as pd
from common import *

R = RESULTS / 'binary'; T = R / 'tables'; S = R / 'statistical_tests'
out = R / 'final_report'; out.mkdir(parents=True, exist_ok=True)
rd = lambda f: pd.read_csv(f, encoding='utf-8-sig')
md = lambda f: (T / f).read_text(encoding='utf-8')

comp = rd(T / 'completeness.csv'); n_done = int(comp.n_complete.sum())
fr = rd(S / 'friedman_tests.csv')
fr_md = fr.assign(p_value=fr.p_value.map(lambda v: f'{v:.3g}'), friedman_chi2=fr.friedman_chi2.round(2), kendalls_W=fr.kendalls_W.round(3))[
    ['metric', 'n_folds', 'n_models', 'friedman_chi2', 'p_value', 'kendalls_W']].to_markdown(index=False)
pw = rd(S / 'pairwise_tests.csv')
sig_all = pw[pw.wilcoxon_p_holm_all_pairs < .05]; sig_best = pw[pw.wilcoxon_p_holm_vs_best < .05]
sig_best_txt = '; '.join(f"{r.metric}: {r.model_a} vs {r.model_b} (Holm p = {r.wilcoxon_p_holm_vs_best:.3f})" for r in sig_best.itertuples()) or 'none'
cmp = rd(S / 'binary_vs_3class_collapsed.csv')
auc_cmp = cmp[cmp.metric == 'roc_auc'][['model', 'binary_mean', 'from_3class_mean', 'mean_diff', 'wilcoxon_p', 'wilcoxon_p_holm', 'rank_biserial']]
n_lower = int((auc_cmp.mean_diff < 0).sum())
auc_cmp_md = auc_cmp.assign(**{c: (auc_cmp[c] * 100).round(1) for c in ['binary_mean', 'from_3class_mean', 'mean_diff']},
                            wilcoxon_p=auc_cmp.wilcoxon_p.round(3), wilcoxon_p_holm=auc_cmp.wilcoxon_p_holm.round(3),
                            rank_biserial=auc_cmp.rank_biserial.round(2)).rename(columns={
    'binary_mean': 'AUC binary (%)', 'from_3class_mean': 'AUC 3-class collapsed (%)', 'mean_diff': 'Diff (pp)',
    'wilcoxon_p': 'Wilcoxon p', 'wilcoxon_p_holm': 'Holm p', 'rank_biserial': 'r_rb'}).to_markdown(index=False)
lv = rd(T / 'pooled_and_patient_auc_binary_vs_3class.csv')
n_pool = int((lv.pooled_image_auc_3class > lv.pooled_image_auc_binary).sum()); n_pat = int((lv.patient_auc_3class > lv.patient_auc_binary).sum())
lv_md = lv.round(3).to_markdown(index=False)
pat = rd(T / 'patient_level_oof_metrics.csv')[['model', 'roc_auc', 'cancer_patients_detected', 'benign_patients_flagged', 'normal_patients_flagged']]
pat_md = pat.assign(roc_auc=pat.roc_auc.round(3)).to_markdown(index=False)
mp = rd(T / 'malignant_patients_mean_probability.csv').round(2).to_markdown(index=False)
bsf = R / 'bootstrap_ci' / 'BOOTSTRAP_CI_binary.md'
bs_md = bsf.read_text(encoding='utf-8').replace('### ', '#### ') if bsf.exists() else '_not computed yet: run `python 07_bootstrap_ci.py`_'

txt = f"""# Binary experiment: cancer vs non-cancer (patient-grouped 5-fold CV)

Runs completed: **{n_done} / {len(MODELS) * N_FOLDS}** ({len(MODELS)} models × {N_FOLDS} folds, one fixed seed 42); no failed runs.

## 1. Design

- **Task.** Cancer = `malignant` (11 patients, 83 images); non-cancer = `benign` + `normal` (81 patients, 394 images).
- **Same everything as the 3-class experiment.** Same images, same patient-grouped folds (`results/fold_splits/folds.csv`), same
  test/validation fold assignment (validation = fold k+1), same models, same two-phase training, same augmentation, same seed.
  Only the target and the output layer (2 units) change. Class weights are recomputed from each training split.
- **Decision rules reported.**
  1. Binary model, threshold 0.5 (primary, fixed before the results were seen).
  2. Binary model, threshold chosen on that fold's validation predictions (Youden J). The test fold is never used for this.
  3. The 3-class models of the main experiment collapsed to binary: score = P(malignant), decision = argmax == malignant.
     This is a paired comparison on identical test images.
- **Subgroup analysis.** The non-cancer class is split back into its benign and normal subgroups. For each, we report how often it
  is called cancer (false-positive rate), and the benign-vs-malignant AUC with normal images excluded.

## 2. Main results: binary model, threshold 0.5 (mean ± SD over 5 test folds; sorted by ROC-AUC)

{md('main_table_binary_thr0.5.md')}

- **ResNet50** has the highest ROC-AUC (77.7 ± 10.7 %), PR-AUC, sensitivity (60.8 %), F1 and MCC, and the best (for F1 tied) mean rank
  on every metric in the Friedman analysis.
- **ConvNeXtTiny** has the highest accuracy and specificity (88.2 %) but detects the fewest cancer images (sensitivity 31.3 %).
- Sensitivity varies enormously between folds (SD 23–42 pp), because each test fold contains only 2–3 cancer patients.

### Subgroup false-positive rates (binary model, threshold 0.5)

{md('subgroup_false_positive_rates.md')}

## 3. Alternative decision rules

### 3a. Binary model, validation-selected threshold

{md('main_table_binary_valthr.md')}

The validation-selected threshold did **not** help. Balanced accuracy fell for 4 of the 5 models compared with 0.5, and for DenseNet121 and
EfficientNetB0 accuracy dropped by 7–16 pp. Each validation fold contains only 2–3 cancer patients, so the Youden threshold is unstable
and does not carry over to the test fold.

### 3b. 3-class models collapsed to cancer / non-cancer

{md('main_table_from_3class.md')}

## 4. Binary-trained vs 3-class-collapsed (paired, same test images)

Fold-mean ROC-AUC (paired Wilcoxon over the 5 folds; Holm correction across the {len(MODELS)} models):

{auc_cmp_md}

Pooled out-of-fold image-level AUC and patient-level AUC (mean probability per patient):

{lv_md}

**Result.**
- Fold-mean AUC is lower for the binary-trained model in {n_lower}/{len(MODELS)} architectures (EfficientNetB0, MobileNetV3Large, ConvNeXtTiny)
  and slightly higher for ResNet50 (+2.4 pp) and DenseNet121 (+1.4 pp). Pooled image-level AUC is higher for the collapsed 3-class model in
  {n_pool}/{len(MODELS)} architectures, patient-level AUC in {n_pat}/{len(MODELS)}.
- No single model's fold-mean difference survives Holm correction (5 folds; minimum Wilcoxon p = 0.0625).
- The subgroup rates show the mechanism: binary models call normal images cancer about as often as benign images (fold means 11–41 %),
  whereas the 3-class models rarely flag normal mucosa (pooled 3–22 %, below 10 % for ResNet50, DenseNet121 and ConvNeXtTiny).
- Learning normal and benign as separate classes therefore gives a cleaner non-cancer representation. The difficult boundary is the
  same in both formulations: benign vs malignant (binary benign-vs-malignant AUC 58–80 %).
- **Practical recommendation:** to report a cancer / non-cancer result, train 3-class and collapse the output, rather than training
  a binary model. The binary model does not rank cancer images better; it mainly moves the operating point towards more cancer calls.

**Patient-level bootstrap (B = 2000, paired; full tables in the Appendix).**
- AUC: the binary-minus-3-class difference is negative at image level in 5/5 models, but its 95 % CI excludes 0 only for EfficientNetB0
  (−20.2 pp) and ConvNeXtTiny (−3.8 pp); at patient level only for EfficientNetB0.
- Specificity and false alarms on **normal** images show a clear effect. At image level the binary model flags more normal images as cancer
  in 5/5 models, with the CI excluding 0 in 4/5 (e.g. ResNet50 +33.8 pp [+14.8, +50.5]). Specificity is lower in 5/5 models, with the CI
  excluding 0 in all 5 at image level and in 4/5 at patient level.
- Sensitivity is higher for the binary model in 4/5 models at image level, but the CI excludes 0 only for ResNet50 (+12.0 pp [+3.1, +29.0]);
  at patient level no sensitivity difference is significant.
- Conclusion: training binary costs specificity, mainly through extra false alarms on normal mucosa, without a measurable gain in
  discrimination. p_boot values are not multiplicity-corrected.

## 5. Statistical comparison between architectures (binary model, threshold 0.5)

Friedman tests (folds as blocks, {len(MODELS)} models):

{fr_md}

- Pairwise Wilcoxon tests with Holm correction over all {len(MODELS) * (len(MODELS) - 1) // 2} pairs: **{len(sig_all)}** significant pairs.
- Comparisons against the best-ranked model, Holm-corrected: {sig_best_txt}.
- The Friedman test shows an architecture effect for ROC-AUC (p = 0.017), PR-AUC (p = 0.045) and MCC (p = 0.047), but not for balanced
  accuracy, F1 or sensitivity. The effect is driven by EfficientNetB0 falling behind (mean rank 4.4–4.8); under the patient bootstrap
  it is the only model whose AUC is significantly below the best model (ResNet50 − EfficientNetB0 +16.4 pp [+1.7, +31.2]).
  ResNet50, DenseNet121, MobileNetV3Large and ConvNeXtTiny cannot be separated.

## 6. Patient-level results (binary model, threshold 0.5 on the mean probability per patient)

{pat_md}

## 7. Why the spread across folds is so large

Each test fold contains only 2–3 cancer patients, so a fold's AUC is essentially "how these few patients rank against the non-cancer
images in the fold". The table below gives the mean predicted probability for each malignant patient, averaged over all {len(MODELS)} models:

{mp}

- **M003** and **M004** (both in fold 1) are scored very low by every architecture and by both formulations (mean P(cancer) 0.03–0.19).
  Fold 1 is accordingly the fold where binary AUC falls below 0.5 for ConvNeXtTiny, EfficientNetB0 and MobileNetV3Large.
- **M001** (fold 4) is the largest malignant patient (24 of the 83 images, 29 %) and is also scored low (≈ 0.30). Image-level pooled
  metrics are therefore strongly weighted by one patient, which is why patient-level metrics are reported alongside them.
- **M011** and **M008** are called cancer by the binary models more often than by the 3-class models (0.47–0.53 vs 0.23–0.24).
- Fold 2 (M005, M012) is the easiest fold for every model (AUC up to 0.97).
- This is a property of those patients' images, not a training failure. M003, M004 and M001 are worth a clinical review: atypical
  presentation, image quality, or label verification.

## 8. Limitations

- Bootstrap CIs (Appendix) capture patient sampling uncertainty only, not training-seed variability (one seed, 42).
- Only 11 cancer patients. Every estimate has a wide uncertainty, and fold-level SDs of ROC-AUC are about 9–18 points.
- Five folds give little power: no Wilcoxon test can reach p < 0.05. The comparison in §4 should be described as a consistent trend
  in AUC and a clear effect in specificity, not as a proven difference in discrimination.
- The binary experiment reuses the same folds as the 3-class experiment by design, so the two are paired, not independent, replicates.
- Inference latency and training time were measured on a GPU shared with other jobs and are indicative only.

## Appendix: patient-level bootstrap 95% CIs (binary)

{bs_md}

## 9. Files

| What | Where |
|---|---|
| Per-fold metrics (all rules) | `results/binary/fold_metrics/all_models_fold_metrics.csv` |
| Tables | `results/binary/tables/` |
| Statistics | `results/binary/statistical_tests/` (`friedman_tests.csv`, `pairwise_tests.csv`, `binary_vs_3class_collapsed.csv`) |
| Bootstrap CIs | `results/binary/bootstrap_ci/` (`ci_binary.csv`, `diff_binary_minus_3class.csv`, `diff_vs_best_binary.csv`) |
| Figures (PNG 400 dpi + PDF) | `results/binary/comparison_plots/`, `confusion_matrices/`, `roc_curves/` |
| Test and validation predictions | `results/binary/predictions/<model>/fold_XX_{{test,val}}_predictions.csv` |
| Weights (fp16) | `results/binary/trained_models/<model>/fold_XX/best_weights_fp16.pt` |
"""
(out / 'REPORT_binary.md').write_text(txt, encoding='utf-8')
print('written', out / 'REPORT_binary.md', len(txt))
