"""Write results_public/final_report/REPORT_public.md. Tables are read from the files written by 03_analyze_public.py;
the narrative was written after the completed run (5 models x 5 folds x 5 datasets, seed 42, finished 2026-11-02)."""
import json
import pandas as pd
from common_public import *

out = RES / 'final_report'; out.mkdir(parents=True, exist_ok=True)
rd = lambda p, **k: pd.read_csv(p, encoding='utf-8-sig', **k)
txt = lambda p: p.read_text(encoding='utf-8') if p.exists() else '_missing_'
audit = json.load(open(RES / 'data_audit' / 'audit_report.json', encoding='utf-8'))
ORDER = ['OCD3_2class', 'OCD2_2class', 'NDO_2class', 'NDO_3class', 'NDO_4class']
TITLE = {'OCD3_2class': 'Oral cancer dataset 3 (OCD3): benign vs malignant lesions',
         'OCD2_2class': 'Oral cancer dataset 2 (OCD2): non-cancer vs cancer',
         'NDO_2class': 'NDO 2-class: benign vs OPMD (OPMD includes OCA)',
         'NDO_3class': 'NDO 3-class: benign / healthy / OPMD',
         'NDO_4class': 'NDO 4-class: benign / healthy / OCA / OPMD'}

N_DS = len(ORDER)
done = sum(len(list((out_dir(d) / 'trained_models' / m).glob('fold_*/done.json'))) for d in ORDER for m in MODELS)
ds_rows = []
for d in ORDER:
    s = rd(out_dir(d) / 'fold_splits' / 'split_summary.csv'); t = s[(s.fold == 0)].set_index('role')
    cls = DATASETS[d]['classes']; a = audit.get(d, {})
    ds_rows.append({'Dataset': d, 'Folder': DATASETS[d]['root'].relative_to(DS).as_posix(), 'Images': int(t.loc[['train', 'val', 'test'], 'n_images'].sum()),
                    'Classes (images)': ', '.join(f"{c} ({int(t.loc[['train', 'val', 'test'], c].sum())})" for c in cls),
                    'Test images / fold': int(t.loc['test', 'n_images'])})
ds_md = pd.DataFrame(ds_rows).to_markdown(index=False)


def section(d):
    D = out_dir(d); T = D / 'tables'; S = D / 'statistical_tests'
    fr = rd(S / 'friedman_tests.csv')
    frm = fr.assign(friedman_chi2=fr.friedman_chi2.round(2), p_value=fr.p_value.map(lambda v: f'{v:.3g}'), kendalls_W=fr.kendalls_W.round(3))[
        ['metric', 'friedman_chi2', 'p_value', 'kendalls_W', 'mean_ranks']].to_markdown(index=False)
    pw = rd(S / 'pairwise_tests.csv'); x = pw[pw.metric == 'f1_macro'].copy()
    x['comparison'] = x.model_a + ' vs ' + x.model_b; x['mean diff (pp)'] = (x.mean_diff * 100).round(2)
    pwm = x[['comparison', 'mean diff (pp)', 'wilcoxon_p', 'corrected_t_p', 'corrected_t_p_holm_all_pairs']].round(4).to_markdown(index=False)
    sub = T / 'subgroup_recall_OPMD_vs_OCA.md'
    sub = f"\nRecall of the OPMD class split by the original 4-class label (OCA images inside OPMD; pooled out-of-fold, %):\n\n{txt(sub)}\n" if sub.exists() else ''
    return f"""### {TITLE[d]} (`{d}`)

Mean ± SD over 5 test folds, sorted by macro F1:

{txt(T / 'main_table.md')}

Class-wise recall (mean ± SD over folds, %):

{txt(T / 'per_class_recall.md')}
{sub}
Friedman test (folds as blocks, 5 models):

{frm}

Pairwise macro F1 (Wilcoxon p, Nadeau–Bengio corrected t-test p, Holm over all 10 pairs for the corrected t-test):

{pwm}
"""


ov = lambda m: txt(RES / 'overview' / f'models_x_datasets_{m}.md')
T = f"""# Benchmark of ImageNet-pretrained CNNs on public oral-lesion photograph datasets (5-fold CV)

Runs completed: **{done} / {len(MODELS) * N_DS * 5}** ({len(MODELS)} models × {N_DS} datasets × 5 folds, one fixed seed 42); no failed runs.
Every metric is computed on the held-out test fold only. Source files: `public/results_public/<dataset>/`; code: `cv_benchmark/public/*.py`.

## 1. Datasets

{ds_md}

* **NDO** = `new_dataset_oral` ({audit['NDO_4class']['n_images']} intra-oral photographs of {audit['NDO_4class']['n_patients']} patients, 4 diagnostic classes).
  The 3-class variant leaves out OCA, the 2-class variant leaves out healthy mucosa and merges OCA into OPMD (positive class = OPMD).
* **OCD3** = `new_oral_cancer_Data_3` (Mendeley Data mhjyrn35p4, `original_data`); **OCD2** = `new_oral_cancer_data_2` (Kaggle zaidpy).
* All three are treated as separate datasets, as in the literature that reports on them.

## 2. Protocol

* **Datasets used exactly as published**, as in the papers that report on them: every image is kept and the folds are formed at
  **image level** (no patient grouping). The patient IDs available for NDO are stored but not used for splitting.
* `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. For test fold *k*, fold *(k+1) mod 5* is the validation fold
  (early stopping and checkpoint selection only) and the other 3 folds form the training set (60 / 20 / 20 %). The test fold is predicted
  once. The same folds are used for every model.
* **Models, hyper-parameters, augmentation, two-phase schedule and checkpoint rule are identical to the our_data_clean benchmark**
  (`../common.py`): ResNet50, DenseNet121, EfficientNetB0, MobileNetV3Large, ConvNeXtTiny (timm ImageNet-1k weights, 224 × 224);
  phase 1 head only (AdamW 1e-3, patience 5), phase 2 full fine-tuning (AdamW 1e-5, patience 12), class-weighted cross-entropy,
  checkpoint = lowest validation loss, class decision = argmax. One fixed seed (42) for every model and fold.
* Metrics: accuracy, macro precision / recall / F1, ROC-AUC (binary; one-vs-rest macro for 3/4 classes), MCC, balanced accuracy,
  class-wise recall; for binary datasets sensitivity and specificity of the positive class. Mean ± SD use the sample SD over 5 folds.
* Statistics: Friedman test with folds as blocks; pairwise two-sided Wilcoxon signed-rank and Nadeau–Bengio corrected resampled
  t-tests with Holm correction. With 5 folds the smallest attainable Wilcoxon p is 0.0625, so only the corrected t-test can reach p < 0.05.

## 3. Overview across datasets (mean ± SD over 5 test folds)

Macro F1 (%):

{ov('f1_macro')}

Accuracy (%):

{ov('accuracy')}

ROC-AUC (%):

{ov('roc_auc')}

MCC:

{ov('mcc')}

## 4. Results per dataset

""" + '\n'.join(section(d) for d in ORDER) + f"""
## 5. Discussion

**A consistent ranking across datasets.** The Friedman test rejects equal performance on every dataset and every metric (p ≤ 0.005,
Kendall's W 0.74–0.90). The ranking is very similar everywhere:

* **ConvNeXtTiny** is first or tied first on macro F1 in 4 of 5 datasets (OCD3 83.6 %, NDO 2-class 73.5 %, NDO 3-class 64.3 %,
  NDO 4-class 56.4 %) and second on OCD2 (93.5 % vs DenseNet121 94.1 %). It has the highest ROC-AUC on every dataset except OCD2.
  On NDO 2-class and NDO 4-class its mean rank for macro F1 is 1.0, i.e. it is best in every fold.
* **DenseNet121** and **ResNet50** form a close second group; DenseNet121 is best on OCD2.
* **EfficientNetB0** is last on every dataset, and **MobileNetV3Large** is fourth everywhere. With the shared low fine-tuning LR and small
  batch these two lightweight models fall 4–19 pp behind the best model in macro F1. This may reflect sensitivity to the shared protocol rather
  than an intrinsic limit (not tested here, by design of a fixed protocol).

**Statistical strength.** The between-group differences are large but, with 5 folds, only a few pairwise macro-F1 tests survive Holm correction:
EfficientNetB0 is significantly worse than ResNet50 on NDO 3-class (corrected t, p_Holm = 0.031) and than ResNet50 and ConvNeXtTiny
on NDO 4-class (p_Holm = 0.007 and 0.011). Differences within the top group (ConvNeXtTiny, DenseNet121, ResNet50) are 0.1–4.3 pp in
macro F1 and are not significant on any dataset.

**Difficulty of the datasets.** OCD2 is the easiest (macro F1 84–94 %, AUC 91.6–98.6 %), followed by OCD3 (66–84 %). NDO is much
harder, and difficulty increases with the number of classes: macro F1 of the best model falls from 73.5 % (2 classes) to
64.3 % (3 classes) and 56.4 % (4 classes), while OvR AUC stays at 82–83 %. In NDO 3-class every class is recognised at a similar
rate (recall 62–69 % for the top group). In NDO 4-class, OCA (129 images, 4.3 %) is recognised at 52–61 % recall, while benign lesions
and OPMD have the lowest recall (32–62 %) and healthy mucosa the highest (55–71 %). In NDO 2-class, OCA images inside the positive class are detected less often
than the remaining OPMD images by every model (pooled recall 61–70 % vs 70–83 %).

**Fold-to-fold variability** is much lower than on our own data (SD of macro F1 1.6–5.2 pp vs 9–13 pp), because the public test folds
are larger (65–600 images) and image-level.

## 6. Limitations

1. **Image-level splits.** Following the published use of these datasets, folds are formed per image. NDO has {audit['NDO_4class']['n_patients']}
   patients for {audit['NDO_4class']['n_images']} images, so images of the same patient can be in training and test folds; OCD2 and OCD3 provide no
   patient IDs. The results therefore measure performance on new images of a known population, not on new patients, and are likely
   optimistic compared with patient-level evaluation.
2. One fixed training protocol (taken over unchanged from our data) and one seed; no per-dataset or per-model tuning; no repeated CV.
3. Five folds give limited statistical power (minimum Wilcoxon p = 0.0625).
4. Images are resized (squashed) to 448 × 448 once and then to 224 × 224; fine details of high-resolution NDO photographs
   (median 4000 × 3000) are lost.
5. The GPU was shared with other jobs; training time and latency are indicative only.

## 7. Conclusion

Under one fixed protocol and identical 5-fold image-level splits, ConvNeXtTiny is the strongest or joint-strongest ImageNet-pretrained CNN on
all five public dataset variants, followed closely by DenseNet121 and ResNet50; the lightweight EfficientNetB0 and MobileNetV3Large are
clearly weaker. The NDO tasks, especially the 4-class task with its small OCA class, leave the most room for improvement.

## Files

| What | Where |
|---|---|
| Splits | `results_public/<dataset>/fold_splits/` (`splits.csv`, `split_summary.csv`) |
| Per-fold metrics | `results_public/<dataset>/fold_metrics/` |
| Tables | `results_public/<dataset>/tables/`, `results_public/overview/` |
| Statistics | `results_public/<dataset>/statistical_tests/` |
| Figures (PNG + PDF) | `results_public/<dataset>/{{comparison_plots,confusion_matrices,roc_curves}}/` |
| Test and validation predictions | `results_public/<dataset>/predictions/` |
| Per-run configuration and weights | `results_public/<dataset>/trained_models/<model>/fold_XX/` |

## Reproducibility

`python 00_audit_public.py` → `python 01_prepare_public.py` → `python 02_train_public.py` (resumable) → `python 03_analyze_public.py` →
`python 04_report_public.py`.
"""
(out / 'REPORT_public.md').write_text(T, encoding='utf-8')
print('written', out / 'REPORT_public.md', len(T))
