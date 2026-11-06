# Foundation-Model-Guided ConvNeXt Classification of Oral Lesions in Clinical Images

Code for the manuscript *"Foundation-Model-Guided ConvNeXt Classification of Oral Lesions in Clinical Images"*
(submitted to the *Journal of Imaging Informatics in Medicine*).

The repository contains:

1. A **benchmark** of five ImageNet-pretrained CNNs (ResNet50, DenseNet121, EfficientNet-B0, MobileNetV3-Large,
   ConvNeXt-Tiny). All five are trained with one generic transfer-learning protocol and evaluated with five-fold
   cross-validation.
2. The **proposed model**: a ConvNeXt-Tiny (ImageNet-12k initialisation) trained with a strong fine-tuning recipe and
   three components that are used only during training:
   - label-free feature distillation from a frozen **DINOv3 ViT-L/16** teacher;
   - a **supervised contrastive** loss;
   - **class-balanced classifier re-training** (cRW).

   At inference only the plain ConvNeXt-Tiny is used (27.8 M parameters).
3. The **ablation study**, the **robustness analyses** and the **statistical analysis** reported in the paper:
   - robustness: training seeds, patient-grouped NDO, nested cross-validation;
   - statistics: Nadeau–Bengio corrected t-test with Holm correction, and a patient-cluster bootstrap.

The training scheme is shown in Fig. 2 of the paper. In the code the proposed model is the configuration **`A6`**.

## Data

| Data set | Task(s) | Images | Split unit | Availability |
|---|---|---|---|---|
| In-house clinical images (Mersin University) | 3-class (benign / malignant / normal); cancer vs non-cancer | 477 (92 patients) | patient | **not public** (patient privacy); available from the corresponding author on reasonable request with ethics-committee approval |
| NDO (Piyarathne et al., *Oral Oncol* 2024; [Zenodo 10.5281/zenodo.10664056](https://doi.org/10.5281/zenodo.10664056)) | 2-, 3- and 4-class | 2271 / 2871 / 3000 | image; patient (re-evaluation) | public, CC BY-NC-ND 4.0 |
| OCD3, "Oral Images Dataset" ([Mendeley 10.17632/mhjyrn35p4.2](https://doi.org/10.17632/mhjyrn35p4.2)), original images only | benign vs malignant | 323 | image | public, CC BY 4.0 |
| OCD2, "Oral Cancer Dataset" ([Kaggle zaidpy/oral-cancer-dataset](https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset)) | cancer vs non-cancer | 950 | image | public (Kaggle) |

No images, predictions, trained weights or patient identifiers are included in this repository.

Expected folder layout. The paths can be changed with environment variables.

```
<TUSEB_PUBLIC_ROOT>/                 # default: parent folder of this repository
├── our_data_clean/                  # in-house data: manifest.csv + images  (or set TUSEB_DATA)
├── new_dataset_oral/
│   ├── dataset/                     # NDO 4-class: Benign, Healthy, OCA, OPMD
│   ├── dataset_3class/              # NDO 3-class: Benign, Healthy, OPMD
│   └── dataset_two/                 # NDO 2-class: Benign, OPMD (OCA merged into OPMD)
├── dataset_oral/Imagewise_Data.csv  # NDO patient identifiers (for the patient-grouped re-evaluation)
├── new_oral_cancer_Data_3/          # OCD3 original images: benign_lesions, malignant_lesions
└── new_oral_cancer_data_2/          # OCD2: NON CANCER, CANCER
```

The in-house `manifest.csv` lists one row per image with the file name, a coded patient identifier and the class.
Patient-grouped folds are built from this coded identifier.

## Installation

Tested on Windows 11 with Python 3.10, PyTorch 2.14 (CUDA 13.0) and one NVIDIA RTX 4090 GPU.

```bash
pip install -r requirements.txt
```

The DINOv3 teacher (`vit_large_patch16_dinov3.lvd1689m`) is downloaded through `timm`. Its use is governed by
the DINOv3 License.

Optional environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `TUSEB_DATA` | in-house image folder | `../our_data_clean` |
| `TUSEB_PUBLIC_ROOT` | folder holding the public data sets | parent folder of the repository |
| `TUSEB_NOVEL_CACHE` | image and teacher-embedding cache of the proposed model (several GB) | `./novel_cache` |
| `TUSEB_PYTHON` | Python interpreter used by the PowerShell run scripts | `python` |

## Evaluation protocol

- **Five-fold cross-validation**, the same folds for every model. Test fold *k*, validation fold (*k* + 1) mod 5,
  and the remaining three folds for training (60/20/20 %). The validation fold is used only for early stopping,
  checkpoint selection, flip test-time augmentation and temperature scaling.
- **In-house data:** folds are formed at patient level, stratified by class.
- **Public data:** stratified image-level folds, as in the published use of these data sets.
- **NDO patient-grouped re-evaluation** (`public/05_patient_folds.py`): NDO is re-split by patient, and one
  patient-to-fold map is shared by the 2-, 3- and 4-class variants.
- **Training seed:** one fixed seed (42) for all main results; seeds 43 and 44 for the seed-robustness analysis.

## Reproducing the paper

### 1. Benchmark CNNs: in-house data (repository root)
```bash
python 01_audit_and_splits.py            # data audit and patient-grouped folds
python -u 02_train_cv.py                 # 3-class benchmark (resumable)
python -u 02_train_cv.py --task binary   # cancer vs non-cancer benchmark
python 03_analyze.py                     # metrics, figures, statistics
python 05_analyze_binary.py
python 07_bootstrap_ci.py                # patient-cluster bootstrap CIs
python 04_report.py; python 06_report_binary.py
```

### 2. Benchmark CNNs: public data (`public/`)
```bash
python 00_audit_public.py                # data audit, exact-duplicate check
python 01_prepare_public.py              # stratified image-level folds, image cache
python -u 02_train_public.py             # OCD3, OCD2, NDO 2/3/4-class
python 05_patient_folds.py               # patient-grouped NDO folds
python -u 02_train_public.py --datasets NDO_2class_pg NDO_3class_pg NDO_4class_pg
python 03_analyze_public.py; python 04_report_public.py
```

### 3. Proposed model and ablation (`novel/`)
```bash
# cache the frozen DINOv3 ViT-L/16 embeddings once (label-free, no augmentation)
python cache_teacher.py --teacher dinov3_vitl --tasks ours_3class OCD3_2class OCD2_2class NDO_2class NDO_3class NDO_4class

# proposed model (A6) on all tasks; the cancer-vs-non-cancer result is obtained from the 3-class model
python train_novel.py --tasks ours_3class OCD3_2class OCD2_2class NDO_2class NDO_3class NDO_4class --rows A6

# ablation rows of Table 8 (A1, A2a-c, A4c, A4s, A2r, A5, A5x, A7, A10, B4, C8), e.g.
python train_novel.py --tasks ours_3class --rows A1 A2a A2b A2c A4c A4s A2r A5 A5x A6 A7

python analyze_novel.py --ref A0         # fold-level tests vs the benchmark ConvNeXt-Tiny
python analyze_novel.py --ref A2b        # ... vs the matched control
python report_novel.py                   # tables, bootstrap, final report
```
The PowerShell scripts `novel/run_stage1.ps1` … `run_stage5.ps1`, `run_seeds.ps1` and `run_revision.ps1` contain
the exact run queues used for the paper, including the seed and nested cross-validation runs.

### 4. Robustness analyses (`novel/`)
```bash
python seed_analysis.py                  # seeds 42-44 (Table 9)
python pg_analysis.py                    # patient-grouped NDO (Table 9)
python revision_analyses.py              # tests vs strong controls, binary decision rules, patient-level results
```
Nested cross-validation (inner loop): `python train_novel.py --tasks ours_3class --rows A1 A2b A5 A6 --nested <outer fold>`
(see `run_revision.ps1`); results are written to `results/nested/outer<k>/`.

A quick pipeline check: `python train_novel.py --tasks OCD3_2class --rows A6 --folds 0 --smoke`.
`--smoke` writes to `results/smoke/`; these are not real results.

## Main configuration of the proposed model (`A6`, see Table 2 of the paper)

| Setting | Value |
|---|---|
| Backbone | ConvNeXt-Tiny, ImageNet-12k initialisation (`timm`) |
| Input | 384 × 384 (from a 768 px cache) |
| Augmentation | random resized crop (0.7–1.0), flip, rotation ± 15°, RandAugment(2, 9), colour jitter, random erasing (p = 0.25), Shades-of-Gray colour normalisation |
| Stage 1 | linear probing, AdamW, lr 1e-3, 5 epochs |
| Stage 2 | full fine-tuning, AdamW, lr 1e-4, 30 epochs, 2 warm-up epochs + cosine, layer-wise decay 0.8, weight decay 0.05, drop-path 0.1, label smoothing 0.1, batch 16 (each class ≥ 2 per batch) |
| Distillation | DINOv3 ViT-L/16 teacher (256 px input, mean patch token, L2-normalised); head 768 → 512 → 1024; cosine distance; λ_KD = 1 in epochs 1–21, 0 in epochs 22–30; unlabeled NDO images as extra distillation-only batches for the in-house data |
| Supervised contrastive loss | head 768 → 768 → 128, τ = 0.07, λ_SC = 0.2 |
| Stage 3 (cRW) | linear classifier only, class-balanced sampling, AdamW, lr 1e-3, 10 epochs |
| Checkpoint | lowest class-weighted validation loss; flip TTA and temperature chosen on the validation fold |

## Citation

If you use this code, please cite the paper (the reference will be added after publication).

## Acknowledgements

This work was supported by the Health Institutes of Türkiye (TUSEB) under the "2023-C1-YZ" call, Project No. 33798.
The study was approved by the Mersin University Clinical Research Ethics Committee (decision no. 2024/1122).

## License

The code is released under the MIT License (see `LICENSE`). The data sets and the DINOv3 weights keep their own
licences.
