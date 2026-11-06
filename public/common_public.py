"""Shared configuration for the benchmark of ImageNet-pretrained models on the public oral datasets.
Models, hyper-parameters, augmentation and checkpointing are imported unchanged from ../common.py (our_data_clean
benchmark), so the public results are directly comparable with ours."""
import os, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import MODELS, TRAIN, SEED, CACHE_SIDE, seed_everything  # noqa: E402,F401

ROOT = pathlib.Path(__file__).resolve().parent
DS = pathlib.Path(os.environ.get('TUSEB_PUBLIC_ROOT', ROOT.parent.parent))  # folder holding the public data sets
RES = ROOT / 'results_public'
CACHE = ROOT / 'cache'

# name -> folder, class order (for 2 classes the positive / disease class is index 1), number of folds (5 everywhere,
# as for our data), split unit (image: the datasets are used as published). NDO = new_dataset_oral (patient IDs from dataset_oral/Imagewise_Data.csv).
DATASETS = {
    'OCD3_2class': dict(root=DS / 'new_oral_cancer_Data_3', classes=['benign_lesions', 'malignant_lesions'], k=5, group='image'),
    'OCD2_2class': dict(root=DS / 'new_oral_cancer_data_2', classes=['NON CANCER', 'CANCER'], k=5, group='image'),
    'NDO_2class': dict(root=DS / 'new_dataset_oral' / 'dataset_two', classes=['Benign', 'OPMD'], k=5, group='image'),
    'NDO_3class': dict(root=DS / 'new_dataset_oral' / 'dataset_3class', classes=['Benign', 'Healthy', 'OPMD'], k=5, group='image'),
    'NDO_4class': dict(root=DS / 'new_dataset_oral' / 'dataset', classes=['Benign', 'Healthy', 'OCA', 'OPMD'], k=5, group='image'),
}
# patient-grouped NDO variants (internal review 2026-11-03): same images and file order as the base set, folds from
# 05_patient_folds.py (one patient -> fold map shared by the three variants); caches are shared with the base set
for _n in ('NDO_2class', 'NDO_3class', 'NDO_4class'):
    DATASETS[_n + '_pg'] = dict(DATASETS[_n], group='patient', base=_n)
PUBLISHED = [n for n in DATASETS if 'base' not in DATASETS[n]]  # the five sets as published (default of every script)
EXT = {'.jpg', '.jpeg', '.png'}


def out_dir(name): return RES / name


def load_split(name):
    import pandas as pd
    return pd.read_csv(out_dir(name) / 'fold_splits' / 'splits.csv', encoding='utf-8-sig')


def load_cache(name, files):
    """uint8 (N, 3, CACHE_SIDE, CACHE_SIDE), aligned with `files`; built once per dataset (squashed resize, as for our data)."""
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # one truncated JPEG in new_dataset_oral (OPMD/N-260-01.jpg); see the audit
    CACHE.mkdir(exist_ok=True)
    ck = DATASETS[name].get('base', name)
    f = CACHE / f'{ck}_{CACHE_SIDE}.npy'; nf = CACHE / f'{ck}_{CACHE_SIDE}.txt'
    files = list(files); root = DATASETS[name]['root']
    if f.exists() and nf.exists() and nf.read_text(encoding='utf-8').split('\n') == files:
        return np.load(f, mmap_mode='r')
    arr = np.zeros((len(files), 3, CACHE_SIDE, CACHE_SIDE), np.uint8)
    for i, fn in enumerate(files):
        im = Image.open(root / fn)
        im.draft('RGB', (CACHE_SIDE * 2, CACHE_SIDE * 2))  # fast JPEG decode at reduced scale (still >= 2x the cache side)
        im = im.convert('RGBA').convert('RGB') if im.mode in ('P', 'LA') else im.convert('RGB')
        arr[i] = np.asarray(im.resize((CACHE_SIDE, CACHE_SIDE), Image.BICUBIC)).transpose(2, 0, 1)
    np.save(f, arr); nf.write_text('\n'.join(files), encoding='utf-8')
    return np.load(f, mmap_mode='r')
