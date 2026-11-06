"""Manifest and fixed outer/inner splits for each public dataset (identical for every model).

The datasets are used exactly as published, as in the papers that report on them: every image is kept (no
de-duplication, nothing removed) and the split is at image level (no patient grouping). patient_id (NDO only, from
dataset_oral/Imagewise_Data.csv) is stored for reference but not used for splitting.
The earlier patient/duplicate-grouped runs are kept in results_public_grouped_archive/.

Splits: StratifiedKFold (5 folds, shuffle, seed 42). As for our data, test fold k uses fold (k+1) % 5 for validation
(early stopping / checkpoint selection only) and the other 3 folds for training (60/20/20). Column role_k<k> holds
train / val / test.

Outputs: results_public/<dataset>/fold_splits/{splits.csv, split_summary.csv}, cache/<dataset>_448.npy
"""
import pathlib, sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from common_public import *

A = RES / 'data_audit'
inv = pd.read_csv(A / 'inventory.csv', encoding='utf-8-sig')
iw = pd.read_csv(DS / 'dataset_oral' / 'Imagewise_Data.csv', header=None, names=['stem', 'class', 'subtype', 'flag'])
iw = iw.set_index('stem')
only = sys.argv[1:] or PUBLISHED

for name in only:
    cfg = DATASETS[name]
    m = inv[inv.dataset == name][['file', 'label', 'md5', 'width', 'height']].copy().sort_values('file').reset_index(drop=True)
    assert set(m.label) == set(cfg['classes']), (name, set(m.label))
    m['y'] = m.label.map({c: i for i, c in enumerate(cfg['classes'])})
    if name.startswith('NDO'):
        stem = m.file.map(lambda f: pathlib.Path(f).stem)
        m['patient_id'] = stem.map(lambda s: s.rsplit('-', 2)[0] if s.startswith('C-') else s.rsplit('-', 1)[0])
        meta = iw.reindex(stem)
        assert meta['class'].notna().all(), 'image missing from Imagewise_Data.csv'
        m['orig_class'] = meta['class'].values; m['subtype'] = meta['subtype'].values  # 4-class label, e.g. OCA inside 2-class OPMD
        pw = pd.read_csv(DS / 'dataset_oral' / 'Patientwise_Data.csv')
        assert set(m.patient_id) <= set(pw['Patient ID']), 'patient ID not in Patientwise_Data.csv'
    else:
        m['patient_id'] = None; m['orig_class'] = m.label; m['subtype'] = None
    m['group'] = m.file  # image-level split: every image is its own unit

    K = cfg['k']
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    m['fold'] = -1
    for k, (_, te) in enumerate(skf.split(m, m.y)): m.loc[te, 'fold'] = k
    summ = []
    for k in range(K):
        role = np.where(m.fold == k, 'test', np.where(m.fold == (k + 1) % K, 'val', 'train')).astype(object)
        m[f'role_k{k}'] = role
        for r in ('train', 'val', 'test'):
            d = m[role == r]
            summ.append(dict(fold=k, role=r, n_images=len(d),
                             **{c: int((d.label == c).sum()) for c in cfg['classes']}))
    od = out_dir(name) / 'fold_splits'; od.mkdir(parents=True, exist_ok=True)
    m.to_csv(od / 'splits.csv', index=False, encoding='utf-8-sig')
    s = pd.DataFrame(summ); s.to_csv(od / 'split_summary.csv', index=False)
    print(f'== {name}: {len(m)} images, image-level stratified K={K}')
    print(s[s.role == 'test'].to_string(index=False))
    load_cache(name, m.file)
    print('cache ok', flush=True)
