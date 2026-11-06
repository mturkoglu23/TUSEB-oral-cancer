"""Data audit + patient-grouped stratified 5-fold split (fixed for every model).

Outputs: results/data_audit/*, results/fold_splits/*
Validation fold for test fold k is fold (k+1) % 5, so train/val/test are disjoint at patient level.
"""
import hashlib, json, collections
import numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from common import *

aud = RESULTS / 'data_audit'; spl = RESULTS / 'fold_splits'
aud.mkdir(parents=True, exist_ok=True); spl.mkdir(parents=True, exist_ok=True)
m = load_manifest()
report = {}

# --- inventory / integrity
on_disk = sorted(p.relative_to(DATA).as_posix() for p in DATA.rglob('*.jpg'))
report['files_on_disk'] = len(on_disk)
report['manifest_rows'] = len(m)
report['manifest_matches_disk'] = on_disk == sorted(m['file'])
report['folder_label_consistent'] = bool((m['file'].str.split('/').str[0] == m['label']).all())
report['folder_patient_consistent'] = bool((m['file'].str.split('/').str[1] == m['patient_id']).all())
report['patients_with_multiple_labels'] = int((m.groupby('patient_id')['label'].nunique() > 1).sum())
md5 = [hashlib.md5((DATA / f).read_bytes()).hexdigest() for f in m['file']]
report['duplicate_file_md5'] = int(pd.Series(md5).duplicated().sum())
report['duplicate_source_md5'] = int(m['source_md5'].duplicated().sum())

per_class = m.groupby('label').agg(images=('file', 'size'), patients=('patient_id', 'nunique'))
ppi = m.groupby(['label', 'patient_id']).size()
per_class['img_per_patient_min'] = ppi.groupby('label').min()
per_class['img_per_patient_median'] = ppi.groupby('label').median()
per_class['img_per_patient_max'] = ppi.groupby('label').max()
per_class.to_csv(aud / 'class_distribution.csv')
m.groupby(['label', 'subtype']).agg(images=('file', 'size'), patients=('patient_id', 'nunique')).to_csv(aud / 'subtype_distribution.csv')
sizes = m[['width', 'height']].describe().round(1); sizes.to_csv(aud / 'image_size_summary.csv')

# --- near-duplicate search ACROSS patients (within-patient duplicates were already removed when the set was built)
def feats(fn):
    g = Image.open(DATA / fn).convert('L')
    a = np.asarray(g.resize((33, 32), Image.BILINEAR), np.float32)
    dh = (a[:, 1:] > a[:, :-1]).ravel()                      # 32x32 difference hash (1024 bits)
    s = np.asarray(g.resize((32, 32), Image.BILINEAR), np.float32).ravel()
    s = (s - s.mean()) / (s.std() + 1e-6)
    return dh, s
F = [feats(f) for f in m['file']]
DH = np.array([f[0] for f in F]); S = np.array([f[1] for f in F]) / 32.0
S_flip = np.array([np.fliplr(f[1].reshape(32, 32)).ravel() for f in F]) / 32.0
corr = np.maximum(S @ S.T, S @ S_flip.T)
ham = (DH[:, None, :] != DH[None, :, :]).mean(2)
pid = m['patient_id'].values
pairs = []
for i in range(len(m)):
    for j in range(i + 1, len(m)):
        if pid[i] != pid[j] and (corr[i, j] > 0.95 or ham[i, j] < 0.12):
            pairs.append(dict(file_a=m.file[i], file_b=m.file[j], corr=round(float(corr[i, j]), 4), dhash_dist=round(float(ham[i, j]), 4)))
pd.DataFrame(pairs, columns=['file_a', 'file_b', 'corr', 'dhash_dist']).to_csv(aud / 'cross_patient_near_duplicates.csv', index=False)
report['cross_patient_near_duplicate_pairs'] = len(pairs)
off = corr[pid[:, None] != pid[None, :]]
report['cross_patient_max_gray_corr'] = round(float(off.max()), 4)
report['cross_patient_min_dhash_dist'] = round(float(ham[pid[:, None] != pid[None, :]].min()), 4)

# --- split. sklearn's StratifiedGroupKFold is tried first and kept only if every fold holds every class;
# with 11 malignant patients it leaves some folds without malignant cases, so a greedy patient-level
# stratified assignment is used instead: per class, patients (largest first, seeded tie-break) go to the
# fold with the fewest images of that class, then fewest images overall. Any class with >= 10 patients
# therefore appears in every fold.
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
f_sk = np.full(len(m), -1)
for k, (_, te) in enumerate(sgkf.split(m, m['y'], groups=m['patient_id'])):
    f_sk[te] = k
sk_ok = all(set(m.label[f_sk == k]) == set(CLASSES) for k in range(N_FOLDS))
report['sklearn_StratifiedGroupKFold_folds_missing_a_class'] = [k for k in range(N_FOLDS) if set(m.label[f_sk == k]) != set(CLASSES)]
if sk_ok:
    m['fold'] = f_sk; method = f'sklearn StratifiedGroupKFold(n_splits={N_FOLDS}, shuffle=True, random_state=42)'
else:
    rng = np.random.RandomState(SEED)
    size = m.groupby('patient_id').size(); lab = m.groupby('patient_id')['label'].first()
    total = np.zeros(N_FOLDS); fold_of = {}
    for c in CLASSES:
        pats = list(lab.index[lab == c]); rng.shuffle(pats)
        pats.sort(key=lambda p: -size[p])  # stable sort keeps the seeded order among equal sizes
        load = np.zeros(N_FOLDS)
        for p in pats:
            k = min(range(N_FOLDS), key=lambda i: (load[i], total[i], i))
            fold_of[p] = k; load[k] += size[p]; total[k] += size[p]
    m['fold'] = m['patient_id'].map(fold_of)
    method = f'greedy patient-level stratified group {N_FOLDS}-fold (seed 42); sklearn StratifiedGroupKFold rejected (folds without malignant)'
assert (m.groupby('patient_id')['fold'].nunique() == 1).all()
m[['file', 'patient_id', 'label', 'y', 'fold']].to_csv(spl / 'folds.csv', index=False)

rows = []
for k in range(N_FOLDS):
    v = (k + 1) % N_FOLDS
    role = np.where(m.fold == k, 'test', np.where(m.fold == v, 'val', 'train'))
    d = m[['file', 'patient_id', 'label']].assign(role=role)
    d.to_csv(spl / f'fold_{k:02d}_files.csv', index=False)
    sets = {r: set(d.patient_id[d.role == r]) for r in ('train', 'val', 'test')}
    assert not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test'])
    for r in ('train', 'val', 'test'):
        s = d[d.role == r]
        row = dict(test_fold=k, val_fold=v, role=r, images=len(s), patients=s.patient_id.nunique())
        for c in CLASSES:
            row[f'{c}_img'] = int((s.label == c).sum()); row[f'{c}_pat'] = s.patient_id[s.label == c].nunique()
        rows.append(row)
summary = pd.DataFrame(rows); summary.to_csv(spl / 'split_summary.csv', index=False)
te = summary[summary.role == 'test']
report['all_classes_in_every_test_fold'] = bool((te[[f'{c}_img' for c in CLASSES]] > 0).all().all())
report['all_classes_in_every_val_fold'] = bool((summary[summary.role == 'val'][[f'{c}_img' for c in CLASSES]] > 0).all().all())
report['malignant_patients_per_test_fold'] = te['malignant_pat'].tolist()
report['malignant_images_per_test_fold'] = te['malignant_img'].tolist()
report['test_images_per_fold'] = te['images'].tolist()
report['split_method'] = method + f'; groups=patient_id; val fold = (test+1) % {N_FOLDS}'
json.dump(report, open(aud / 'audit_report.json', 'w'), indent=2)
print(json.dumps(report, indent=2)); print(per_class); print(te.to_string(index=False))
