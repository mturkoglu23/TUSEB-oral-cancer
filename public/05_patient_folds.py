"""Patient-grouped 5-fold splits of NDO (internal review 2026-11-03). The published image-level folds put 540 of the 714
patients into more than one fold. Here one patient -> fold map is built on NDO_4class (StratifiedGroupKFold, seed 42;
patients that share a byte-identical image are merged into one group) and applied unchanged to the 2-, 3- and 4-class
variants, so a patient is never in two folds of any variant. Roles as everywhere: test = fold k, val = fold (k+1) % 5.
Writes results_public/NDO_<v>class_pg/fold_splits/splits.csv in the file order of the base set (caches are shared).
  python 05_patient_folds.py
"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from common_public import out_dir, load_split, SEED

K = 5
base = load_split('NDO_4class')
assert base.patient_id.notna().all()

# union-find over patients sharing an identical image (md5)
parent = {p: p for p in base.patient_id.unique()}
def find(p):
    while parent[p] != p: parent[p] = parent[parent[p]]; p = parent[p]
    return p
for _, g in base.groupby('md5'):
    ps = g.patient_id.unique()
    for p in ps[1:]: parent[find(p)] = find(ps[0])
grp = {p: find(p) for p in parent}
merged = {p for p in parent if grp[p] != p} | {grp[p] for p in parent if grp[p] != p}
print('patients', len(parent), '| groups', len(set(grp.values())), '| patients in merged groups:', sorted(merged))

g4 = base.patient_id.map(grp).values
sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED)
fold_of = {}
for k, (_, te) in enumerate(sgkf.split(np.zeros(len(base)), base.y.values, g4)):
    for g in set(g4[te]): fold_of[g] = k

for v in ('NDO_2class', 'NDO_3class', 'NDO_4class'):
    s = load_split(v).copy()
    s['group'] = s.patient_id.map(grp); assert s.group.notna().all(), v
    s['fold'] = s.group.map(fold_of).astype(int)
    for k in range(K):
        s[f'role_k{k}'] = np.where(s.fold == k, 'test', np.where(s.fold == (k + 1) % K, 'val', 'train'))
    assert (s.groupby('group').fold.nunique() == 1).all()
    d = out_dir(v + '_pg') / 'fold_splits'; d.mkdir(parents=True, exist_ok=True)
    s.to_csv(d / 'splits.csv', index=False, encoding='utf-8-sig')
    tab = pd.crosstab(s.fold, s.label); tab['patients'] = s.groupby('fold').group.nunique()
    print(f'\n{v}_pg: {len(s)} images, {s.group.nunique()} patient groups'); print(tab.to_string())
