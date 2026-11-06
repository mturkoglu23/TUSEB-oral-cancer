"""Audit of the three public oral datasets before any training: inventory, image sizes, exact and near duplicates,
and (for new_dataset_oral) the patient ID encoded in the file name.

Outputs: results_public/data_audit/
"""
import hashlib, json, re, sys, pathlib
import numpy as np, pandas as pd
from PIL import Image, ImageFile

ROOT = pathlib.Path(__file__).resolve().parent
DS = ROOT.parent.parent  # .../TUSEB/dataset
OUT = ROOT / 'results_public' / 'data_audit'; OUT.mkdir(parents=True, exist_ok=True)

SETS = {
    'NDO_4class': DS / 'new_dataset_oral' / 'dataset',
    'NDO_3class': DS / 'new_dataset_oral' / 'dataset_3class',
    'NDO_2class': DS / 'new_dataset_oral' / 'dataset_two',
    'OCD3_2class': DS / 'new_oral_cancer_Data_3',
    'OCD2_2class': DS / 'new_oral_cancer_data_2',
}
EXT = {'.jpg', '.jpeg', '.png'}


def feats(p):
    im = Image.open(p); w, h = im.size; mode = im.mode
    try:
        im.load(); trunc = False
    except OSError:  # truncated file: record it, decode what is there
        ImageFile.LOAD_TRUNCATED_IMAGES = True; im = Image.open(p); im.load(); ImageFile.LOAD_TRUNCATED_IMAGES = False; trunc = True
    g = im.convert('L')
    s = np.asarray(g.resize((32, 32), Image.BILINEAR), np.float32).ravel()
    s = (s - s.mean()) / (s.std() + 1e-6) / 32.0
    sf = np.fliplr(s.reshape(32, 32)).ravel()
    return w, h, mode, s, sf, trunc


rows = []
for name, root in SETS.items():
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() not in EXT: continue
        rows.append(dict(dataset=name, label=p.parent.name, file=p.relative_to(root).as_posix(), path=str(p),
                         md5=hashlib.md5(p.read_bytes()).hexdigest()))
m = pd.DataFrame(rows)
print(m.groupby(['dataset', 'label']).size())

# features once per unique md5
uniq = m.drop_duplicates('md5')[['md5', 'path']]
F = {}
for r in uniq.itertuples():
    F[r.md5] = feats(r.path)
m['width'] = m.md5.map(lambda k: F[k][0]); m['height'] = m.md5.map(lambda k: F[k][1]); m['mode'] = m.md5.map(lambda k: F[k][2])
m['truncated'] = m.md5.map(lambda k: F[k][5])

# patient ID for new_dataset_oral: prefix-number (C-10-12-5 -> C-10, K-004-01 -> K-004)
def pid(fn):
    stem = pathlib.Path(fn).stem
    mm = re.match(r'^([A-Za-z]+)-(\d+)', stem)
    return f'{mm.group(1)}-{mm.group(2)}' if mm else None
m['patient_id'] = np.where(m.dataset.str.startswith('NDO'), m.file.map(pid), None)
m.to_csv(OUT / 'inventory.csv', index=False, encoding='utf-8-sig')

report = {}
for name in SETS:
    d = m[m.dataset == name]
    r = dict(n_images=len(d), per_class=d.label.value_counts().to_dict(),
             unique_md5=int(d.md5.nunique()),
             exact_dup_extra_copies=int(len(d) - d.md5.nunique()),
             md5_in_more_than_one_class=int((d.groupby('md5').label.nunique() > 1).sum()),
             width=d.width.describe().round(0).to_dict(), height=d.height.describe().round(0).to_dict(),
             modes=d['mode'].value_counts().to_dict(), truncated_files=d[d.truncated].file.tolist(),
             top_sizes=(d.width.astype(str) + 'x' + d.height.astype(str)).value_counts().head(5).to_dict())
    if name.startswith('NDO'):
        r['n_patients'] = int(d.patient_id.nunique()); r['unparsed_patient_id'] = int(d.patient_id.isna().sum())
        r['patients_per_class'] = d.groupby('label').patient_id.nunique().to_dict()
        r['patients_with_multiple_labels'] = int((d.groupby('patient_id').label.nunique() > 1).sum())
        r['patient_prefixes'] = d.patient_id.str.split('-').str[0].value_counts().to_dict()
        ipp = d.groupby('patient_id').size(); r['img_per_patient'] = ipp.describe().round(1).to_dict()

    # near duplicates (different md5): grayscale 32x32 correlation incl. horizontal flip
    keys = d.drop_duplicates('md5')
    S = np.stack([F[k][3] for k in keys.md5]); SF = np.stack([F[k][4] for k in keys.md5])
    C = np.maximum(S @ S.T, S @ SF.T); np.fill_diagonal(C, -1)
    lab = keys.label.values; fl = keys.file.values
    ii, jj = np.where(np.triu(C > 0.95, 1))
    pairs = pd.DataFrame(dict(file_a=fl[ii], label_a=lab[ii], file_b=fl[jj], label_b=lab[jj], corr=C[ii, jj].round(4)))
    if name.startswith('NDO'):
        pa = keys.patient_id.values; pairs['patient_a'] = pa[ii]; pairs['patient_b'] = pa[jj]
        pairs['same_patient'] = pairs.patient_a == pairs.patient_b
        r['near_dup_pairs_cross_patient'] = int((~pairs.same_patient).sum())
    pairs.to_csv(OUT / f'near_duplicates_{name}.csv', index=False, encoding='utf-8-sig')
    r['near_dup_pairs_corr>0.95'] = len(pairs)
    r['near_dup_pairs_cross_class'] = int((pairs.label_a != pairs.label_b).sum())
    # connected components of near-duplicate graph -> how many images are in a duplicate cluster
    parent = list(range(len(keys)))
    def f(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in zip(ii, jj): parent[f(a)] = f(b)
    comp = pd.Series([f(x) for x in range(len(keys))])
    sz = comp.map(comp.value_counts())
    r['images_in_near_dup_clusters'] = int((sz > 1).sum()); r['n_clusters_size>1'] = int((comp.value_counts() > 1).sum())
    r['largest_cluster'] = int(sz.max())
    report[name] = r

# relations between the three NDO variants
def mset(n, lab=None):
    d = m[m.dataset == n]; d = d if lab is None else d[d.label == lab]; return set(d.md5)
rel = {}
for lab in ['Benign', 'Healthy', 'OPMD']:
    rel[f'3class/{lab} == 4class/{lab}'] = mset('NDO_3class', lab) == mset('NDO_4class', lab)
rel['2class/Benign == 4class/Benign'] = mset('NDO_2class', 'Benign') == mset('NDO_4class', 'Benign')
rel['2class/OPMD == 4class/OPMD U 4class/OCA'] = mset('NDO_2class', 'OPMD') == (mset('NDO_4class', 'OPMD') | mset('NDO_4class', 'OCA'))
rel['2class/OPMD minus (OPMD U OCA)'] = len(mset('NDO_2class', 'OPMD') - (mset('NDO_4class', 'OPMD') | mset('NDO_4class', 'OCA')))
rel['(OPMD U OCA) minus 2class/OPMD'] = len((mset('NDO_4class', 'OPMD') | mset('NDO_4class', 'OCA')) - mset('NDO_2class', 'OPMD'))
rel['OCD2 md5 overlap OCD3'] = len(mset('OCD2_2class') & mset('OCD3_2class'))
rel['OCD2/OCD3 md5 overlap NDO'] = len((mset('OCD2_2class') | mset('OCD3_2class')) & mset('NDO_4class'))
report['relations'] = rel
json.dump(report, open(OUT / 'audit_report.json', 'w'), indent=2, default=str)
print(json.dumps(report, indent=1, default=str))
