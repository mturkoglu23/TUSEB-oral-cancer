"""Shared data layer for the novel ConvNeXt-Tiny designs (A: FM-distilled, B: Look-Twice ZoomNeXt, C: ProtoFormer).

Tasks are the benchmark tasks with their *existing* 5-fold splits (same folds, same fixed seed 42 as the benchmark):
  ours_3class, ours_binary            -> patient-grouped folds of our_data_clean (../results/fold_splits/folds.csv)
  OCD3_2class, OCD2_2class, NDO_*     -> image-level folds of the public datasets (../public/results_public/<name>/fold_splits)
Test = fold k, validation = fold (k+1)%5, training = the other 3 folds.

Images are cached once per task at HI x HI (squashed resize, uint8) so that Design B can crop the lesion from a view
with more pixels than the 384 px network input. The cache lives on E: (C: has too little free space); override with
the environment variable TUSEB_NOVEL_CACHE.
"""
import os, sys, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
BENCH = ROOT.parent
sys.path.insert(0, str(BENCH)); sys.path.insert(0, str(BENCH / 'public'))
from common import DATA, RESULTS, CLASSES, BINARY_CLASSES, SEED, N_FOLDS, seed_everything  # noqa: E402,F401
from common_public import DATASETS, out_dir as public_out_dir  # noqa: E402

HI = 768          # cached side; network input is cfg['size'] (384)
MASK_SIDE = 192   # NDO lesion masks are cached at HI/4
CACHE = pathlib.Path(os.environ.get('TUSEB_NOVEL_CACHE', BENCH / 'novel_cache'))
RES = ROOT / 'results'
TEACHER_CACHE = CACHE / 'teacher_embeddings'
MASK_DIR = pathlib.Path(os.environ.get('TUSEB_PUBLIC_ROOT', BENCH.parent)) / 'dataset_oral' / 'masks'  # NDO masks: <image stem>_mask.png

TASKS = ['ours_3class', 'ours_binary', 'OCD3_2class', 'OCD2_2class', 'NDO_2class', 'NDO_3class', 'NDO_4class']


class Task:
    """files, labels, classes, split roles and the hi-res image cache of one benchmark task."""

    def __init__(self, name):
        self.name = name
        if name.startswith('ours'):
            f = pd.read_csv(RESULTS / 'fold_splits' / 'folds.csv')
            self.files = list(f.file); self.group = f.patient_id.values; self.root = DATA
            if name == 'ours_binary':
                self.classes = BINARY_CLASSES; self.y = (f.label == 'malignant').astype(int).values
            else:
                self.classes = CLASSES; self.y = f.y.values
            self.labels = f.label.values; self.patient = f.patient_id.values
            self._role = lambda k: np.where(f.fold == k, 'test', np.where(f.fold == (k + 1) % N_FOLDS, 'val', 'train'))
            self.cache_key = 'ours'  # 3-class and binary share images
        else:
            f = pd.read_csv(public_out_dir(name) / 'fold_splits' / 'splits.csv', encoding='utf-8-sig')
            self.files = list(f.file); self.group = f.group.values; self.root = DATASETS[name]['root']
            self.classes = DATASETS[name]['classes']; self.y = f.y.values
            self.labels = f.label.values; self.patient = f.patient_id.values if 'patient_id' in f else f.group.values
            self._role = lambda k: f[f'role_k{k}'].values
            self.cache_key = DATASETS[name].get('base', name)  # patient-grouped NDO variants share images and caches
        self.n_classes = len(self.classes)
        self.frame = f

    def split(self, k):
        role = self._role(k)
        te, va, tr = (np.where(role == r)[0] for r in ('test', 'val', 'train'))
        g = self.group
        assert not (set(g[tr]) & set(g[te]) or set(g[tr]) & set(g[va]) or set(g[va]) & set(g[te]))
        return tr, va, te

    def images(self):
        """uint8 (N, 3, HI, HI) memmap aligned with self.files."""
        return _cache(self.cache_key + f'_{HI}', self.files, lambda fn: self.root / fn, 3, HI)

    def masks(self):
        """uint8 (N, 1, MASK_SIDE, MASK_SIDE) lesion masks (NDO only); all-zero where no mask file exists."""
        if not self.name.startswith('NDO'):
            return None
        return _cache(self.cache_key + f'_mask{MASK_SIDE}', self.files,
                      lambda fn: MASK_DIR / (pathlib.Path(fn).stem + '_mask.png'), 1, MASK_SIDE)


def _cache(key, files, path_of, ch, side):
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # one truncated JPEG in new_dataset_oral
    CACHE.mkdir(parents=True, exist_ok=True)
    f, nf = CACHE / f'{key}.npy', CACHE / f'{key}.txt'
    if f.exists() and nf.exists() and nf.read_text(encoding='utf-8').split('\n') == files:
        return np.load(f, mmap_mode='r')
    arr = np.lib.format.open_memmap(str(f) + '.tmp.npy', mode='w+', dtype=np.uint8, shape=(len(files), ch, side, side))
    for i, fn in enumerate(files):
        p = path_of(fn)
        if ch == 1:
            if p.exists():
                arr[i, 0] = (np.asarray(Image.open(p).convert('L').resize((side, side), Image.BILINEAR)) > 127) * 255
            continue
        im = Image.open(p)
        im.draft('RGB', (side * 2, side * 2))
        im = im.convert('RGBA').convert('RGB') if im.mode in ('P', 'LA') else im.convert('RGB')
        arr[i] = np.asarray(im.resize((side, side), Image.BICUBIC)).transpose(2, 0, 1)
        if i % 500 == 0: print(f'cache {key}: {i}/{len(files)}', flush=True)
    arr.flush(); del arr
    os.replace(str(f) + '.tmp.npy', f); nf.write_text('\n'.join(files), encoding='utf-8')
    return np.load(f, mmap_mode='r')
