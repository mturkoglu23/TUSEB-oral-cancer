"""Shared configuration for the patient-grouped 5-fold benchmark on our_data_clean (one fixed seed, 42, for every model and fold)."""
import os, random, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path(os.environ.get('TUSEB_DATA', ROOT.parent / 'our_data_clean'))  # in-house data (not public)
RESULTS = ROOT / 'results'
CACHE = ROOT / 'cache'
CLASSES = ['benign', 'malignant', 'normal']
BINARY_CLASSES = ['non_cancer', 'cancer']  # cancer = malignant; non_cancer = benign + normal
SEED = 42
N_FOLDS = 5
CACHE_SIDE = 448  # images are pre-resized (squashed) to CACHE_SIDE x CACHE_SIDE once; crops/resizes start from here

# name -> (timm model id, input size). 224 for all current models (InceptionV3 / Xception 299 and
# EfficientNetB3 300 in OTHER_MODELS, their canonical ImageNet resolution as in Keras Applications).
# MODELS = the models currently benchmarked (user's choice, 2026-11-01); transformer models are to be added later.
MODELS = {
    'ResNet50':         ('resnet50.tv_in1k', 224),
    'DenseNet121':      ('densenet121.tv_in1k', 224),
    'EfficientNetB0':   ('tf_efficientnet_b0.in1k', 224),
    'MobileNetV3Large': ('mobilenetv3_large_100.ra_in1k', 224),
    'ConvNeXtTiny':     ('convnext_tiny.fb_in1k', 224),
}
# not in the current benchmark (kept for reference; results already on disk are not used by the analysis scripts)
OTHER_MODELS = {
    'ResNet101':        ('resnet101.tv_in1k', 224),
    'DenseNet201':      ('densenet201.tv_in1k', 224),
    'EfficientNetB3':   ('tf_efficientnet_b3.in1k', 300),
    'MobileNetV2':      ('mobilenetv2_100.ra_in1k', 224),
    'Xception':         ('legacy_xception.tf_in1k', 299),
    'InceptionV3':      ('inception_v3.tv_in1k', 299),
    'VGG16':            ('vgg16.tv_in1k', 224),
}

TRAIN = dict(
    batch_size=16, optimizer='AdamW', weight_decay=1e-4, drop_rate=0.2,
    head_lr=1e-3, head_max_epochs=20, head_patience=5,
    ft_lr=1e-5, ft_max_epochs=100, ft_patience=12,
    plateau_factor=0.5, plateau_patience=4, min_lr=1e-7,
    monitor='val_loss', loss='class-weighted cross-entropy (weights from the training split only)',
    amp='bfloat16 autocast',
    augmentation='train only: RandomResizedCrop(scale 0.7-1.0, ratio 0.85-1.18), h-flip, rotation +-15 deg, '
                 'ColorJitter(0.15, 0.15, 0.10, 0.02)',
    eval_transform='resize whole image to input size (no crop)',
    normalization='fixed mean/std of each model\'s ImageNet pretraining (timm pretrained_cfg); nothing learned from data',
)


def seed_everything(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_manifest():
    import pandas as pd
    m = pd.read_csv(DATA / 'manifest.csv', encoding='utf-8-sig')
    m['y'] = m['label'].map({c: i for i, c in enumerate(CLASSES)})
    return m.sort_values('file').reset_index(drop=True)


def load_cache(manifest):
    """uint8 array (N, 3, CACHE_SIDE, CACHE_SIDE) aligned with the manifest order; built once."""
    from PIL import Image
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f'images_{CACHE_SIDE}.npy'
    names = CACHE / f'images_{CACHE_SIDE}.txt'
    files = list(manifest['file'])
    if f.exists() and names.exists() and names.read_text(encoding='utf-8').split('\n') == files:
        return np.load(f, mmap_mode='r')
    arr = np.zeros((len(files), 3, CACHE_SIDE, CACHE_SIDE), np.uint8)
    for i, fn in enumerate(files):
        im = Image.open(DATA / fn).convert('RGB').resize((CACHE_SIDE, CACHE_SIDE), Image.BICUBIC)
        arr[i] = np.asarray(im).transpose(2, 0, 1)
    np.save(f, arr); names.write_text('\n'.join(files), encoding='utf-8')
    return arr
