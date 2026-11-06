"""Losses, augmentation, sampler, SAM and calibration helpers for train_novel.py."""
import math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.v2 as T


# ---------------------------------------------------------------------------------------------- losses
def supcon(z, y, tau=0.07):
    """Supervised contrastive loss (Khosla et al. 2020) on L2-normalised embeddings z (B, d)."""
    sim = z @ z.t() / tau
    self_mask = torch.eye(len(y), device=z.device, dtype=torch.bool)
    pos = (y[:, None] == y[None]) & ~self_mask
    sim = sim.masked_fill(self_mask, -1e9)
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    n = pos.sum(1)
    ok = n > 0
    if not ok.any(): return z.sum() * 0
    return -((logp * pos).sum(1)[ok] / n[ok]).mean()


class ClsLoss(nn.Module):
    """'wce' class-weighted CE (+ label smoothing) | 'logit_adj' logit-adjusted CE (Menon et al.) |
    'ldam_drw' LDAM margins with deferred class-balanced re-weighting (Cao et al.)."""
    def __init__(self, kind, counts, smoothing=0.1, drw_epoch=None, ldam_s=10.0):
        super().__init__(); self.kind, self.smoothing, self.drw_epoch, self.ldam_s = kind, smoothing, drw_epoch, ldam_s
        counts = torch.tensor(counts, dtype=torch.float32).clamp_min(1)
        self.register_buffer('w', counts.sum() / (len(counts) * counts))  # 'balanced' weights, training split only
        self.register_buffer('log_prior', (counts / counts.sum()).log())
        m = counts.pow(-.25); self.register_buffer('margin', m * (.5 / m.max()))
        beta = .9999; eff = (1 - beta ** counts) / (1 - beta); cb = 1 / eff
        self.register_buffer('cb', cb / cb.sum() * len(counts))
        self.epoch = 0

    def forward(self, logits, y):
        logits = logits.float()
        if self.kind == 'wce':
            return F.cross_entropy(logits, y, weight=self.w, label_smoothing=self.smoothing)
        if self.kind == 'logit_adj':
            return F.cross_entropy(logits + self.log_prior, y, label_smoothing=self.smoothing)
        if self.kind == 'ldam_drw':  # margin m_c ~ n_c^-1/4 (max 0.5) subtracted from the true-class logit, scaled by s
            out = logits - self.ldam_s * self.margin[y][:, None] * F.one_hot(y, logits.shape[1])
            w = self.cb if self.drw_epoch is not None and self.epoch >= self.drw_epoch else None
            return F.cross_entropy(out, y, weight=w)
        if self.kind == 'ce':
            return F.cross_entropy(logits, y)
        raise ValueError(self.kind)


def val_loss(p, y, w):
    """class-weighted NLL of probabilities (checkpoint criterion, same as the benchmark)."""
    return F.nll_loss(torch.log(torch.as_tensor(p).clamp_min(1e-8)), torch.as_tensor(y).long(), weight=w.cpu().float()).item()


# ---------------------------------------------------------------------------------------------- augmentation (GPU)
def shades_of_gray(x, p=6):
    """colour constancy on float images in [0, 1], (B, 3, H, W)."""
    illum = x.clamp_min(1e-6).pow(p).mean((2, 3), keepdim=True).pow(1 / p)
    return (x * (illum.mean(1, keepdim=True) / illum.clamp_min(1e-6))).clamp(0, 1)


class GpuAug:
    """per-sample augmentation of a uint8 hi-res batch on the GPU. recipe 'base' = the benchmark augmentation;
    'strong' = A1 recipe (geometric + RandAugment(2, 9) + stronger colour jitter, random erasing 0.25)."""
    def __init__(self, side, recipe, mean, std, sog):
        self.sog = sog; self.mean = torch.tensor(mean).view(1, 3, 1, 1); self.std = torch.tensor(std).view(1, 3, 1, 1)
        geo = [T.RandomResizedCrop(side, scale=(0.7, 1.0), ratio=(0.85, 1.18), antialias=True), T.RandomHorizontalFlip(), T.RandomRotation(15)]
        self.geo = T.Compose(geo)
        if recipe == 'strong':
            self.photo = T.Compose([T.RandAugment(num_ops=2, magnitude=9), T.ColorJitter(0.3, 0.3, 0.2, 0.03)])
            self.erase = T.RandomErasing(0.25, value=0)
        else:
            self.photo = T.ColorJitter(0.15, 0.15, 0.10, 0.02); self.erase = None

    def __call__(self, x, mask=None):
        """x uint8 (B,3,H,W) on GPU; mask uint8 (B,1,h,w) or None -> normalised float batch (+ float mask at x's side)."""
        out, mout = [], []
        for i in range(len(x)):
            xi = x[i]
            if mask is not None:
                mi = F.interpolate(mask[i:i + 1].float(), size=xi.shape[-2:], mode='nearest')[0].to(torch.uint8)
                both = self.geo(torch.cat([xi, mi], 0)); xi, mi = both[:3], both[3:]
                mout.append(mi)
            else:
                xi = self.geo(xi)
            out.append(self.photo(xi))
        x = torch.stack(out).float() / 255
        x = self.normalise(x)
        if self.erase is not None: x = torch.stack([self.erase(xi) for xi in x])
        return x, (torch.stack(mout).float() / 255 if mask is not None else None)

    def normalise(self, x):
        if self.sog: x = shades_of_gray(x)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def eval(self, x):
        return self.normalise(x.float() / 255)


# ---------------------------------------------------------------------------------------------- samplers
def class_aware_batches(y, bs, per_class, rng):
    """one epoch of batches (len(y)//bs); each batch holds >= per_class images of every class (for SupCon)."""
    y = np.asarray(y); idx_c = [np.where(y == c)[0] for c in np.unique(y)]
    out = []
    for _ in range(max(1, len(y) // bs)):
        b = [rng.choice(ic, per_class, replace=len(ic) < per_class) for ic in idx_c]
        b = np.concatenate(b); rest = rng.choice(len(y), bs - len(b), replace=False)
        out.append(rng.permutation(np.concatenate([b, rest])))
    return out


def balanced_batches(y, bs, rng):
    """class-balanced sampling with replacement (stage-2 classifier re-weighting, cRW)."""
    y = np.asarray(y); idx_c = [np.where(y == c)[0] for c in np.unique(y)]; C = len(idx_c)
    return [np.concatenate([rng.choice(idx_c[c], 1) for c in rng.integers(0, C, bs)]) for _ in range(max(1, len(y) // bs))]


def plain_batches(n, bs, rng, drop_last=True):
    p = rng.permutation(n); k = n // bs if drop_last else math.ceil(n / bs)
    return [p[i * bs:(i + 1) * bs] for i in range(k)]


# ---------------------------------------------------------------------------------------------- SAM
class SAM:
    """two-step sharpness-aware minimisation (Foret et al. 2021) over a parameter subset; others step normally."""
    def __init__(self, sam_params, rho=0.05):
        self.params = [p for p in sam_params if p.requires_grad]; self.rho = rho

    @torch.no_grad()
    def ascend(self):
        g = [p.grad for p in self.params if p.grad is not None]
        if not g: self.eps = []; return
        norm = torch.norm(torch.stack([x.norm(2) for x in g]), 2) + 1e-12
        self.eps = []
        for p in self.params:
            if p.grad is None: self.eps.append(None); continue
            e = p.grad * (self.rho / norm); p.add_(e); self.eps.append(e)

    @torch.no_grad()
    def descend(self):
        for p, e in zip(self.params, self.eps):
            if e is not None: p.sub_(e)


# ---------------------------------------------------------------------------------------------- calibration / threshold
def fit_temperature(logits, y):
    """temperature scaling (Guo et al. 2017) fitted on validation logits by NLL."""
    lg = torch.as_tensor(logits, dtype=torch.float32); yy = torch.as_tensor(y).long()
    t = torch.zeros(1, requires_grad=True); opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad(); l = F.cross_entropy(lg / t.exp(), yy); l.backward(); return l
    opt.step(closure)
    return float(t.exp().clamp(0.05, 20))


def youden_threshold(p_pos, y):
    """threshold on P(positive) maximising sensitivity + specificity - 1 (validation fold only)."""
    from sklearn.metrics import roc_curve
    if len(set(np.asarray(y).tolist())) < 2: return 0.5
    fpr, tpr, thr = roc_curve(y, p_pos)
    j = np.argmax(tpr - fpr); return float(np.clip(thr[j], 1e-6, 1 - 1e-6))


def ece(p, y, bins=10):
    """adaptive-bin expected calibration error of the top-class confidence."""
    p = np.asarray(p); y = np.asarray(y); conf = p.max(1); acc = (p.argmax(1) == y).astype(float)
    order = np.argsort(conf); chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(y) * abs(acc[c].mean() - conf[c].mean()) for c in chunks if len(c)))
