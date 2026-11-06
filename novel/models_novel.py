"""Model zoo of the novel designs, all on a timm ConvNeXt-Tiny backbone (stage widths 96/192/384/768).

Every model maps a batch of hi-res images (B, 3, HI, HI, already normalised) to a dict:
  logits  final logits used for evaluation
  ce      list of (logits, weight) terms for the classification loss
  emb     embedding fed to the optional KD / SupCon projectors (Design A components)
  reg     extra regulariser (scalar tensor) or 0
  att     lesion-ness map (Design B only, for the optional mask loss / audits)

  GlobalNet    Design A family and baseline A1: stage-4 GAP -> LayerNorm -> linear
  ZoomNeXt     Design B: multi-scale GeM fusion + zero-init attention + attention-guided zoom with the same backbone
  ProtoFormer  Design C: 2-layer token transformer head (+ CLS cross-attention to stage 3) + multi-prototype cosine head
"""
import copy, math
import torch, torch.nn as nn, torch.nn.functional as F, timm
from torchvision.ops import roi_align
from timm.layers import DropPath

INITS = {'in1k': 'convnext_tiny.fb_in1k', 'in22k_ft_in1k': 'convnext_tiny.fb_in22k_ft_in1k',
         'in12k_ft_in1k': 'convnext_tiny.in12k_ft_in1k', 'dinov3': 'convnext_tiny.dinov3_lvd1689m'}
WIDTHS = [96, 192, 384, 768]


# ---------------------------------------------------------------------------------------------- attention modules
class CoordAtt(nn.Module):
    """Coordinate attention (Hou et al., CVPR 2021)."""
    def __init__(self, c, r=32):
        super().__init__(); m = max(8, c // r)
        self.conv1 = nn.Conv2d(c, m, 1); self.bn = nn.BatchNorm2d(m); self.act = nn.Hardswish()
        self.ch, self.cw = nn.Conv2d(m, c, 1), nn.Conv2d(m, c, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        xh, xw = x.mean(3, keepdim=True), x.mean(2, keepdim=True).transpose(2, 3)
        y = self.act(self.bn(self.conv1(torch.cat([xh, xw], 2))))
        ah, aw = torch.split(y, [h, w], 2)
        return x * self.ch(ah).sigmoid() * self.cw(aw.transpose(2, 3)).sigmoid()


class SimAM(nn.Module):
    """Parameter-free SimAM attention (Yang et al., ICML 2021)."""
    def __init__(self, c=None, lam=1e-4):
        super().__init__(); self.lam = lam

    def forward(self, x):
        n = x.shape[2] * x.shape[3] - 1
        d = (x - x.mean((2, 3), keepdim=True)).pow(2)
        e = d / (4 * (d.sum((2, 3), keepdim=True) / n + self.lam)) + 0.5
        return x * e.sigmoid()


class GatedAtt(nn.Module):
    """x + gamma * Att(x) with gamma = 0 at init, so the pretrained features are untouched at step one."""
    def __init__(self, kind, c):
        super().__init__(); self.att = {'coord': CoordAtt, 'simam': SimAM}[kind](c); self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x): return x + self.gamma * self.att(x)


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__(); self.p = nn.Parameter(torch.tensor(float(p))); self.eps = eps

    def forward(self, x):  # (B, C, H, W) -> (B, C)
        return x.float().clamp_min(self.eps).pow(self.p).mean((2, 3)).pow(1 / self.p)


# ---------------------------------------------------------------------------------------------- backbone
class Backbone(nn.Module):
    """ConvNeXt-Tiny returning all four stage maps; optional gated attention after selected stages (0-based)."""
    def __init__(self, init='in1k', drop_path=0.1, att=None, att_stages=(2, 3)):
        super().__init__()
        self.net = timm.create_model(INITS[init], pretrained=True, num_classes=0, drop_path_rate=drop_path)
        self.cfg = self.net.pretrained_cfg
        self.att = nn.ModuleDict({str(i): GatedAtt(att, WIDTHS[i]) for i in att_stages} if att else {})

    def forward(self, x):
        x = self.net.stem(x); outs = []
        for i, st in enumerate(self.net.stages):
            x = st(x)
            if str(i) in self.att: x = self.att[str(i)](x)
            outs.append(x)
        return outs

    def pooled(self, s4):  # timm head without the classifier: GAP -> LayerNorm (768)
        return self.net.forward_head(self.net.norm_pre(s4), pre_logits=True)


def _linear(i, o, scale=1e-3):  # small head init (ConvNeXt fine-tuning recipe)
    l = nn.Linear(i, o); nn.init.trunc_normal_(l.weight, std=.02)
    l.weight.data.mul_(scale); nn.init.zeros_(l.bias); return l


def resize(x, s): return F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False, antialias=True)


# ---------------------------------------------------------------------------------------------- Design A / baseline
class GlobalNet(nn.Module):
    def __init__(self, n_classes, cfg):
        super().__init__(); self.size = cfg['size']
        self.backbone = Backbone(cfg['init'], cfg['drop_path'], cfg.get('att'))
        self.head = _linear(768, n_classes, cfg['head_init_scale']); self.emb_dim = 768

    def forward(self, x, state=None):
        e = self.backbone.pooled(self.backbone(resize(x, self.size))[3])
        lg = self.head(e)
        return dict(logits=lg, ce=[(lg, 1.0)], emb=e, reg=0.0)

    def classifier_params(self): return list(self.head.parameters())


# ---------------------------------------------------------------------------------------------- Design B
class MultiScaleFusion(nn.Module):
    """GeM-pool S2, S3, S4 (+ attention-pooled S4) -> LN + Linear(256) each -> softmax-weighted sum -> LN."""
    def __init__(self, stages=(1, 2, 3), d=256, att_pool=True):
        super().__init__(); self.stages = stages; self.att_pool = att_pool
        self.pool = nn.ModuleList([GeM() for _ in stages])
        dims = [WIDTHS[s] for s in stages] + ([768] if att_pool else [])
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(c), nn.Linear(c, d)) for c in dims])
        self.w = nn.Parameter(torch.zeros(len(dims))); self.norm = nn.LayerNorm(d)

    def forward(self, feats, a=None):
        parts = [p(feats[s]) for p, s in zip(self.pool, self.stages)]
        if self.att_pool:  # MIL-style attention pooling of S4 with the lesion-ness map: this is what trains the map
            s4 = feats[3].float(); parts.append((s4 * a).sum((2, 3)) / a.sum((2, 3)).clamp_min(1e-6))
        z = torch.stack([pr(p.to(self.w.dtype)) for pr, p in zip(self.proj, parts)], 1)
        return self.norm((z * self.w.softmax(0)[None, :, None]).sum(1))


class ZoomNeXt(nn.Module):
    def __init__(self, n_classes, cfg):
        super().__init__(); self.cfg = cfg; self.size = cfg['size']; d = 256
        self.backbone = Backbone(cfg['init'], cfg['drop_path'], cfg.get('att'))
        self.multiscale = cfg.get('multiscale', True)
        self.lesion = nn.Conv2d(768, 1, 1)  # lesion-ness map on S4
        if self.multiscale:
            self.fusion = MultiScaleFusion(d=d)
        else:
            d = 768
        self.zoom = cfg.get('zoom', 'att')  # None | 'att' | 'random'
        if self.zoom and cfg.get('zoom_separate'):  # B5: unshared zoom backbone
            self.zoom_backbone = copy.deepcopy(self.backbone)
        self.head_g = _linear(d, n_classes, cfg['head_init_scale'])
        self.head_l = _linear(d, n_classes, cfg['head_init_scale'])
        self.head_z = _linear(d, n_classes, cfg['head_init_scale'])
        self.alpha = nn.Parameter(torch.tensor(-2.0))
        self.emb_dim = d

    def encode(self, x, backbone):
        f = backbone(x)
        a = self.lesion(f[3]).float().sigmoid()
        e = self.fusion(f, a) if self.multiscale else backbone.pooled(f[3])
        return e, a

    def boxes(self, a, rand):
        """lesion-ness map (B,1,h,w) -> normalised boxes (B,4) x1,y1,x2,y2: cells >= 0.5*max, expand x1.3, min side 0.35."""
        B, _, h, w = a.shape; c = self.cfg
        m = a[:, 0] >= 0.5 * a.flatten(1).max(1)[0][:, None, None]
        ys = torch.arange(h, device=a.device)[None, :, None].expand(B, h, w); xs = torch.arange(w, device=a.device)[None, None].expand(B, h, w)
        big = torch.tensor(1e9, device=a.device)
        y1 = torch.where(m, ys, big).flatten(1).min(1)[0] / h; y2 = (torch.where(m, ys, -big).flatten(1).max(1)[0] + 1) / h
        x1 = torch.where(m, xs, big).flatten(1).min(1)[0] / w; x2 = (torch.where(m, xs, -big).flatten(1).max(1)[0] + 1) / w
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw = ((x2 - x1) * c.get('zoom_expand', 1.3)).clamp(c.get('zoom_min_side', .35), 1.0)
        bh = ((y2 - y1) * c.get('zoom_expand', 1.3)).clamp(c.get('zoom_min_side', .35), 1.0)
        if rand:  # B3 control: same box size, random position in training, image centre at evaluation
            if self.training:
                cx = bw / 2 + torch.rand_like(bw) * (1 - bw); cy = bh / 2 + torch.rand_like(bh) * (1 - bh)
            else:
                cx = torch.full_like(bw, .5); cy = torch.full_like(bh, .5)
        cx = cx.clamp(bw / 2, 1 - bw / 2); cy = cy.clamp(bh / 2, 1 - bh / 2)
        return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)

    def forward(self, x, state=None):
        xg = resize(x, self.size)
        g, a = self.encode(xg, self.backbone)
        lg = self.head_g(g)
        zoom_on = self.zoom and (state is None or state.get('zoom_on', True))
        if not zoom_on:
            return dict(logits=lg, ce=[(lg, 1.0)], emb=g, reg=0.0, att=a)
        src = xg if self.cfg.get('zoom_source') == 'resized' else x  # B7: crop from the 384 input instead of the hi-res view
        bx = self.boxes(a.detach(), self.zoom == 'random') * src.shape[-1]
        crops = roi_align(src.float(), torch.cat([torch.arange(len(bx), device=x.device)[:, None].float(), bx.float()], 1),
                          output_size=self.size, aligned=True).to(x.dtype)
        l, _ = self.encode(crops, getattr(self, 'zoom_backbone', self.backbone))
        ll = self.head_l(l)
        if self.cfg.get('fusion', 'gate') == 'avg':  # B6: logit averaging
            lz = (lg + ll) / 2; ce = [(lz, 1.0), (lg, .5), (ll, .5)]; e = (g + l) / 2
        else:
            e = g + self.alpha.sigmoid() * l; lz = self.head_z(e); ce = [(lz, 1.0), (lg, .5), (ll, .5)]
        return dict(logits=lz, ce=ce, emb=e, reg=0.0, att=a, boxes=bx / src.shape[-1])

    def classifier_params(self): return [*self.head_g.parameters(), *self.head_l.parameters(), *self.head_z.parameters(), self.alpha]


# ---------------------------------------------------------------------------------------------- Design C
class Mlp(nn.Sequential):
    def __init__(self, d, r): super().__init__(nn.Linear(d, d * r), nn.GELU(), nn.Linear(d * r, d))


class TokenBlock(nn.Module):
    """pre-norm self-attention over [CLS; S4 tokens], optional CLS cross-attention to S3 tokens, MLP."""
    def __init__(self, d=256, heads=4, mlp=2, dp=.1, cross=True):
        super().__init__(); self.cross = cross
        self.n1, self.sa = nn.LayerNorm(d), nn.MultiheadAttention(d, heads, batch_first=True)
        if cross: self.nq, self.nk, self.ca = nn.LayerNorm(d), nn.LayerNorm(d), nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2, self.mlp, self.dp = nn.LayerNorm(d), Mlp(d, mlp), DropPath(dp)

    def forward(self, t, s3):
        h = self.n1(t); t = t + self.dp(self.sa(h, h, h, need_weights=False)[0])
        if self.cross:
            q, kv = self.nq(t[:, :1]), self.nk(s3)
            t = torch.cat([t[:, :1] + self.dp(self.ca(q, kv, kv, need_weights=False)[0]), t[:, 1:]], 1)
        return t + self.dp(self.mlp(self.n2(t)))


class ProtoFormer(nn.Module):
    def __init__(self, n_classes, cfg):
        super().__init__(); self.cfg = cfg; self.size = cfg['size']; d = 256; self.C = n_classes
        self.backbone = Backbone(cfg['init'], cfg['drop_path'], cfg.get('att'))
        g4 = self.size // 32
        self.p3, self.p4 = nn.Linear(384, d), nn.Linear(768, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d)); self.pos = nn.Parameter(torch.zeros(1, g4 * g4, d)); self.g4 = g4
        nn.init.trunc_normal_(self.pos, std=.02); nn.init.trunc_normal_(self.cls, std=.02)
        self.blocks = nn.ModuleList([TokenBlock(d, 4, 2, .1, cfg.get('cross', True)) for _ in range(cfg.get('depth', 2))])
        self.norm = nn.LayerNorm(d)
        self.K = cfg.get('protos', 0)  # 0 = no prototype head (C1, C2)
        if self.K:
            self.protos = nn.Parameter(F.normalize(torch.randn(n_classes, self.K, d), dim=-1))
        self.linear = _linear(d, n_classes, cfg['head_init_scale']) if cfg.get('linear_head', True) or not self.K else None
        self.emb_dim = d

    def proto_logits(self, e, y=None):
        cos = torch.einsum('bd,ckd->bck', F.normalize(e, dim=-1), F.normalize(self.protos, dim=-1))
        gam = 10.0  # smooth max over a class's prototypes
        cc = (torch.logsumexp(gam * cos, 2) - math.log(self.K)) / gam if self.K > 1 else cos[:, :, 0]
        if y is not None and self.cfg.get('margin', 0):
            cc = cc - self.cfg['margin'] * F.one_hot(y, self.C)
        return self.cfg.get('scale', 16.0) * cc

    def diversity(self):
        if self.K < 2: return 0.0
        p = F.normalize(self.protos, dim=-1); g = torch.einsum('ckd,cjd->ckj', p, p)
        off = g - torch.eye(self.K, device=g.device)[None]
        return off.clamp_min(0).pow(2).mean()

    def forward(self, x, state=None):
        f = self.backbone(resize(x, self.size))
        s3 = self.p3(f[2].flatten(2).transpose(1, 2)); s4 = self.p4(f[3].flatten(2).transpose(1, 2))
        pos = self.pos
        if s4.shape[1] != pos.shape[1]:
            g = int(math.sqrt(s4.shape[1]))
            pos = F.interpolate(pos.reshape(1, self.g4, self.g4, -1).permute(0, 3, 1, 2), (g, g), mode='bilinear').flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(x), -1, -1), s4 + pos], 1)
        for b in self.blocks: t = b(t, s3)
        e = self.norm(t[:, 0])
        y = state.get('y') if state else None
        ce, outs = [], []
        if self.K:
            lp = self.proto_logits(e, y); ce.append((lp, 1.0)); outs.append(self.proto_logits(e))
        if self.linear is not None:
            ll = self.linear(e); ce.append((ll, 1.0)); outs.append(ll)
        logits = torch.stack(outs).mean(0)
        reg = self.cfg.get('div_weight', .01) * self.diversity() if self.K else 0.0
        return dict(logits=logits, ce=ce, emb=e, reg=reg)

    def classifier_params(self):
        return ([self.protos] if self.K else []) + (list(self.linear.parameters()) if self.linear is not None else [])


# ---------------------------------------------------------------------------------------------- wrapper
class NovelNet(nn.Module):
    """core model + optional Design-A projectors (feature KD to a teacher embedding, SupCon embedding)."""
    def __init__(self, n_classes, cfg, teacher_dim=None):
        super().__init__()
        self.core = {'global': GlobalNet, 'zoom': ZoomNeXt, 'proto': ProtoFormer}[cfg['arch']](n_classes, cfg)
        d = self.core.emb_dim
        self.kd = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, teacher_dim)) if teacher_dim else None
        self.sc = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128)) if cfg.get('supcon') else None

    def forward(self, x, state=None):
        o = self.core(x, state)
        if self.kd is not None: o['kd'] = self.kd(o['emb'].float())
        if self.sc is not None: o['sc'] = F.normalize(self.sc(o['emb'].float()), dim=-1)
        return o

    @property
    def backbone(self): return self.core.backbone

    def classifier_params(self): return self.core.classifier_params()
