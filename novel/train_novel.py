"""Train / evaluate the novel ConvNeXt-Tiny designs (ablation rows in configs_novel.py) on the benchmark's 5-fold
splits with the fixed seed 42. Resumable: a (task, row, fold) is complete only when its done.json exists.

  python train_novel.py --tasks ours_3class --rows A1 A2c
  python train_novel.py --tasks OCD3_2class --rows B4 --folds 0 --smoke        # pipeline check -> results/smoke/
  python train_novel.py --tasks ours_binary --rows A6 --set lr=5e-5 --tag lr5e-5

Schedule per fold: LP (frozen backbone, new modules only) -> full fine-tuning with layer-wise LR decay and cosine
schedule (+ optional SAM / EMA) -> optional cRW (classifier re-trained with class-balanced sampling).
Checkpoint = lowest class-weighted validation loss (FT / cRW epochs; for ZoomNeXt only once the zoom branch is on).
Then, on the validation fold only: TTA on/off ('auto'), temperature scaling, binary Youden threshold.
"""
import argparse, json, time, traceback, datetime, copy
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from timm.optim import param_groups_layer_decay
from timm.utils import ModelEmaV3
from common_novel import Task, RES, SEED, N_FOLDS, HI, TEACHER_CACHE, seed_everything
from configs_novel import ROWS
from models_novel import NovelNet
from train_utils import (ClsLoss, GpuAug, supcon, val_loss, class_aware_batches, balanced_batches, plain_batches,
                         SAM, fit_temperature, youden_threshold, ece)
from metrics_public import fold_metrics

ap = argparse.ArgumentParser()
ap.add_argument('--tasks', nargs='+', required=True)
ap.add_argument('--rows', nargs='+', required=True)
ap.add_argument('--folds', nargs='*', type=int, default=list(range(N_FOLDS)))
ap.add_argument('--set', nargs='*', default=[], help='config overrides key=value (python literal)')
ap.add_argument('--tag', default='', help='suffix of the output folder when --set is used')
ap.add_argument('--smoke', action='store_true', help='LP 1 + FT 2 (+ cRW 1) epochs, outputs to results/smoke')
ap.add_argument('--save-weights', action='store_true')
ap.add_argument('--seed', type=int, default=SEED, help='training seed (folds unchanged); != 42 -> row folder <row>_s<seed>')
ap.add_argument('--nested', type=int, default=None, metavar='OUTER',
                help='inner loop of nested CV: outer test fold OUTER is left out; inner fold i (0-3) tests on the i-th '
                     'remaining fold, validates on the next one, trains on the other two -> results/nested/outer<O>/')
args = ap.parse_args()
dev = torch.device('cuda')


def parse_set(items):
    import ast
    out = {}
    for s in items:
        k, v = s.split('=', 1)
        try: out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError): out[k] = v
    return out


def batch(imgs, idx):
    return torch.from_numpy(np.stack([imgs[j] for j in idx])).to(dev, non_blocking=True)


class Runner:
    def __init__(self, task, row, cfg, out):
        self.task, self.row, self.cfg, self.out = task, row, cfg, out
        self.imgs = task.images(); self.y = task.y
        self.masks = task.masks() if cfg.get('mask_weight') else None

    # -------------------------------------------------------------------------------- forward helpers
    def forward(self, model, x, y=None, state=None):
        st = dict(state or {}); st['y'] = y
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model(x, st)

    @torch.no_grad()
    def predict(self, model, idx, flip=False, zoom_on=True):
        model.eval(); L = []
        for i in range(0, len(idx), 32):
            x = self.aug.eval(batch(self.imgs, idx[i:i + 32]))
            lg = self.forward(model, x, state=dict(zoom_on=zoom_on))['logits'].float()
            if flip: lg = (lg + self.forward(model, x.flip(3), state=dict(zoom_on=zoom_on))['logits'].float()) / 2
            L.append(lg.cpu())
        return torch.cat(L).numpy()

    # -------------------------------------------------------------------------------- one epoch
    def epoch(self, model, opt, batches, ep_state, sam=None, ema=None, sched=None, train_backbone=True):
        c = self.cfg; model.train()
        if not train_backbone: model.core.backbone.net.eval()
        tot = n = 0
        for b in batches:
            x8 = batch(self.imgs, b); y = torch.as_tensor(self.y[b], device=dev)
            m8 = batch(self.masks, b) if self.masks is not None else None
            x, m = self.aug(x8, m8)
            if sched: sched()

            def loss_fn():
                o = self.forward(model, x, y, ep_state)
                loss = sum(w * self.crit(lg, y) for lg, w in o['ce']) + o['reg']
                if ep_state.get('kd_w') and 'kd' in o:
                    t = torch.as_tensor(self.temb[b], device=dev)
                    loss = loss + ep_state['kd_w'] * (1 - F.cosine_similarity(o['kd'], t, dim=-1)).mean()
                if c['supcon'] and 'sc' in o and ep_state.get('phase') != 'lp':
                    loss = loss + c['supcon'] * supcon(o['sc'], y, c['supcon_tau'])
                if self.soft is not None and ep_state.get('phase') == 'ft':
                    st = torch.as_tensor(self.soft[b], device=dev)
                    loss = loss + 4 * F.kl_div(F.log_softmax(o['logits'].float() / 2, 1), st, reduction='batchmean')
                if self.masks is not None and 'att' in o:
                    a = o['att']; tgt = F.adaptive_avg_pool2d(m, a.shape[-2:]); has = tgt.flatten(1).amax(1) > 0
                    if has.any(): loss = loss + c['mask_weight'] * F.binary_cross_entropy(a[has].clamp(1e-4, 1 - 1e-4), tgt[has])
                return loss

            loss = loss_fn(); opt.zero_grad(set_to_none=True); loss.backward()
            if sam is not None:
                sam.ascend(); opt.zero_grad(set_to_none=True); loss_fn().backward(); sam.descend()
            opt.step()
            if ema is not None: ema.update(model)
            if self.extra is not None and ep_state.get('kd_w'):  # unlabeled public images: KD loss only (own data)
                j = self.rng.choice(len(self.extra[0]), len(b), replace=False)
                xe, _ = self.aug(batch(self.extra[0], j))
                o = self.forward(model, xe, None, ep_state)
                le = ep_state['kd_w'] * (1 - F.cosine_similarity(o['kd'], torch.as_tensor(self.extra[1][j], device=dev), dim=-1)).mean()
                opt.zero_grad(set_to_none=True); le.backward(); opt.step()
                if ema is not None: ema.update(model)
            tot += loss.item() * len(b); n += len(b)
        return tot / max(n, 1)

    def validate(self, model, zoom_on):
        lv = self.predict(model, self.va, zoom_on=zoom_on)
        pv = torch.softmax(torch.as_tensor(lv), 1).numpy()
        return val_loss(pv, self.y[self.va], self.crit.w), pv

    # -------------------------------------------------------------------------------- fold
    def run(self, k):
        c = self.cfg; d = self.out / f'fold_{k:02d}'; d.mkdir(parents=True, exist_ok=True)
        if (d / 'done.json').exists(): print('skip', self.task.name, self.row, k); return
        if (d / 'error.txt').exists(): (d / 'error.txt').unlink()
        seed_everything(args.seed); self.rng = np.random.default_rng(args.seed)
        tr, va, te = self.task.split(k); self.va = va; Y = self.y
        C = self.task.n_classes; bs = c['batch_size']

        # teacher embeddings (label-free) and optional unlabeled extra corpus for our own data
        self.temb = self.extra = self.soft = None; tdim = None
        if c['teacher']:
            self.temb = np.load(TEACHER_CACHE / f'{self.task.cache_key}_{c["teacher"]}.npy'); tdim = self.temb.shape[1]
            use_extra = c['kd_extra'] is True or (c['kd_extra'] == 'auto' and self.task.name.startswith('ours'))
            if use_extra:
                ex = Task('NDO_4class'); self.extra = (ex.images(), np.load(TEACHER_CACHE / f'{ex.cache_key}_{c["teacher"]}.npy'))
            if c['logit_kd']:  # teacher linear probe; soft targets for training images are out-of-fold within the training split
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import cross_val_predict
                self.soft = np.zeros((len(Y), C), np.float32)
                lr = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')
                self.soft[tr] = cross_val_predict(lr, self.temb[tr], Y[tr], cv=5, method='predict_proba') ** 0.5
                self.soft[tr] /= self.soft[tr].sum(1, keepdims=True)  # softened with temperature T=2

        model = NovelNet(C, c, tdim).to(dev)
        pc = model.core.backbone.cfg
        self.aug = GpuAug(HI, c['recipe'], pc['mean'], pc['std'], c['sog'])
        cnt = np.bincount(Y[tr], minlength=C)
        E = c['ft_epochs']; lp_E = c['lp_epochs']; crw_E = c['crw_epochs'] if c['crw'] else 0
        if args.smoke: E, lp_E, crw_E = 2, 1, min(crw_E, 1)
        self.crit = ClsLoss(c['loss'], cnt, c['label_smoothing'], drw_epoch=int(0.6 * E)).to(dev)
        zoom = c['arch'] == 'zoom' and c.get('zoom')
        zoom_start = int(round(c.get('zoom_start_frac', 0) * E)) if zoom else 0
        log, best = [], dict(val_loss=np.inf)
        bb_params = set(id(p) for p in model.core.backbone.net.parameters())
        new_params = [p for p in model.parameters() if id(p) not in bb_params]
        t0 = time.time()

        def make_batches():
            if c['supcon']: return [tr[b] for b in class_aware_batches(Y[tr], bs, 2, self.rng)]  # >= 2 per class for SupCon
            return [tr[b] for b in plain_batches(len(tr), bs, self.rng)]

        def keep(phase, ep, vl, pv, m):
            log[-1]['best'] = True
            best.update(val_loss=vl, phase=phase, epoch=ep, state={kk: v.detach().cpu().clone() for kk, v in m.state_dict().items()})

        # ---- phase 1: linear probe (frozen backbone; trains the new modules)
        for p in model.core.backbone.net.parameters(): p.requires_grad = False
        opt = torch.optim.AdamW(new_params, lr=c['lp_lr'], weight_decay=1e-4)
        for ep in range(1, lp_E + 1):
            tl = self.epoch(model, opt, make_batches(), dict(phase='lp', zoom_on=False, kd_w=c['kd_weight'] if c['teacher'] else 0),
                            train_backbone=False)
            vl, _ = self.validate(model, zoom_on=False)
            log.append(dict(phase='lp', epoch=ep, train_loss=tl, val_loss=vl, best=False))

        # ---- phase 2: full fine-tuning, layer-wise LR decay, warmup + cosine
        for p in model.parameters(): p.requires_grad = True
        groups = param_groups_layer_decay(model.core.backbone.net, weight_decay=c['weight_decay'], layer_decay=c['layer_decay'])
        groups.append(dict(params=new_params, weight_decay=c['weight_decay'], lr_scale=1.0))
        opt = torch.optim.AdamW(groups, lr=c['lr'])
        n_steps = E * max(1, len(tr) // bs); warm = c['warmup_epochs'] * max(1, len(tr) // bs); step = [0]

        def sched():
            s = step[0]; step[0] += 1
            f = s / max(1, warm) if s < warm else c['min_lr'] / c['lr'] + (1 - c['min_lr'] / c['lr']) * 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, n_steps - warm)))
            for g in opt.param_groups: g['lr'] = c['lr'] * f * g.get('lr_scale', 1.0)
        sam = None
        if c['sam']:
            sam_ids = set(id(p) for p in new_params) | set(id(p) for s in model.core.backbone.net.stages[2:] for p in s.parameters())
            sam = SAM([p for p in model.parameters() if id(p) in sam_ids], c['sam'])
        # EMA horizon capped at ~10 % of the FT steps: with 0.999 and ~540 steps (our data) the EMA would still be
        # mostly the LP-end weights at the end of training
        ema = ModelEmaV3(model, decay=min(c['ema'], 1 - 10 / n_steps)) if c['ema'] else None
        for ep in range(1, E + 1):
            kd_w = c['kd_weight'] if c['teacher'] and ep <= int((1 - c['kd_anneal_frac']) * E) else 0.0
            zoom_on = bool(zoom) and ep > zoom_start
            self.crit.epoch = ep
            tl = self.epoch(model, opt, make_batches(), dict(phase='ft', zoom_on=zoom_on, kd_w=kd_w), sam=sam, ema=ema, sched=sched)
            m_eval = ema.module if ema is not None else model
            vl, pv = self.validate(m_eval, zoom_on=zoom_on or not zoom)
            log.append(dict(phase='ft', epoch=ep, train_loss=tl, val_loss=vl, kd_w=kd_w, zoom_on=zoom_on, best=False))
            if (not zoom or zoom_on) and vl < best['val_loss'] - 1e-4: keep('ft', ep, vl, pv, m_eval)

        # ---- phase 3: classifier re-training with class-balanced sampling (cRW)
        model.load_state_dict(best['state'])
        if crw_E:
            for p in model.parameters(): p.requires_grad = False
            cp = model.classifier_params()
            for p in cp: p.requires_grad = True
            opt = torch.optim.AdamW(cp, lr=c['crw_lr'], weight_decay=1e-4)
            plain = ClsLoss('ce', cnt).to(dev); wcrit = self.crit; best_c = dict(val_loss=np.inf)
            for ep in range(1, crw_E + 1):
                self.crit = plain
                tl = self.epoch(model, opt, [tr[b] for b in balanced_batches(Y[tr], bs, self.rng)],
                                dict(phase='crw', zoom_on=bool(zoom)), train_backbone=False)
                self.crit = wcrit
                vl, pv = self.validate(model, zoom_on=True)
                log.append(dict(phase='crw', epoch=ep, train_loss=tl, val_loss=vl, best=False))
                if vl < best_c['val_loss'] - 1e-4:
                    best_c.update(val_loss=vl); keep('crw', ep, vl, pv, model)
            model.load_state_dict(best['state'])
        train_time = time.time() - t0

        # ---- validation-fold decisions: TTA, temperature, threshold; then one test evaluation
        lv0, lt0 = self.predict(model, va), self.predict(model, te)
        use_tta = False
        if c['tta'] in ('auto', True):
            lv1 = self.predict(model, va, flip=True)
            if c['tta'] is True or val_loss(torch.softmax(torch.as_tensor(lv1), 1).numpy(), Y[va], self.crit.w) < \
                    val_loss(torch.softmax(torch.as_tensor(lv0), 1).numpy(), Y[va], self.crit.w):
                use_tta = True; lv0 = lv1; lt0 = self.predict(model, te, flip=True)
        T = fit_temperature(lv0, Y[va]) if c['calibrate'] else 1.0
        pv = torch.softmax(torch.as_tensor(lv0) / T, 1).numpy(); pt = torch.softmax(torch.as_tensor(lt0) / T, 1).numpy()
        classes = self.task.classes
        met = fold_metrics(Y[te], pt, classes)
        info = dict(task=self.task.name, row=self.row, fold=k, val_fold=(k + 1) % N_FOLDS, classes=classes, cfg=c,
                    n_train=len(tr), n_val=len(va), n_test=len(te), train_class_counts=cnt.tolist(),
                    best_phase=best['phase'], best_epoch=best['epoch'], best_val_loss=best['val_loss'],
                    tta=use_tta, temperature=T, train_time_s=train_time,
                    params_total=sum(p.numel() for p in model.parameters()),
                    ece_test=ece(pt, Y[te]), brier_test=float(((pt - np.eye(len(classes))[Y[te]]) ** 2).sum(1).mean()),
                    finished=datetime.datetime.now().isoformat(timespec='seconds'), smoke=args.smoke, test_metrics=met)
        if len(classes) == 2 and c['threshold'] == 'youden':
            thr = youden_threshold(pv[:, 1], Y[va])
            q = pt[:, 1]; lo = np.log(np.clip(q, 1e-7, 1 - 1e-7) / np.clip(1 - q, 1e-7, 1)) - np.log(thr / (1 - thr))
            ps = 1 / (1 + np.exp(-lo)); info.update(threshold=thr, test_metrics_thr=fold_metrics(Y[te], np.stack([1 - ps, ps], 1), classes))
        for split, ix, pp, lg in (('test', te, pt, lt0), ('val', va, pv, lv0)):
            pred = pd.DataFrame(dict(file=np.array(self.task.files)[ix], group=self.task.group[ix], label=self.task.labels[ix], y=Y[ix],
                                     y_pred=pp.argmax(1)))
            for i, cl in enumerate(classes): pred[f'p_{cl}'] = pp[:, i]; pred[f'logit_{cl}'] = lg[:, i]
            pred.to_csv(d / f'{split}_predictions.csv', index=False)
        pd.DataFrame(log).to_csv(d / 'training_log.csv', index=False)
        if args.save_weights:
            torch.save({kk: v.half() if v.is_floating_point() else v for kk, v in best['state'].items()}, d / 'best_weights_fp16.pt')
        json.dump(info, open(d / 'done.json', 'w'), indent=2, default=float)
        s = f"{self.task.name} {self.row} fold {k}: best {best['phase']}@{best['epoch']} {train_time/60:.1f} min | " \
            f"acc {met['accuracy']:.3f} f1M {met['f1_macro']:.3f} auc {met['roc_auc']:.3f}"
        if 'test_metrics_thr' in info:
            mt = info['test_metrics_thr']; s += f" | thr {info['threshold']:.2f} sens {mt['sensitivity']:.3f} spec {mt['specificity']:.3f}"
        print(s, flush=True)
        del model; torch.cuda.empty_cache()


overrides = parse_set(args.set)
for tname in args.tasks:
    task = Task(tname)
    if args.nested is not None:
        rem = [f for f in range(N_FOLDS) if f != args.nested]; fo = task.frame['fold'].values
        task._role = lambda i, rem=rem, fo=fo: np.where(fo == rem[i], 'test', np.where(fo == rem[(i + 1) % len(rem)], 'val',
                                                         np.where(fo == args.nested, 'none', 'train')))
        if args.folds == list(range(N_FOLDS)): args.folds = list(range(len(rem)))
    for row in args.rows:
        cfg = dict(ROWS[row]); cfg.update(overrides)
        if cfg.get('mask_weight') and not tname.startswith('NDO'):
            print('skip', row, 'on', tname, '(lesion masks exist for NDO only)'); continue
        name = row + (f'_{args.tag}' if args.tag else '') + (f'_s{args.seed}' if args.seed != SEED else '')
        out = (RES / 'smoke' if args.smoke else RES) / tname / name
        if args.nested is not None: out = (RES / 'smoke' if args.smoke else RES) / 'nested' / f'outer{args.nested}' / tname / name
        r = Runner(task, name, cfg, out)
        for k in args.folds:
            try:
                r.run(k)
            except Exception:
                d = out / f'fold_{k:02d}'; d.mkdir(parents=True, exist_ok=True)
                (d / 'error.txt').write_text(datetime.datetime.now().isoformat() + '\n' + traceback.format_exc(), encoding='utf-8')
                print('FAILED', tname, name, k, traceback.format_exc().splitlines()[-1], flush=True)
                torch.cuda.empty_cache()
