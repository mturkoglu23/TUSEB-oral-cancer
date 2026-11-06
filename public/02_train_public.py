"""Train/evaluate the pretrained models in MODELS on a public dataset with its fixed 5-fold split. Resumable.
Same models, hyper-parameters, augmentation, two-phase schedule and checkpoint rule as ../02_train_cv.py.

  python 02_train_public.py --datasets OCD3_2class OCD2_2class NDO_2class NDO_3class NDO_4class
  python 02_train_public.py --datasets OCD3_2class --models ResNet50 --folds 0 --smoke   # pipeline check

A (model, fold) is complete only when its done.json exists; anything else is retrained from scratch.
Failures are written to error.txt and the run continues with the next fold.
"""
import argparse, json, time, traceback, datetime, platform
import numpy as np, pandas as pd, torch, torch.nn as nn, timm
import torchvision.transforms.v2 as T
from sklearn.metrics import f1_score
from common_public import *
from metrics_public import fold_metrics

ap = argparse.ArgumentParser()
ap.add_argument('--datasets', nargs='*', default=PUBLISHED)
ap.add_argument('--models', nargs='*', default=list(MODELS))
ap.add_argument('--folds', nargs='*', type=int, default=None)
ap.add_argument('--smoke', action='store_true', help='2+2 epochs, outputs to results_public/<dataset>/smoke')
ap.add_argument('--seed', type=int, default=SEED, help='training seed (folds unchanged); != 42 -> <dataset>/seeds/s<seed>/')
args = ap.parse_args()

dev = torch.device('cuda')
CFG = dict(TRAIN)
if args.smoke:
    CFG.update(head_max_epochs=2, ft_max_epochs=2)
DSN = OUT = folds = IMGS = Y = TASK_CLASSES = None  # set per dataset in the main loop


class DS(torch.utils.data.Dataset):
    def __init__(self, idx, tf): self.idx, self.tf = idx, tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return self.tf(torch.from_numpy(np.array(IMGS[j]))), Y[j]


def transforms(size, mean, std):
    norm = [T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)]
    train = T.Compose([T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.18), antialias=True),
                       T.RandomHorizontalFlip(), T.RandomRotation(15),
                       T.ColorJitter(0.15, 0.15, 0.10, 0.02)] + norm)
    ev = T.Compose([T.Resize((size, size), antialias=True)] + norm)
    return train, ev


@torch.no_grad()
def predict(model, loader):
    model.eval(); P, L = [], []
    for x, y in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x.to(dev, non_blocking=True))
        P.append(torch.softmax(out.float(), 1).cpu()); L.append(y)
    return torch.cat(P).numpy(), torch.cat(L).numpy()


def run_phase(model, params, lr, max_ep, patience, tl, vl, crit, phase, log, best, freeze_backbone):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=CFG['weight_decay'])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=CFG['plateau_factor'],
                                                     patience=CFG['plateau_patience'], min_lr=CFG['min_lr'])
    wait = 0
    for ep in range(1, max_ep + 1):
        model.train()
        if freeze_backbone:  # frozen backbone keeps its ImageNet BatchNorm statistics
            for mod in model.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm): mod.eval()
        tl_sum = tc = n = 0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(x); loss = crit(out.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tl_sum += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); n += len(y)
        pv, yv = predict(model, vl)
        vloss = nn.functional.nll_loss(torch.log(torch.from_numpy(pv).clamp_min(1e-8)), torch.from_numpy(yv).long(), weight=crit.weight.cpu()).item()  # same weighted CE as training
        vacc = float((pv.argmax(1) == yv).mean()); vf1 = f1_score(yv, pv.argmax(1), average='macro', zero_division=0)
        lr_now = opt.param_groups[0]['lr']; sch.step(vloss)
        improved = vloss < best['val_loss'] - 1e-4
        log.append(dict(phase=phase, epoch=ep, global_epoch=len(log) + 1, lr=lr_now, train_loss=tl_sum / n, train_acc=tc / n,
                        val_loss=vloss, val_acc=vacc, val_f1_macro=vf1, best=improved))
        if improved:  # checkpoint = best validation loss over both phases
            best.update(val_loss=vloss, val_acc=vacc, val_f1_macro=vf1, phase=phase, epoch=ep, global_epoch=len(log),
                        state={k: v.detach().clone() for k, v in model.state_dict().items()})
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
    return ep


def run_fold(name, k):
    tid, size = MODELS[name]
    d = OUT / 'trained_models' / name / f'fold_{k:02d}'
    d.mkdir(parents=True, exist_ok=True)
    if (d / 'done.json').exists(): print('skip', name, k); return
    for f in ('error.txt',):
        if (d / f).exists(): (d / f).unlink()
    seed_everything(args.seed)  # one fixed seed for every model and fold
    role = folds[f'role_k{k}'].values
    te, va, tr = (np.where(role == r)[0] for r in ('test', 'val', 'train'))
    assert not (set(folds.group[tr]) & set(folds.group[te]) or set(folds.group[tr]) & set(folds.group[va])
                or set(folds.group[va]) & set(folds.group[te]))

    model = timm.create_model(tid, pretrained=True, num_classes=len(TASK_CLASSES), drop_rate=CFG['drop_rate']).to(dev)
    dc = timm.data.resolve_data_config({}, model=model)
    ttf, etf = transforms(size, dc['mean'], dc['std'])
    g = torch.Generator(); g.manual_seed(args.seed)
    tl = torch.utils.data.DataLoader(DS(tr, ttf), batch_size=CFG['batch_size'], shuffle=True, generator=g, drop_last=True)
    vl = torch.utils.data.DataLoader(DS(va, etf), batch_size=32)
    tel = torch.utils.data.DataLoader(DS(te, etf), batch_size=32)

    cnt = np.bincount(Y[tr], minlength=len(TASK_CLASSES))
    w = torch.tensor(len(tr) / (len(TASK_CLASSES) * cnt), dtype=torch.float32, device=dev)  # 'balanced' weights, train split only
    crit = nn.CrossEntropyLoss(weight=w)
    log, best = [], dict(val_loss=np.inf)
    t0 = time.time()
    # phase 1: classification head only
    head = list(model.get_classifier().parameters())
    for p in model.parameters(): p.requires_grad = False
    for p in head: p.requires_grad = True
    run_phase(model, head, CFG['head_lr'], CFG['head_max_epochs'], CFG['head_patience'], tl, vl, crit, 'head', log, best, True)
    # phase 2: full fine-tuning, starting from the best head-phase weights
    model.load_state_dict(best['state'])
    for p in model.parameters(): p.requires_grad = True
    run_phase(model, list(model.parameters()), CFG['ft_lr'], CFG['ft_max_epochs'], CFG['ft_patience'], tl, vl, crit, 'finetune', log, best, False)
    train_time = time.time() - t0

    # single test evaluation with the validation-selected weights
    model.load_state_dict(best['state'])
    pt, yt = predict(model, tel)
    pv, yv = predict(model, vl)  # kept for validation-only threshold selection in the analysis
    # latency: batch 1, same precision as evaluation (GPU is shared with other jobs -> indicative only)
    model.eval(); x1 = torch.randn(1, 3, size, size, device=dev); ts = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(60):
            torch.cuda.synchronize(); t = time.perf_counter(); model(x1); torch.cuda.synchronize()
            if i >= 10: ts.append((time.perf_counter() - t) * 1000)

    pd.DataFrame(log).to_csv(d / 'training_log.csv', index=False)
    torch.save({kk: v.half() if v.is_floating_point() else v for kk, v in best['state'].items()}, d / 'best_weights_fp16.pt')
    (OUT / 'predictions' / name).mkdir(parents=True, exist_ok=True)
    for split, ix, pp in (('test', te, pt), ('val', va, pv)):
        pred = folds.iloc[ix][['file', 'group', 'patient_id', 'label', 'orig_class']].copy()
        pred['y'] = Y[ix]; pred['y_pred'] = pp.argmax(1)
        for i, c in enumerate(TASK_CLASSES): pred[f'p_{c}'] = pp[:, i]
        pred.to_csv(OUT / 'predictions' / name / f'fold_{k:02d}_{split}_predictions.csv', index=False)
    met = fold_metrics(yt, pt, TASK_CLASSES)
    info = dict(dataset=DSN, classes=TASK_CLASSES, model=name, timm_id=tid, input_size=size, fold=k, n_folds=DATASETS[DSN]['k'],
                n_train=len(tr), n_val=len(va), n_test=len(te), train_class_counts=cnt.tolist(), class_weights=w.tolist(),
                best_phase=best['phase'], best_epoch=best['epoch'], best_global_epoch=best['global_epoch'],
                epochs_run=len(log), best_val_loss=best['val_loss'], best_val_acc=best['val_acc'], best_val_f1_macro=best['val_f1_macro'],
                train_time_s=train_time, latency_ms_bs1_median=float(np.median(ts)),
                params_total=sum(p.numel() for p in model.parameters()), mean=dc['mean'], std=dc['std'],
                config=CFG, torch=torch.__version__, timm=timm.__version__, gpu=torch.cuda.get_device_name(0),
                finished=datetime.datetime.now().isoformat(timespec='seconds'), smoke=args.smoke, test_metrics=met)
    json.dump(info, open(d / 'done.json', 'w'), indent=2, default=float)
    print(f"{DSN} {name} fold {k}: epochs {len(log)} best {best['phase']}@{best['epoch']} {train_time/60:.1f} min | "
          f"test acc {met['accuracy']:.3f} f1M {met['f1_macro']:.3f} auc {met['roc_auc']:.3f}", flush=True)
    del model; torch.cuda.empty_cache()


for DSN in args.datasets:
    OUT = out_dir(DSN) / 'smoke' if args.smoke else out_dir(DSN)
    if args.seed != SEED: OUT = OUT / 'seeds' / f's{args.seed}'  # seed-robustness reruns kept apart
    folds = load_split(DSN)
    IMGS = load_cache(DSN, folds.file)
    Y = folds['y'].values; TASK_CLASSES = DATASETS[DSN]['classes']
    for name in args.models:
        for k in (args.folds if args.folds is not None else range(DATASETS[DSN]['k'])):
            try:
                run_fold(name, k)
            except Exception:
                d = OUT / 'trained_models' / name / f'fold_{k:02d}'; d.mkdir(parents=True, exist_ok=True)
                (d / 'error.txt').write_text(datetime.datetime.now().isoformat() + '\n' + traceback.format_exc(), encoding='utf-8')
                print('FAILED', DSN, name, k, traceback.format_exc().splitlines()[-1], flush=True)
                torch.cuda.empty_cache()
