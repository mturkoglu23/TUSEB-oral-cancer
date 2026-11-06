"""Cache frozen-teacher image embeddings for Design A (feature distillation). One L2-normalised vector per image,
computed on the clean (un-augmented) image; labels are never used.

  python cache_teacher.py --teacher dinov3_vitl --tasks ours_3class OCD3_2class OCD2_2class NDO_2class NDO_3class NDO_4class

Teachers:
  dinov3_vitl  timm vit_large_patch16_dinov3.lvd1689m (DINOv3 License)
  dinov2_vitl  timm vit_large_patch14_dinov2.lvd142m (Apache-2.0)
  medsiglip    Hugging Face google/medsiglip-448 (gated: accept the HAI-DEF terms on the model page and log in first)
  timm:<id>    any timm model id
Not implemented (no PyTorch weights / manual download): Google Derm Foundation (TensorFlow), PanDerm (CC-BY-NC-ND).
"""
import argparse
import numpy as np, torch, torch.nn.functional as F
from common_novel import Task, TEACHER_CACHE

ap = argparse.ArgumentParser()
ap.add_argument('--teacher', required=True)
ap.add_argument('--tasks', nargs='+', required=True)
ap.add_argument('--bs', type=int, default=32)
args = ap.parse_args()
dev = torch.device('cuda')


def load_teacher(name):
    if name == 'medsiglip':
        from transformers import AutoModel
        m = AutoModel.from_pretrained('google/medsiglip-448').to(dev).eval()
        return (lambda x: m.get_image_features(pixel_values=x)), 448, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    import timm
    tid = {'dinov3_vitl': 'vit_large_patch16_dinov3.lvd1689m', 'dinov2_vitl': 'vit_large_patch14_dinov2.lvd142m'}.get(name)
    tid = tid or name.removeprefix('timm:')
    kw = dict(img_size=224) if 'dinov2' in tid else {}
    m = timm.create_model(tid, pretrained=True, num_classes=0, **kw).to(dev).eval()
    c = m.pretrained_cfg; size = 224 if 'dinov2' in tid else c['input_size'][-1]
    return m, size, c['mean'], c['std']


fn, size, mean, std = load_teacher(args.teacher)
mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1); std = torch.tensor(std, device=dev).view(1, 3, 1, 1)
TEACHER_CACHE.mkdir(parents=True, exist_ok=True)
done = set()
for t in args.tasks:
    task = Task(t)
    out = TEACHER_CACHE / f'{task.cache_key}_{args.teacher}.npy'
    if task.cache_key in done or out.exists():
        print('exists', out); continue
    imgs = task.images(); embs = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(imgs), args.bs):
            x = torch.from_numpy(np.array(imgs[i:i + args.bs])).to(dev).float() / 255
            x = F.interpolate(x, size=(size, size), mode='bilinear', antialias=True, align_corners=False)
            embs.append(F.normalize(fn((x - mean) / std).float(), dim=-1).cpu())
    e = torch.cat(embs).numpy().astype(np.float32)
    np.save(out, e); done.add(task.cache_key)
    print(t, e.shape, '->', out, flush=True)
