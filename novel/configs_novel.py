"""Ablation rows of the three designs.

Every row gets the same budget (LP 5 epochs + FT 30 epochs, batch 16) and the same validation-fold rules
(checkpoint = lowest class-weighted validation loss; temperature, binary threshold and TTA on/off chosen on the
validation fold only). A0 is the existing benchmark run (results/ and public/results_public/, ConvNeXtTiny).

Rows that build on "the best previous row" use BEST_INIT / BEST_TEACHER / B_FROM / C_FROM below. They hold the
expected winners; set them from the A2 / A3-A4 / Design-A / Design-B results before running the later rows.
"""
FINAL_ROW = 'A6'              # the single proposed model (decided 2026-11-03); binary task = 3-class output collapsed
BEST_INIT = 'in12k_ft_in1k'   # winner of A2a/b/c (A2b: best or tied on NDO_2class, OCD2, OCD3; best AUC on our data)
BEST_TEACHER = 'dinov3_vitl'  # winner of A3/A4* (MedSigLIP needs gated Hugging Face access; see README)

A1 = dict(
    arch='global', init='in1k', size=384, drop_path=0.1, head_init_scale=1e-3, att=None,
    recipe='strong', sog=True,                       # RandAugment(2, 9) + colour jitter + random erasing; Shades-of-Gray
    batch_size=16, lp_epochs=5, lp_lr=1e-3,          # LP: frozen backbone, new modules only
    ft_epochs=30, warmup_epochs=2, lr=1e-4, min_lr=1e-6, weight_decay=0.05, layer_decay=0.8,
    loss='wce', label_smoothing=0.1,
    teacher=None, kd_weight=1.0, kd_anneal_frac=0.3, kd_extra='auto', logit_kd=False,
    supcon=0.0, supcon_tau=0.07,
    crw=False, crw_epochs=10, crw_lr=1e-3,
    sam=0.0, ema=0.0,
    tta='auto', calibrate=True, threshold='youden',
)


def row(base, **kw):
    c = dict(base); c.update(kw); return c


A_FULL = row(A1, init=BEST_INIT, teacher=BEST_TEACHER, supcon=0.2, crw=True)
# Designs B and C were run before the rest of Design A (decided 2026-11-02), so they build on the A1 recipe with
# the A2 winner init; B0 = C0 = A2b (not re-run: use --ref A2b in analyze_novel.py).
B_FROM = row(A1, init=BEST_INIT)
B_BASE = row(B_FROM, arch='zoom', multiscale=True, zoom=None, zoom_start_frac=0.25,
             zoom_expand=1.3, zoom_min_side=0.35, fusion='gate', zoom_source='hires')
B_FULL = row(B_BASE, att='coord', zoom='att')
C_FROM = row(B_FROM, supcon=0.0)                      # C0 = A2b (C6 adds SupCon on the head embedding)
C_BASE = row(C_FROM, arch='proto', depth=2, cross=False, protos=0, linear_head=True, margin=0.0, scale=16.0, div_weight=0.01)

ROWS = {
    # ---------------- Design A: FM-distilled ConvNeXt with contrastive re-balancing
    'A1':  A1,
    'A2a': row(A1, init='in22k_ft_in1k'),
    'A2b': row(A1, init='in12k_ft_in1k'),
    'A2c': row(A1, init='dinov3'),
    'A3':  row(A1, init=BEST_INIT, teacher='medsiglip'),
    'A4b': row(A1, init=BEST_INIT, teacher='dinov2_vitl'),
    'A4c': row(A1, init=BEST_INIT, teacher='dinov3_vitl'),
    'A5':  row(A1, init=BEST_INIT, teacher=BEST_TEACHER, supcon=0.2),
    'A6':  A_FULL,                                    # full Design A
    'A7':  row(A_FULL, kd_extra=False),               # own data only: without the unlabeled public KD images
    # factorial controls (internal review 2026-11-03): SupCon alone, KD+SupCon without the extra images, cRW alone
    'A4s': row(A1, init=BEST_INIT, supcon=0.2),
    'A5x': row(A1, init=BEST_INIT, teacher=BEST_TEACHER, supcon=0.2, kd_extra=False),
    'A2r': row(A1, init=BEST_INIT, crw=True),
    'A8':  row(A_FULL, logit_kd=True),
    'A9a': row(A_FULL, loss='ldam_drw', label_smoothing=0.0),
    'A9b': row(A_FULL, loss='logit_adj'),
    # final single-model candidates (decided 2026-11-03, after stage 4): DINOv3 init (A2c won NDO_4class) combined
    # with DINOv3 ViT-L feature KD + SupCon; A10 without cRW (cRW cut binary sensitivity on our data), A11 with it
    # A10 ran on ours_3class only (macro F1 0.505, far below A6); stage 5 was then stopped and A11 never finished
    'A10': row(A1, init='dinov3', teacher=BEST_TEACHER, supcon=0.2),
    'A11': row(A1, init='dinov3', teacher=BEST_TEACHER, supcon=0.2, crw=True),
    # ---------------- Design B: Look-Twice ZoomNeXt
    'B0':  B_FROM,
    'B1':  B_BASE,                                    # + multi-scale GeM fusion
    'B2':  row(B_BASE, att='coord'),
    'B2s': row(B_BASE, att='simam'),
    'B3':  row(B_BASE, att='coord', zoom='random'),   # compute / augmentation control
    'B4':  B_FULL,                                    # full Design B (attention-guided zoom)
    'B5':  row(B_FULL, zoom_separate=True),
    'B6':  row(B_FULL, fusion='avg'),
    'B7':  row(B_FULL, zoom_source='resized'),
    'B8':  row(B_FULL, mask_weight=0.5),              # NDO only (lesion masks)
    # ---------------- Design C: ProtoFormer head
    'C0':  C_FROM,
    'C1':  C_BASE,
    'C2':  row(C_BASE, cross=True),
    'C3':  row(C_BASE, cross=True, protos=1),
    'C4':  row(C_BASE, cross=True, protos=3),
    'C5':  row(C_BASE, cross=True, protos=3, margin=0.15),
    'C6':  row(C_BASE, cross=True, protos=3, margin=0.15, supcon=0.1),
    'C7':  row(C_BASE, cross=True, protos=3, margin=0.15, supcon=0.1, sam=0.05),
    'C8':  row(C_BASE, cross=True, protos=3, margin=0.15, supcon=0.1, sam=0.05, ema=0.999),
    'C9':  row(C_BASE, cross=True, protos=3, margin=0.15, supcon=0.1, sam=0.05, ema=0.999, linear_head=False),
}
