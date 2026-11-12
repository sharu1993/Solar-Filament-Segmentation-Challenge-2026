# Solar Filaments – To-Do Checklist

_Last updated: 2026-11-11 (after the ResNet-34 run and its PQ evaluation). Each code change gets approval before editing._

## Dataset facts (Kaggle Data tab + train JSON analysis)
- **Images:** 2048×2048, 8-bit **grayscale** JPEG (converted from FITS) – do not treat as RGB.
- **File names:** `YYYYMMDDHHMMSSII`, e.g. `20140609195854Bh` = 2014-06-09 19:58:54, Big Bear (`Bh`). 6 stations in train: B, C, L, M, T, U.
- **Multiple filaments per image:** 8,199 filaments over 1,154 entries – min 1, median 7, max 26 per entry. Filaments rarely overlap each other.
- **Multiple annotators per image:** image `id` = `<annotator batch>-<file name>`, e.g. `010401-…`, `010402-…`, `010403-…` are three independent annotations of the same file. 1,154 entries → **707 unique files** (411 annotated once, 145 twice, 151 three times). Each copy is a complete set of that image's filaments.
  - Kaggle guidance: *"You may treat those images simply as 'different' images."* → training on every copy is allowed.
  - Annotators disagree: 84% of filaments matched between copies, matched IoU median ~0.60, chirality agreement 65%. Useful reference point, **not** a hard ceiling (a model can beat individual annotators).
- **Annotation format:** COCO-style; `pycocotools` recommended. One closed polygon per filament (Kaggle text says "RLE" but the file holds plain polygons), coords in 2048 space, `iscrowd` always 0. `image_id` is a string (docs say int).
- **Size:** foreground ≈ 0.3% of pixels (median); filament bbox median ~75 px; thinnest ~3 px.
- **Categories (chirality):** Left 2,535 / Right 2,590 / Unidentifiable 3,074 / Ambiguous 0. Ground truth for the competition is the segmentation → collapse to one "filament" class.
- Each annotation also has a `spine` polyline (single connected path; possible auxiliary target).
- **Test:** 180 unlabeled images, no file overlap with train, but median 1.5 days from the nearest train image (filaments persist across days). Years skewed to 2011–2017.
- **Metric (Overview tab, updated Aug 7 2026):** leaderboard = **Panoptic Quality (PQ)**, a per-filament score: PQ = Σ IoU(matched pairs) / (#TP + 0.5·#FP + 0.5·#FN). Judging rubric: 70% quantitative (PQ, mean Dice via `torchmetrics.segmentation.DiceScore`, Dice/IoU distributions, **one-to-many / many-to-one** fragmentation and over-merging), 30% qualitative (pipeline description, predicted morphology, code quality); computational efficiency also counts. Organisers: *any PQ > 0.35 is of great value*.
- **Submission format:** one CSV for the whole test set, **one row per predicted filament**: `filament_id` (`<image name>_<n>`) and `segmentation_rle` (pycocotools RLE *counts* only, masks at full **2048×2048**). Predictions are matched to ground truth by overlap, not by index.
- **Rules to respect:** inference uses only the test H-alpha images; models *"may not use any other ground-truth metadata for training"* (our crop placement uses the `spine` – ambiguous, see Phase 1). Official self-evaluation notebook: kaggle.com/code/azimahmadzadeh/self-evaluation-notebook.
- **Final submission (if entered; deadline Nov 15 2026 – not targeted):** 4-page report, public Git repo with `requirements.txt`, a notebook running the whole pipeline, and model **weights in the repo** (current checkpoint 373 MB > GitHub's 100 MB file limit → save weights only / fp16, or Git LFS).

## Results so far (validation IoU, 1024 resolution, file-grouped split, threshold 0.5)
| Change | Best val IoU | Commit | Notes |
|---|---|---|---|
| BCE only (honest split) | 0.391 | 623ae65 | plateau from epoch 4 |
| + BCE + Dice loss | 0.450 | 4f80bcd | stopped at epoch 8, still rising |
| + filament-aware crops | **0.493** | 2332271 | best at epoch 13 (0.464 at epoch 8 vs 0.450) |
| same setup, MLflow run `2026-11-10_06-01_7dd4d8b` | 0.445 | 7dd4d8b | stopped at epoch 5 |
| + augmentation (flips/rot90, brightness/contrast), run `2026-11-11_06-06_e82f99f` | 0.516 | e82f99f | best at epoch 26; levelled off from ~epoch 14; stopped at 28 (VS Code restart) |
| + **ImageNet-pretrained ResNet-34 encoder**, run `2026-11-11_18-34_d1b3d7f` | **0.536** | d1b3d7f | passed 0.516 at epoch 16; best at epoch 47; stopped manually at 50 |

Speed on M1 (MPS, batch 4, 4 workers): plain U-Net ~12.9 min/epoch; ResNet-34 U-Net **~2.8 min/epoch** (24.5M vs 31.0M parameters).
LR schedule (ReduceLROnPlateau, patience 2) halves the LR every ~4 epochs: both runs reach ~6e-5 by epoch 25–28 and then barely improve. Train and val loss stay close (no overfitting).

**Panoptic Quality by model** (`evaluate.py --sweep`, all 141 val images / 217 annotator copies, full 2048 resolution).
| Model (best checkpoint) | Best setting (thr / grouping / min area) | PQ | SQ | RQ | TP / FP / FN | split GT (1-to-many) | recall ≥5000 px |
|---|---|---|---|---|---|---|---|
| U-Net, 5 epochs (7dd4d8b) | 0.4 / 61 / 500 | 0.277 | 0.633 | 0.437 | 680 / 839 / 913 | – | – |
| U-Net + augmentation (e82f99f, epoch 26) | 0.5 / 31 / 500 | 0.346 | 0.652 | 0.532 | 851 / 758 / 742 | 125 | 0.49 |
| **ResNet-34 U-Net (d1b3d7f, epoch 47)** | 0.5 / 31 / 250 | **0.395** | 0.661 | 0.598 | 996 / 742 / 597 | 106 | 0.63 |

Same setting for both later models (0.5 / 31 / 500): FP 758 → 553, split GT 125 → 79, PQ 0.346 → 0.395. The pretrained encoder keeps long filaments in one piece. Misses (FN) now cost more than false positives. Top settings tie within 0.003 → fix 0.5 / 31 / 250 from now on.

Post-processing detail for the first model (7dd4d8b, epoch 5):
| Post-processing (threshold / grouping px / min area px) | PQ | SQ | RQ | TP / FP / FN | pred per image (GT 7.3) |
|---|---|---|---|---|---|
| 0.5 / 0 / 0 (none) | 0.173 | 0.639 | 0.271 | 696 / 2849 / 897 | 16.3 |
| 0.5 / 0 / 500 (size filter) | 0.242 | 0.640 | 0.377 | 606 / 1014 / 987 | 7.5 |
| **0.4 / 61 / 500 (+ grouping)** | **0.277** | 0.633 | 0.437 | 680 / 839 / 913 | 7.0 |

Findings: matched filaments are outlined reasonably (SQ ~0.63) but only 43% of real filaments are matched (RQ is the bottleneck); 451 / 1593 GT filaments are split into several raw pieces; threshold 0.3–0.5 makes little difference; grouping 61 px was the largest tried. Organisers' reference: PQ > 0.35 is "of great value".

## Phase 0 – Make it run
- [x] Remove `pdb.set_trace()` + shape prints (unet.py ×2, train_class.py ×1) — afd12f9
- [x] `ConvBlock`: pass `padding=padding` by keyword (currently lands in `stride`, so padding=0); drop the interpolate hack — afd12f9
- [x] Fix channel typo 126 → 128 — afd12f9
- [x] Scheduler: pass one (e.g. ReduceLROnPlateau) or guard `None` — afd12f9
- [x] Define the validation metric (`acc` is undefined) — afd12f9 (validation IoU)
- [x] Move `best_val_loss` init outside the epoch loop — afd12f9
- [x] Accumulate `loss.item()` not the tensor — afd12f9
- [x] Save best model with `copy.deepcopy` / state_dict, not a reference — afd12f9
- [x] Single device-selection logic (main.py vs TrainBox disagree) — afd12f9
- [x] Re-run and confirm the `Fatal Python error` is gone (suspects: pdb, duplicate OpenMP on macOS) — training runs end to end on MPS

## Phase 1 – Data correctness
- [x] Build masks at 2048×2048 (from `coco.imgs[id]` height/width), then resize/crop — done in 5427174 (resize to 1024 + 512 crop; verified with check_masks.py)
- [x] Rasterise **every** filament of the image and merge them into one mask (current `frPyObjects` + `.squeeze(-1)` path is fragile; use `coco.annToMask` or `cv2.fillPoly` per filament, OR-combined) — done in 5427174 (coco.annToMask)
- [x] Also keep an **instance map** (filament id per pixel) alongside the binary mask – needed for per-filament metrics — 7bb0d4c (`evaluate.gt_label_map`, built at evaluation time)
- [x] Training samples: keep each annotator copy as its own sample (allowed by Kaggle) – baseline approach — already the case: dataset.py uses every image entry (coco.getImgIds)
- [ ] _Optional later:_ label fusion across copies (union / majority vote / soft label = fraction of annotators; match filaments by IoU first) and compare against the baseline
- [x] Seeded train/val split **grouped by file_name** (all copies of an image on the same side) — 623ae65 (80/20, seed 42: 937 / 217 entries, 566 / 141 files)
- [ ] _Optional later:_ group the split by date as well (filaments persist across days); stratify by filaments-per-image
- [x] Validation: score each val image against all of its annotator copies — 623ae65 (every copy is a val entry; whole 1024 image, same pixels every epoch; IoU pooled over all)
- [x] Collapse the 4 categories to one binary class — 5427174 (mask merges all filaments regardless of chirality)
- [x] Load images as single-channel grayscale (`convert("L")`)
- [x] Crops must not drop filaments silently: bias random crops toward regions containing filaments — 2332271 (80% of crops on a filament spine; crops with filament 88% → 98%)
- [ ] Replace ImageNet normalisation with dataset mean/std (or per-image / limb-darkening normalisation)
- [x] Resolution strategy: 256 erases thin filaments → 512–1024 input or random 512 crops from 2048 — 5427174 (2048 → 1024 with area averaging, 512 training crops; validation on whole 1024 image)
- [ ] Update `annotations.py` viewer: colour each filament separately, option to show all annotator copies side by side
- [ ] Rule safety: place filament-aware crops from **mask pixels** instead of the `spine` annotation (same effect, uses only segmentation ground truth)

## Phase 2 – Training that fits the problem
- [x] BCE + Dice loss for 0.3% foreground — 4f80bcd (`losses.BCEDiceLoss`, Dice over the batch; +0.06 IoU)
- [ ] _Optional:_ output logits (remove sigmoid) + `BCEWithLogitsLoss` – more numerically stable, needed for mixed precision
- [ ] Watch small filaments: whole-image Dice is dominated by large ones → consider per-filament weighting or boundary/clDice terms later
- [ ] Validation metrics, pixel-level: IoU done (afd12f9); still missing Dice, precision, recall; `eval()` still uses argmax "accuracy" (dead code)
- [x] Validation metrics, **per-filament**: connected components → match to GT filaments by IoU → hit rate, miss rate, false positives, split by filament size — 7bb0d4c, 5fa7091 (`evaluate.py`; AP@IoU not done)
- [x] Augmentations: rotations/flips, brightness/contrast — e82f99f (+0.02 IoU; disk masking not done)
- [x] MLflow: `set_experiment` / `start_run`, log params + config — 7dd4d8b (`mlflow.db` in project folder, experiment `filament-segmentation`, git commit tags, per-epoch metrics incl. timing)
- [x] Config in one place — 7dd4d8b (`config` dict in main.py, logged to MLflow)
- [ ] Fixed seeds for training (torch / random) so runs are reproducible – only the split is seeded now
- [x] Keep best model by validation IoU — 623ae65
- [x] Early stopping (patience 5, max 30 epochs) + parallel data loading (4 workers) — 8b027f5
- [x] Per-epoch timing (train / validation / total, average, time left) — 87aef99
- [x] Resume from checkpoint (`_last.pt` every epoch, `main.py --resume`, same MLflow run) — 8348da1
- [ ] **LR schedule:** slower decay (patience 4) or cosine schedule – the LR currently runs down within ~25 epochs
- [ ] _On hold:_ faster training – mixed precision on MPS, or Kaggle GPU (free T4/P100)

## Phase 3 – Competition deliverable
- [x] Read the Overview/Evaluation tab: exact metric + submission format — PQ, one CSV row per filament with 2048×2048 RLE counts (see Dataset facts)
- [x] **Local PQ evaluation on the validation set** — 7bb0d4c, 0658881 (`evaluate.py`: upsample to 2048, threshold, connected components, grouping, min-area filter; PQ/SQ/RQ, TP/FP/FN, Dice, one-to-many / many-to-one / raw splits; `--sweep`)
- [x] One checkpoint per training run (`checkpoints/<run name>.pt`) so good models are not overwritten; `evaluate.py` defaults to the newest — 7e99e94
- [x] Longer training run → re-run `evaluate.py --sweep` on its checkpoint — e82f99f (PQ 0.346), d1b3d7f (PQ 0.395)
- [ ] **Test-time augmentation** in `evaluate.py`: average predictions over flips / rot90 (no training needed)
- [ ] Save probability maps once (`--save-probs`) so post-processing experiments don't re-run the model
- [ ] Error breakdown: FPs as fragment of a matched filament / poor outline / no overlap; FNs by size and position on the disk
- [ ] Small filaments (<500 px, recall 0.44): they are halved by the 2048 → 1024 resize → try higher resolution or larger crops
- [ ] Post-processing experiments to raise RQ: grow each detected filament to its full extent (dilation by a few px; hysteresis on the probability map; intensity-based region growing into dark pixels); larger grouping distances (91, 121 px) — tried: larger grouping merges neighbours as much as it fixes splits → join fragments only end to end along the filament direction
- [ ] Inference script for the 180 test images at full resolution (tiling + stitching)
- [ ] Post-processing into individual filaments: connected components, min-area filter, merge fragments, split touching filaments if needed
- [ ] Per-filament confidence score (e.g. mean probability inside the component) for AP ranking
- [ ] Threshold tuning on validation (pixel threshold + min-area) – tune for PQ, not pixel IoU — sweeps done for each model; settings now stable (0.5 / 31 / 250)
- [ ] Submission writer
- [ ] Compare our PQ with the official self-evaluation notebook on the same predictions

## Phase 4 – CV polish
- [x] Pretrained encoder — d1b3d7f (torchvision ResNet-34 + own U-Net decoder, weights in `pretrained/`; PQ 0.346 → 0.395)
- [ ] Other encoders (ResNet-18 / 50, EfficientNet, ConvNeXt) – one change at a time, for an ablation table
- [ ] If connected components can't separate filaments well: compare an instance model (e.g. Mask R-CNN / YOLO-seg)
- [ ] Compare against inter-annotator agreement (reference, not ceiling)
- [ ] README: problem, data, method, results table, example overlays
- [ ] requirements.txt / environment file, unit tests for mask creation (incl. an image with many filaments)
- [ ] Pipeline notebook (load weights → predict → submission CSV); weights small enough for the repo (state_dict only / fp16 or Git LFS)
- [ ] Environment cleanup: only one OpenMP runtime (torch + OpenCV from the same source), then drop the `KMP_DUPLICATE_LIB_OK` workaround in main.py / evaluate.py
- [ ] Ideas: spine as auxiliary target, clDice/skeleton loss for thin structures

## Phase 5 – Solar-specific pretraining (later)
- [ ] Pilot: continue pretraining the ImageNet ResNet-34 on ~2–5k unlabelled GONG H-alpha images (masked-image modelling), then the same fine-tuning recipe (needs download approval; exclude validation dates and nearby hours)
- [ ] Compare random vs ImageNet vs ImageNet + GONG pretraining on IoU and PQ
- [ ] If it helps: scale up (more images, cloud GPU) and/or pseudo-labelling of unlabelled GONG images
