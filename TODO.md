# Solar Filaments – To-Do Checklist

_Last updated: 2026-11-11. Each code change gets approval before editing._

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
- **Metric / submission format:** not on the Data tab → read the Overview/Evaluation tab. Pixel-level (IoU, precision, recall) and per-filament (AP@IoU, hit/miss rate) scores are expected; a public repo suggests Panoptic Quality scored against every annotator copy (unconfirmed).

## Results so far (validation IoU, 1024 resolution, file-grouped split, threshold 0.5)
| Change | Best val IoU | Commit | Notes |
|---|---|---|---|
| BCE only (honest split) | 0.391 | 623ae65 | plateau from epoch 4 |
| + BCE + Dice loss | 0.450 | 4f80bcd | stopped at epoch 8, still rising |
| + filament-aware crops | **0.493** | 2332271 | best at epoch 13 (0.464 at epoch 8 vs 0.450) |
| same setup, MLflow run `2026-11-10_06-01_7dd4d8b` | 0.445 | 7dd4d8b | stopped at epoch 5 |

Speed on M1 (MPS, batch 4, 4 workers): ~12.5 min/epoch (10.6 train + 1.9 val) – GPU-bound, not data-bound.

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
- [ ] Also keep an **instance map** (filament id per pixel) alongside the binary mask – needed for per-filament metrics
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

## Phase 2 – Training that fits the problem
- [x] BCE + Dice loss for 0.3% foreground — 4f80bcd (`losses.BCEDiceLoss`, Dice over the batch; +0.06 IoU)
- [ ] _Optional:_ output logits (remove sigmoid) + `BCEWithLogitsLoss` – more numerically stable, needed for mixed precision
- [ ] Watch small filaments: whole-image Dice is dominated by large ones → consider per-filament weighting or boundary/clDice terms later
- [ ] Validation metrics, pixel-level: IoU done (afd12f9); still missing Dice, precision, recall; `eval()` still uses argmax "accuracy" (dead code)
- [ ] Validation metrics, **per-filament**: connected components → match to GT filaments by IoU → hit rate, miss rate, false positives, AP@IoU, split by filament size
- [ ] Augmentations: rotations/flips, brightness/contrast, (disk masking)
- [x] MLflow: `set_experiment` / `start_run`, log params + config — 7dd4d8b (`mlflow.db` in project folder, experiment `filament-segmentation`, git commit tags, per-epoch metrics incl. timing)
- [x] Config in one place — 7dd4d8b (`config` dict in main.py, logged to MLflow)
- [ ] Fixed seeds for training (torch / random) so runs are reproducible – only the split is seeded now
- [x] Keep best model by validation IoU — 623ae65
- [x] Early stopping (patience 5, max 30 epochs) + parallel data loading (4 workers) — 8b027f5
- [x] Per-epoch timing (train / validation / total, average, time left) — 87aef99
- [ ] _On hold:_ faster training – mixed precision on MPS, or Kaggle GPU (free T4/P100)

## Phase 3 – Competition deliverable
- [ ] Read the Overview/Evaluation tab: exact metric (PQ? MIoU?) + submission format (one mask per image or one entry per filament?)
- [ ] Inference script for the 180 test images at full resolution (tiling + stitching)
- [ ] Post-processing into individual filaments: connected components, min-area filter, merge fragments, split touching filaments if needed
- [ ] Per-filament confidence score (e.g. mean probability inside the component) for AP ranking
- [ ] Threshold tuning on validation (pixel threshold + min-area)
- [ ] Submission writer
- [ ] Local implementation of the competition metric

## Phase 4 – CV polish
- [ ] Pretrained encoder (e.g. segmentation_models_pytorch – needs download approval)
- [ ] If connected components can't separate filaments well: compare an instance model (e.g. Mask R-CNN / YOLO-seg)
- [ ] Compare against inter-annotator agreement (reference, not ceiling)
- [ ] README: problem, data, method, results table, example overlays
- [ ] requirements.txt / environment file, unit tests for mask creation (incl. an image with many filaments)
- [ ] Ideas: spine as auxiliary target, clDice/skeleton loss for thin structures
