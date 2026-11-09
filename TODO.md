# Solar Filaments – To-Do Checklist

_Last updated: 2026-11-09. Each code change gets approval before editing._

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

## Phase 0 – Make it run
- [ ] Remove `pdb.set_trace()` + shape prints (unet.py ×2, train_class.py ×1)
- [ ] `ConvBlock`: pass `padding=padding` by keyword (currently lands in `stride`, so padding=0); drop the interpolate hack
- [ ] Fix channel typo 126 → 128
- [ ] Scheduler: pass one (e.g. ReduceLROnPlateau) or guard `None`
- [ ] Define the validation metric (`acc` is undefined)
- [ ] Move `best_val_loss` init outside the epoch loop
- [ ] Accumulate `loss.item()` not the tensor
- [ ] Save best model with `copy.deepcopy` / state_dict, not a reference
- [ ] Single device-selection logic (main.py vs TrainBox disagree)
- [ ] Re-run and confirm the `Fatal Python error` is gone (suspects: pdb, duplicate OpenMP on macOS)

## Phase 1 – Data correctness
- [ ] Build masks at 2048×2048 (from `coco.imgs[id]` height/width), then resize/crop
- [ ] Rasterise **every** filament of the image and merge them into one mask (current `frPyObjects` + `.squeeze(-1)` path is fragile; use `coco.annToMask` or `cv2.fillPoly` per filament, OR-combined)
- [ ] Also keep an **instance map** (filament id per pixel) alongside the binary mask – needed for per-filament metrics
- [ ] Training samples: keep each annotator copy as its own sample (allowed by Kaggle) – baseline approach
- [ ] _Optional later:_ label fusion across copies (union / majority vote / soft label = fraction of annotators; match filaments by IoU first) and compare against the baseline
- [ ] Seeded train/val split **grouped by file_name** (all copies of an image on the same side), optionally grouped by date; stratify by filaments-per-image
- [ ] Validation: score each val image against all of its annotator copies (mean), mirroring the likely official protocol
- [ ] Collapse the 4 categories to one binary class
- [ ] Load images as single-channel grayscale (already `convert("L")` – keep it)
- [ ] Crops must not drop filaments silently: bias random crops toward regions containing filaments
- [ ] Replace ImageNet normalisation with dataset mean/std (or per-image / limb-darkening normalisation)
- [ ] Resolution strategy: 256 erases thin filaments → 512–1024 input or random 512 crops from 2048
- [ ] Update `annotations.py` viewer: colour each filament separately, option to show all annotator copies side by side

## Phase 2 – Training that fits the problem
- [ ] Output logits (remove sigmoid) + `BCEWithLogitsLoss` + Dice loss for 0.3% foreground
- [ ] Watch small filaments: whole-image Dice is dominated by large ones → consider per-filament weighting or boundary/clDice terms later
- [ ] Validation metrics, pixel-level: IoU, Dice, precision, recall (replace argmax "accuracy" in `eval()`)
- [ ] Validation metrics, **per-filament**: connected components → match to GT filaments by IoU → hit rate, miss rate, false positives, AP@IoU, split by filament size
- [ ] Augmentations: rotations/flips, brightness/contrast, (disk masking)
- [ ] MLflow: `set_experiment` / `start_run`, log params + config
- [ ] Config file (lr, batch, img size, epochs) + fixed seeds

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
