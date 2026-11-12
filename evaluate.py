"""
Validation evaluation in the competition's terms: Panoptic Quality (PQ), scored per filament.

Per validation image:
  model (optionally averaged over the 8 flips/rotations with --tta) -> probability map (resize_to x resize_to) -> upsample to the original 2048x2048
        -> threshold -> connected components (each component = one predicted filament)
        -> optionally group pieces lying within ~close_size pixels into one filament (only the predicted
           pixels are kept; the gap between them is not filled)
        -> drop filaments smaller than min_area pixels (at 2048 scale)
        -> optionally drop filaments whose mean probability is below --min-conf
Ground truth: every annotator copy of the image is scored separately (one instance per annotated filament).
Matching: a predicted and a ground-truth filament match when IoU > 0.5 (unique by construction, as in PQ).
  PQ = sum(IoU of matches) / (TP + 0.5*FP + 0.5*FN)    SQ = mean IoU of matches    RQ = TP / (TP + 0.5*FP + 0.5*FN)
Also reported: mean Dice of the whole-image masks, one-to-many (a GT filament touched by >1 prediction =
fragmentation) and many-to-one (a prediction touching >1 GT filament = over-merging), and 'split(raw)':
GT filaments touched by >1 piece *before* grouping and size filtering (fragmentation the filter hides).
For the best setting, recall (share of real filaments matched) is also shown per filament size group, plus an
error breakdown: false positives by type (fragment of a found filament / poor outline of a missed one / no
overlap with any filament), by size and by position on the disk; recall by position; and the confidence
(mean probability) of correct vs false predictions. Disk position comes from a circle fitted to each image.
Probability maps can be saved once (--save-probs) and re-scored later without the model (--load-probs);
giving several folders to --load-probs averages them (ensemble).
--save-overlays writes one picture per annotator copy at the best setting: matched predictions (TP) in blue,
false positives in red, missed filaments (FN) in yellow dashed outlines; plus a matching <name>_gt.jpg with
that annotator's filaments in green, for side-by-side comparison.

Usage (from the project folder):
    python evaluate.py                                # newest checkpoint in checkpoints/, threshold 0.5, min area 0
    python evaluate.py --checkpoint checkpoints/<run name>.pt
    python evaluate.py --threshold 0.4 --close 31 --min-area 250
    python evaluate.py --sweep                        # grid of thresholds x grouping x min areas, best by PQ
    python evaluate.py --limit 10                     # quick check on the first 10 validation images
    python evaluate.py --sweep --tta                  # test-time augmentation: average of 8 flipped/rotated predictions
    python evaluate.py --save-probs probs/<run name>  # also save the probability maps (8-bit PNG, gitignored)
    python evaluate.py --sweep --load-probs probs/<run name>          # re-score saved maps, no model needed
    python evaluate.py --sweep --load-probs probs/<run A> probs/<run B>  # ensemble: average of two models
    python evaluate.py --sweep --load-probs probs/<run name> --save-overlays overlays/<run name>   # TP/FP/FN pictures
    python evaluate.py --sweep --min-conf 0 0.9 0.92 0.94   # also try confidence cut-offs (each value multiplies the sweep)
"""
import os
#same workaround as main.py: torch and cv2 each load their own OpenMP runtime on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import glob
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

import dataset
import resnet_unet
import tiling
import preprocess

IMAGE_DIR="./MAGFiLO_1.0_Kaggle_2026/train/train_images"
ANNOTATION_FILE="./MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
SWEEP_THRESHOLDS=[0.2,0.3,0.4,0.5]
SWEEP_CLOSE=[0,31,61,91,121]            #grouping distance in pixels at 2048x2048 (0 = no grouping)
SWEEP_MIN_AREAS=[0,100,250,500,1000]
#filament size groups for recall, by ground-truth area in pixels at 2048x2048 (val set: 230 / 698 / 523 / 142 filaments)
SIZE_EDGES=[500,1500,5000]
SIZE_NAMES=["<500 px","500-1500 px","1500-5000 px",">=5000 px"]
#position on the solar disk, as distance from the disk centre / radius of the visible limb (preprocess.find_limb)
ZONE_EDGES=[0.7,0.9,1.0]
ZONE_NAMES=["centre <0.7R","0.7-0.9R","limb 0.9-1.0R","off-disk >1R"]
#overlay colours (BGR for OpenCV): reference palette blue / red / yellow, distinguishable with colour blindness
OVERLAY_TP=(214,120,42)    #2a78d6 matched prediction
OVERLAY_FP=(72,73,227)     #e34948 false positive
OVERLAY_FN=(0,161,237)     #eda100 missed filament (dashed)
OVERLAY_GT=(122,175,27)    #1baf7a ground truth (in the separate _gt picture)
OVERLAY_SIZE=1024
FP_TYPES=[("fp_fragment","fragment of a found filament"),("fp_poor","overlaps a missed filament (IoU <= 0.5)"),
          ("fp_none","no overlap with any filament")]


def default_checkpoint():
    """Newest best-model checkpoint in checkpoints/ (one per training run; *_last.pt resume files are skipped);
    falls back to best_model.pt from older runs."""
    runs=[p for p in glob.glob("checkpoints/*.pt") if not p.endswith("_last.pt")]
    return max(runs,key=os.path.getmtime) if runs else "best_model.pt"


def load_model(checkpoint, device):
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    #architecture from the run config saved in the checkpoint; older checkpoints are the plain U-Net
    config=state.get("config") or {}
    model=resnet_unet.build_model(config.get("model","unet"),pretrained=False)   #weights come from the checkpoint
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    return model.to(device).eval(), config


def predict(model, image, device, tta=False, tile=None, overlap=128):
    """Probability map for one image (1, H, W), in overlapping tiles if tile is set (see tiling.py).
    With tta, the model sees the image in all 8 orientations (4 rotations x mirrored or not, the same set
    used for training augmentation); each prediction is turned back to the original orientation and the
    8 are averaged."""
    x=image.unsqueeze(0).to(device)
    run=lambda z: tiling.predict_tiled(model,z,tile,overlap)
    with torch.no_grad():
        if not tta:
            return run(x)[0,0].float().cpu().numpy()
        total=torch.zeros_like(x[:, :1])
        for mirror in (False,True):
            xm=x.flip(-1) if mirror else x
            for k in range(4):
                p=run(torch.rot90(xm,k,dims=(-2,-1)))
                p=torch.rot90(p,-k,dims=(-2,-1))   #undo the rotation...
                total+=p.flip(-1) if mirror else p #...then the mirroring
        return (total/8)[0,0].float().cpu().numpy()


def gt_label_map(coco, img_id):
    """Ground-truth instance map at full size: 0 = background, k = k-th annotated filament."""
    info=coco.imgs[img_id]
    labels=np.zeros((int(info['height']),int(info['width'])),dtype=np.int32)
    anns=coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    for k,ann in enumerate(anns,start=1):
        m=coco.annToMask(ann).astype(bool)
        labels[m & (labels==0)]=k   #filaments rarely overlap; a shared pixel stays with the first one
    return labels, len(anns)


def zone_map(shape, cx, cy, r):
    """Per-pixel disk zone index (see ZONE_EDGES)."""
    yy,xx=np.mgrid[0:shape[0],0:shape[1]]
    return np.digitize(np.hypot(xx-cx,yy-cy)/r,ZONE_EDGES).astype(np.int64)


def label_stats(labels, n, zone, prob):
    """For labels 0..n: the disk zone holding most of its pixels, and its mean probability."""
    nz=len(ZONE_NAMES)
    counts=np.bincount((labels.astype(np.int64)*nz+zone).ravel(),minlength=(n+1)*nz).reshape(n+1,nz)
    area=np.maximum(counts.sum(axis=1),1)
    conf=np.bincount(labels.ravel(),weights=prob.ravel(),minlength=n+1)/area
    return counts.argmax(axis=1), conf


def components(binary, close_size=0):
    """
    Predicted filaments from a 0/1 mask -> (label map, area of each label).
    close_size=0: each connected piece (8-connectivity) is one filament.
    close_size>0: pieces that a morphological closing with a close_size-pixel disk would join are grouped
    into one filament, but only the original predicted pixels are kept (no bridging pixels added).
    """
    binary=binary.astype(np.uint8)
    if close_size>0:
        disk=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(close_size,close_size))
        joined=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,disk)
        n,labels=cv2.connectedComponents(joined,connectivity=8)
        labels=labels*binary                     #keep the original pixels only
        return labels, np.bincount(labels.ravel(),minlength=n)
    n,labels,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=8)
    return labels, stats[:,cv2.CC_STAT_AREA]


def split_count(gt, n_gt, raw, n_raw):
    """Number of GT filaments touched by more than one raw predicted piece."""
    inter=np.bincount((gt.astype(np.int64)*(n_raw+1)+raw).ravel(),minlength=(n_gt+1)*(n_raw+1)).reshape(n_gt+1,n_raw+1)
    return int(((inter[1:,1:]>0).sum(axis=1)>1).sum())


def drop_small(labels, areas, min_area):
    """Remove components smaller than min_area (and empty ones) and renumber the rest 1..n.
    Also returns the old label of each new label (index 0 = background)."""
    keep=np.zeros(len(areas),dtype=np.int32)
    old=[0]
    for i in range(1,len(areas)):
        if areas[i]>0 and areas[i]>=min_area:
            old.append(i)
            keep[i]=len(old)-1
    return keep[labels], len(old)-1, np.array(old)


def mean_conf(pred, n, prob):
    """Mean probability inside each label 0..n."""
    return np.bincount(pred.ravel(),weights=prob.ravel(),minlength=n+1)/np.maximum(np.bincount(pred.ravel(),minlength=n+1),1)


def drop_unconfident(pred, n, conf, min_conf):
    """Remove labels whose mean probability (conf, per label 0..n) is below min_conf and renumber the rest 1..m.
    Also returns the old label of each new label (index 0 = background)."""
    old=np.concatenate([[0],np.where(conf[1:]>=min_conf)[0]+1])
    keep=np.zeros(n+1,dtype=np.int32)
    keep[old]=np.arange(len(old),dtype=np.int32)
    return keep[pred], len(old)-1, old


def match(gt, n_gt, pred, n_pred, gt_zone=None, pred_zone=None, pred_conf=None):
    """Compare GT and predicted instance maps (same size) and return the counts behind PQ, Dice, etc.
    gt_zone / pred_zone / pred_conf (per label, index 0 = background) add the error breakdown."""
    #all pairwise overlaps in one pass: count pixels per (gt label, pred label) combination
    inter=np.bincount((gt.astype(np.int64)*(n_pred+1)+pred).ravel(),
                      minlength=(n_gt+1)*(n_pred+1)).reshape(n_gt+1,n_pred+1)
    gt_area=inter.sum(axis=1)[1:]
    pred_area=inter.sum(axis=0)[1:]
    I=inter[1:,1:]
    U=gt_area[:,None]+pred_area[None,:]-I
    iou=np.where(U>0,I/np.maximum(U,1),0.0)
    matched=iou>0.5
    tp=int(matched.sum())
    overlaps=I>0
    fg_both=int(I.sum())
    fg_sum=int(gt_area.sum()+pred_area.sum())
    #recall per size group: how many GT filaments of each size exist, and how many were matched
    found=matched.any(axis=1)
    group=np.digitize(gt_area,SIZE_EDGES)
    by_size={}
    for g in range(len(SIZE_NAMES)):
        by_size[f"gt_size{g}"]=int((group==g).sum())
        by_size[f"found_size{g}"]=int((found&(group==g)).sum())
    #error breakdown: what kind of prediction each false positive is, its size and where it is on the disk
    is_fp=~matched.any(axis=0)
    touches=overlaps[:,is_fp]
    frag=touches[found].any(axis=0)                  #overlaps a filament that another prediction found
    poor=~frag&touches.any(axis=0)                   #overlaps only filaments nobody found
    by_size["fp_fragment"]=int(frag.sum())
    by_size["fp_poor"]=int(poor.sum())
    by_size["fp_none"]=int((~touches.any(axis=0)).sum())
    fp_group=np.digitize(pred_area[is_fp],SIZE_EDGES)
    for g in range(len(SIZE_NAMES)):
        by_size[f"fp_size{g}"]=int((fp_group==g).sum())
    if gt_zone is not None and pred_zone is not None:
        for z in range(len(ZONE_NAMES)):
            by_size[f"gt_zone{z}"]=int((gt_zone[1:]==z).sum())
            by_size[f"found_zone{z}"]=int((found&(gt_zone[1:]==z)).sum())
            by_size[f"fp_zone{z}"]=int((pred_zone[1:][is_fp]==z).sum())
    if pred_conf is not None:
        by_size["conf_tp"]=list(pred_conf[1:][~is_fp])
        by_size["conf_fp"]=list(pred_conf[1:][is_fp])
    return dict(**by_size,
        tp=tp,
        fp=n_pred-int(matched.any(axis=0).sum()),
        fn=n_gt-int(matched.any(axis=1).sum()),
        sum_iou=float(iou[matched].sum()),
        n_gt=n_gt, n_pred=n_pred,
        one_to_many=int((overlaps.sum(axis=1)>1).sum()),   #GT filaments split over >1 prediction
        many_to_one=int((overlaps.sum(axis=0)>1).sum()),   #predictions covering >1 GT filament
        dice=2*fg_both/fg_sum if fg_sum>0 else 1.0,
    )


def outcomes(gt, n_gt, pred, n_pred):
    """Which predictions matched a filament (IoU > 0.5) and which filaments were found: two boolean arrays."""
    inter=np.bincount((gt.astype(np.int64)*(n_pred+1)+pred).ravel(),
                      minlength=(n_gt+1)*(n_pred+1)).reshape(n_gt+1,n_pred+1)
    I=inter[1:,1:]
    U=inter.sum(axis=1)[1:,None]+inter.sum(axis=0)[None,1:]-I
    matched=np.where(U>0,I/np.maximum(U,1),0.0)>0.5
    return matched.any(axis=0), matched.any(axis=1)


def draw_outlines(canvas, labels, ids, color, scale, dashed=False):
    """Outline the given labels of a full-size label map on a (smaller) canvas."""
    for k in ids:
        contours,_=cv2.findContours((labels==k).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in contours:
            pts=np.round(c[:,0,:]*scale).astype(np.int32)
            if not dashed:
                cv2.polylines(canvas,[pts],True,color,2,cv2.LINE_AA)
                continue
            for i in range(0,len(pts),16):   #dash: draw 8 contour points, skip 8 (in full-size pixels)
                cv2.polylines(canvas,[pts[i:i+9]],False,color,2,cv2.LINE_AA)


def overlay_canvas(image, header):
    """Image shrunk to OVERLAY_SIZE as a colour canvas, with a black header box of text lines. Returns (canvas, scale)."""
    scale=OVERLAY_SIZE/image.shape[1]
    canvas=cv2.cvtColor(cv2.resize(image,(OVERLAY_SIZE,OVERLAY_SIZE),interpolation=cv2.INTER_AREA),cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canvas,(0,0),(OVERLAY_SIZE,52),(0,0,0),-1)
    for i,text in enumerate(header):
        cv2.putText(canvas,text,(10,20+22*i),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    return canvas, scale


def save_gt_picture(path, image, gt, n_gt, title):
    """Ground truth of one annotator copy (green outlines), to compare with its overlay picture."""
    canvas,scale=overlay_canvas(image,[title,f"Ground truth: {n_gt} annotated filaments (green)"])
    draw_outlines(canvas,gt,range(1,n_gt+1),OVERLAY_GT,scale)
    cv2.imwrite(path,canvas,[cv2.IMWRITE_JPEG_QUALITY,90])


def save_overlay(path, image, gt, n_gt, pred, n_pred, title):
    """Picture of one annotator copy: TP predictions blue, FP red, missed filaments yellow dashed."""
    tp,found=outcomes(gt,n_gt,pred,n_pred)
    canvas,scale=overlay_canvas(image,[title,f"TP {int(tp.sum())} (blue)   FP {int((~tp).sum())} (red)   "
                                             f"FN {int((~found).sum())} (yellow, dashed)"])
    draw_outlines(canvas,gt,[k+1 for k in np.where(~found)[0]],OVERLAY_FN,scale,dashed=True)
    draw_outlines(canvas,pred,[k+1 for k in np.where(tp)[0]],OVERLAY_TP,scale)
    draw_outlines(canvas,pred,[k+1 for k in np.where(~tp)[0]],OVERLAY_FP,scale)
    cv2.imwrite(path,canvas,[cv2.IMWRITE_JPEG_QUALITY,90])


def summarise(results):
    """Pooled PQ/SQ/RQ over all validation entries, plus per-image means."""
    tot=defaultdict(float)
    lists=defaultdict(list)
    for r in results:
        for k,v in r.items():
            if isinstance(v,list):
                lists[k]+=v
            else:
                tot[k]+=v
    denom=tot["tp"]+0.5*tot["fp"]+0.5*tot["fn"]
    per_image=[r["sum_iou"]/(r["tp"]+0.5*r["fp"]+0.5*r["fn"]) for r in results if r["n_gt"]+r["n_pred"]>0]
    return dict(
        PQ=tot["sum_iou"]/denom if denom else 0.0,
        SQ=tot["sum_iou"]/tot["tp"] if tot["tp"] else 0.0,
        RQ=tot["tp"]/denom if denom else 0.0,
        PQ_mean_per_image=float(np.mean(per_image)) if per_image else 0.0,
        mean_dice=float(np.mean([r["dice"] for r in results])),
        TP=int(tot["tp"]), FP=int(tot["fp"]), FN=int(tot["fn"]),
        gt_per_image=tot["n_gt"]/len(results), pred_per_image=tot["n_pred"]/len(results),
        one_to_many=int(tot["one_to_many"]), many_to_one=int(tot["many_to_one"]), split_raw=int(tot["split_raw"]),
        entries=len(results),
        recall=tot["tp"]/(tot["tp"]+tot["fn"]) if tot["tp"]+tot["fn"] else 0.0,
        #(size group, number of GT filaments, number matched, recall)
        recall_by_size=[(name,int(tot[f"gt_size{g}"]),int(tot[f"found_size{g}"]),
                         tot[f"found_size{g}"]/tot[f"gt_size{g}"] if tot[f"gt_size{g}"] else 0.0)
                        for g,name in enumerate(SIZE_NAMES)],
        fp_types=[(name,int(tot[k])) for k,name in FP_TYPES],
        fp_by_size=[(name,int(tot[f"fp_size{g}"])) for g,name in enumerate(SIZE_NAMES)],
        #(zone, GT filaments, found, false positives)
        by_zone=[(name,int(tot[f"gt_zone{z}"]),int(tot[f"found_zone{z}"]),int(tot[f"fp_zone{z}"]))
                 for z,name in enumerate(ZONE_NAMES)],
        conf_tp=np.array(lists["conf_tp"]), conf_fp=np.array(lists["conf_fp"]),
    )


def print_breakdown(s):
    """Error breakdown for one setting (normally the best)."""
    fp=max(s["FP"],1)
    print(f"\nFalse positives by type ({s['FP']} in total)")
    for name,n in s["fp_types"]:
        print(f"{name:>42} {n:>6d} {n/fp:>6.1%}")
    print(f"\nFalse positives by size")
    for name,n in s["fp_by_size"]:
        print(f"{name:>14} {n:>6d} {n/fp:>6.1%}")
    print(f"\nBy position on the disk")
    print(f"{'zone':>14} {'filaments':>9} {'found':>6} {'recall':>6} {'FP':>6}")
    for name,n,found,nfp in s["by_zone"]:
        rec=f"{found/n:>6.3f}" if n else f"{'-':>6}"
        print(f"{name:>14} {n:>9d} {found:>6d} {rec} {nfp:>6d}")
    tp,fpc=s["conf_tp"],s["conf_fp"]
    if len(tp) and len(fpc):
        print(f"\nConfidence (mean probability inside each prediction): median correct {np.median(tp):.2f}, false {np.median(fpc):.2f}")
        print(f"{'drop if below':>14} {'FP removed':>11} {'TP lost':>8}")
        for c in (0.7,0.8,0.85,0.9,0.95):
            print(f"{c:>14.2f} {int((fpc<c).sum()):>11d} {int((tp<c).sum()):>8d}")


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint",default=None,help="default: newest file in checkpoints/, else best_model.pt")
    ap.add_argument("--threshold",type=float,default=0.5)
    ap.add_argument("--close",type=int,default=0,help="group pieces within ~this many pixels (at 2048x2048); 0 = off")
    ap.add_argument("--min-area",type=int,default=0,help="minimum filament size in pixels at 2048x2048")
    ap.add_argument("--min-conf",type=float,nargs="+",default=[0.0],
                    help="drop filaments with a lower mean probability (0 = off); several values are all tried, also with --sweep")
    ap.add_argument("--sweep",action="store_true",help="try a grid of thresholds, grouping distances and min areas")
    ap.add_argument("--resize-to",type=int,default=None,help="default: from the checkpoint's config (1024 for older checkpoints)")
    ap.add_argument("--tile",type=int,default=None,help="predict in overlapping tiles of this size; default: from the checkpoint's config (none = whole image)")
    ap.add_argument("--overlap",type=int,default=None,help="pixels shared by neighbouring tiles; default: from the checkpoint's config, else 128")
    ap.add_argument("--val-fraction",type=float,default=0.2,help="must match training")
    ap.add_argument("--split-seed",type=int,default=42,help="must match training")
    ap.add_argument("--limit",type=int,default=None,help="only the first N validation image files")
    ap.add_argument("--tta",action="store_true",help="average predictions over the 8 flips/rotations (8x slower prediction)")
    ap.add_argument("--save-probs",default=None,help="folder to save each image's probability map in (8-bit PNG, 2048x2048)")
    ap.add_argument("--load-probs",nargs="+",default=None,help="use saved probability maps instead of the model; several folders are averaged")
    ap.add_argument("--save-overlays",default=None,help="folder for TP/FP/FN pictures at the best setting (one per annotator copy)")
    args=ap.parse_args()

    device=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if args.load_probs:
        model,config=None,{}
    else:
        if args.checkpoint is None:
            args.checkpoint=default_checkpoint()
        model,config=load_model(args.checkpoint,device)
    if args.save_overlays and not args.load_probs and not args.save_probs:
        #overlays are drawn in a second pass once the best setting is known: keep the maps so the model runs only once
        args.save_probs=os.path.join(args.save_overlays,"probs")
        print(f"--save-overlays: saving probability maps to {args.save_probs} for the drawing pass")
    if args.save_probs:
        os.makedirs(args.save_probs,exist_ok=True)
    #resolution and tiling as in training, unless given on the command line
    if args.resize_to is None:
        args.resize_to=config.get("resize_to",1024)
    if args.tile is None:
        args.tile=config.get("tile_size")
    if args.overlap is None:
        args.overlap=config.get("tile_overlap",128)
    coco=COCO(ANNOTATION_FILE)
    _,val_ids=dataset.split_by_file(coco,val_fraction=args.val_fraction,seed=args.split_seed)
    val_data=dataset.Dataset(IMAGE_DIR,ANNOTATION_FILE,resize_to=args.resize_to,crop_size=None,img_ids=val_ids,
                             preprocess=config.get("preprocess"))   #same preprocessing as in training

    #group annotator copies by image file: predict once per file, score against every copy
    by_file=defaultdict(list)
    for idx,img_id in enumerate(val_data.img_ids):
        by_file[coco.imgs[img_id]['file_name']].append(idx)
    files=sorted(by_file)[:args.limit]

    thresholds=SWEEP_THRESHOLDS if args.sweep else [args.threshold]
    closes=SWEEP_CLOSE if args.sweep else [args.close]
    min_areas=SWEEP_MIN_AREAS if args.sweep else [args.min_area]
    results=defaultdict(list)   #(threshold, close, min_area) -> one result per validation entry

    if args.load_probs:
        print(f"Evaluating saved probability maps from {', '.join(args.load_probs)}"
              f"{' (averaged)' if len(args.load_probs)>1 else ''} on {len(files)} validation images "
              f"({sum(len(by_file[f]) for f in files)} annotator copies)")
    else:
        print(f"Evaluating {args.checkpoint} on {len(files)} validation images "
              f"({sum(len(by_file[f]) for f in files)} annotator copies) on {device}, "
              f"resolution {args.resize_to}, tiles {f'{args.tile} (overlap {args.overlap})' if args.tile else 'off'}, "
              f"TTA {'on (8 orientations)' if args.tta else 'off'}")
    start=time.perf_counter()
    for n,fname in enumerate(files,start=1):
        idxs=by_file[fname]
        gts=[gt_label_map(coco,val_data.img_ids[i]) for i in idxs]
        h,w=gts[0][0].shape
        if args.load_probs:
            stem=os.path.splitext(fname)[0]
            prob=np.mean([cv2.imread(os.path.join(d,f"{stem}.png"),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
                          for d in args.load_probs],axis=0)
        else:
            image,_=val_data[idxs[0]]
            prob=predict(model,image,device,args.tta,args.tile,args.overlap)
            prob=cv2.resize(prob,(w,h),interpolation=cv2.INTER_LINEAR)   #back to full resolution
            if args.save_probs:
                cv2.imwrite(os.path.join(args.save_probs,f"{os.path.splitext(fname)[0]}.png"),
                            np.round(prob*255).astype(np.uint8))
        #position on the disk, from the raw image
        zone=zone_map((h,w),*preprocess.find_limb(cv2.imread(os.path.join(IMAGE_DIR,fname),cv2.IMREAD_GRAYSCALE)))
        gt_zones=[label_stats(gt,n_gt,zone,np.zeros((h,w)))[0] for gt,n_gt in gts]
        for t in thresholds:
            raw,raw_areas=components(prob>t)
            splits=[split_count(gt,n_gt,raw,len(raw_areas)-1) for gt,n_gt in gts]
            for c in closes:
                labels,areas=(raw,raw_areas) if c==0 else components(prob>t,c)
                label_zone,label_conf=label_stats(labels,len(areas)-1,zone,prob)
                for a in min_areas:
                    pred,n_pred,old=drop_small(labels,areas,a)
                    for m in args.min_conf:
                        sel=old
                        p,n_p=pred,n_pred
                        if m>0:
                            p,n_p,kept=drop_unconfident(pred,n_pred,label_conf[old],m)
                            sel=old[kept]
                        for (gt,n_gt),gz,split in zip(gts,gt_zones,splits):
                            r=match(gt,n_gt,p,n_p,gz,label_zone[sel],label_conf[sel])
                            r["split_raw"]=split
                            results[(t,c,a,m)].append(r)
        if n%20==0 or n==len(files):
            print(f"  {n}/{len(files)} images, {time.perf_counter()-start:.0f}s")

    rows=sorted(((t,c,a,m,summarise(r)) for (t,c,a,m),r in results.items()),key=lambda x:-x[4]["PQ"])
    print(f"\n{'thresh':>6} {'close':>5} {'min_area':>8} {'min_conf':>8} {'PQ':>6} {'SQ':>6} {'RQ':>6} {'PQ/img':>6} {'Dice':>6} "
          f"{'TP':>5} {'FP':>5} {'FN':>5} {'GT/img':>6} {'pred/img':>8} {'1-to-many':>9} {'many-to-1':>9} {'split(raw)':>10}")
    for t,c,a,m,s in rows:
        print(f"{t:>6.2f} {c:>5d} {a:>8d} {m:>8.2f} {s['PQ']:>6.3f} {s['SQ']:>6.3f} {s['RQ']:>6.3f} {s['PQ_mean_per_image']:>6.3f} "
              f"{s['mean_dice']:>6.3f} {s['TP']:>5d} {s['FP']:>5d} {s['FN']:>5d} {s['gt_per_image']:>6.1f} "
              f"{s['pred_per_image']:>8.1f} {s['one_to_many']:>9d} {s['many_to_one']:>9d} {s['split_raw']:>10d}")
    t,c,a,m,s=rows[0]
    print(f"\nBest: threshold {t}, grouping {c} px, min area {a}, min confidence {m} -> PQ {s['PQ']:.3f} (SQ {s['SQ']:.3f} x RQ {s['RQ']:.3f}), "
          f"mean Dice {s['mean_dice']:.3f}, over {s['entries']} annotator copies")
    print(f"\nRecall by filament size (best setting; all sizes: {s['recall']:.3f})")
    print(f"{'size':>14} {'filaments':>9} {'found':>6} {'recall':>6}")
    for name,n,found,rec in s["recall_by_size"]:
        print(f"{name:>14} {n:>9d} {found:>6d} {rec:>6.3f}" if n else f"{name:>14} {n:>9d} {found:>6d} {'-':>6}")
    print_breakdown(s)

    if args.save_overlays:
        os.makedirs(args.save_overlays,exist_ok=True)
        folders=args.load_probs or [args.save_probs]
        for fname in files:
            stem=os.path.splitext(fname)[0]
            prob=np.mean([cv2.imread(os.path.join(d,f"{stem}.png"),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
                          for d in folders],axis=0)
            labels,areas=components(prob>t,c)
            pred,n_pred,_=drop_small(labels,areas,a)
            if m>0:
                pred,n_pred,_=drop_unconfident(pred,n_pred,mean_conf(pred,n_pred,prob),m)
            image=cv2.imread(os.path.join(IMAGE_DIR,fname),cv2.IMREAD_GRAYSCALE)
            idxs=by_file[fname]
            for k,i in enumerate(idxs,start=1):
                gt,n_gt=gt_label_map(coco,val_data.img_ids[i])
                suffix=f"_a{k}" if len(idxs)>1 else ""
                save_overlay(os.path.join(args.save_overlays,f"{stem}{suffix}.jpg"),image,gt,n_gt,pred,n_pred,
                             f"{stem}{suffix}  (threshold {t}, grouping {c} px, min area {a} px, min confidence {m})")
                annotator=val_data.img_ids[i].split("-",1)[0]   #annotation batch id, e.g. 050103
                save_gt_picture(os.path.join(args.save_overlays,f"{stem}{suffix}_gt.jpg"),image,gt,n_gt,
                                f"{stem}{suffix}  (annotation set {annotator}, copy {k} of {len(idxs)})")
        print(f"\nOverlays at the best setting written to {args.save_overlays} "
              f"({sum(len(by_file[f]) for f in files)} pictures, each with a matching _gt picture)")


if __name__=="__main__":
    main()
