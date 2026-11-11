"""
Validation evaluation in the competition's terms: Panoptic Quality (PQ), scored per filament.

Per validation image:
  model -> probability map (resize_to x resize_to) -> upsample to the original 2048x2048
        -> threshold -> connected components (each component = one predicted filament)
        -> optionally group pieces lying within ~close_size pixels into one filament (only the predicted
           pixels are kept; the gap between them is not filled)
        -> drop filaments smaller than min_area pixels (at 2048 scale)
Ground truth: every annotator copy of the image is scored separately (one instance per annotated filament).
Matching: a predicted and a ground-truth filament match when IoU > 0.5 (unique by construction, as in PQ).
  PQ = sum(IoU of matches) / (TP + 0.5*FP + 0.5*FN)    SQ = mean IoU of matches    RQ = TP / (TP + 0.5*FP + 0.5*FN)
Also reported: mean Dice of the whole-image masks, one-to-many (a GT filament touched by >1 prediction =
fragmentation) and many-to-one (a prediction touching >1 GT filament = over-merging), and 'split(raw)':
GT filaments touched by >1 piece *before* grouping and size filtering (fragmentation the filter hides).

Usage (from the project folder):
    python evaluate.py                                # best_model.pt, threshold 0.5, min area 0
    python evaluate.py --threshold 0.4 --close 31 --min-area 250
    python evaluate.py --sweep                        # grid of thresholds x grouping x min areas, best by PQ
    python evaluate.py --limit 10                     # quick check on the first 10 validation images
"""
import argparse
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

import dataset
import unet

IMAGE_DIR="./MAGFiLO_1.0_Kaggle_2026/train/train_images"
ANNOTATION_FILE="./MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
SWEEP_THRESHOLDS=[0.2,0.3,0.4,0.5]
SWEEP_CLOSE=[0,15,31,61]                 #grouping distance in pixels at 2048x2048 (0 = no grouping)
SWEEP_MIN_AREAS=[0,100,250,500,1000]


def load_model(checkpoint, device):
    model=unet.UNet(in_channels=1,out_channels=1)
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    return model.to(device).eval()


def gt_label_map(coco, img_id):
    """Ground-truth instance map at full size: 0 = background, k = k-th annotated filament."""
    info=coco.imgs[img_id]
    labels=np.zeros((int(info['height']),int(info['width'])),dtype=np.int32)
    anns=coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    for k,ann in enumerate(anns,start=1):
        m=coco.annToMask(ann).astype(bool)
        labels[m & (labels==0)]=k   #filaments rarely overlap; a shared pixel stays with the first one
    return labels, len(anns)


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
    """Remove components smaller than min_area (and empty ones) and renumber the rest 1..n."""
    keep=np.zeros(len(areas),dtype=np.int32)
    n=0
    for i in range(1,len(areas)):
        if areas[i]>0 and areas[i]>=min_area:
            n+=1
            keep[i]=n
    return keep[labels], n


def match(gt, n_gt, pred, n_pred):
    """Compare GT and predicted instance maps (same size) and return the counts behind PQ, Dice, etc."""
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
    return dict(
        tp=tp,
        fp=n_pred-int(matched.any(axis=0).sum()),
        fn=n_gt-int(matched.any(axis=1).sum()),
        sum_iou=float(iou[matched].sum()),
        n_gt=n_gt, n_pred=n_pred,
        one_to_many=int((overlaps.sum(axis=1)>1).sum()),   #GT filaments split over >1 prediction
        many_to_one=int((overlaps.sum(axis=0)>1).sum()),   #predictions covering >1 GT filament
        dice=2*fg_both/fg_sum if fg_sum>0 else 1.0,
    )


def summarise(results):
    """Pooled PQ/SQ/RQ over all validation entries, plus per-image means."""
    tot=defaultdict(float)
    for r in results:
        for k,v in r.items():
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
    )


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint",default="best_model.pt")
    ap.add_argument("--threshold",type=float,default=0.5)
    ap.add_argument("--close",type=int,default=0,help="group pieces within ~this many pixels (at 2048x2048); 0 = off")
    ap.add_argument("--min-area",type=int,default=0,help="minimum filament size in pixels at 2048x2048")
    ap.add_argument("--sweep",action="store_true",help="try a grid of thresholds, grouping distances and min areas")
    ap.add_argument("--resize-to",type=int,default=1024,help="must match training")
    ap.add_argument("--val-fraction",type=float,default=0.2,help="must match training")
    ap.add_argument("--split-seed",type=int,default=42,help="must match training")
    ap.add_argument("--limit",type=int,default=None,help="only the first N validation image files")
    args=ap.parse_args()

    device=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    coco=COCO(ANNOTATION_FILE)
    _,val_ids=dataset.split_by_file(coco,val_fraction=args.val_fraction,seed=args.split_seed)
    val_data=dataset.Dataset(IMAGE_DIR,ANNOTATION_FILE,resize_to=args.resize_to,crop_size=None,img_ids=val_ids)

    #group annotator copies by image file: predict once per file, score against every copy
    by_file=defaultdict(list)
    for idx,img_id in enumerate(val_data.img_ids):
        by_file[coco.imgs[img_id]['file_name']].append(idx)
    files=sorted(by_file)[:args.limit]

    thresholds=SWEEP_THRESHOLDS if args.sweep else [args.threshold]
    closes=SWEEP_CLOSE if args.sweep else [args.close]
    min_areas=SWEEP_MIN_AREAS if args.sweep else [args.min_area]
    results=defaultdict(list)   #(threshold, close, min_area) -> one result per validation entry

    model=load_model(args.checkpoint,device)
    print(f"Evaluating {args.checkpoint} on {len(files)} validation images "
          f"({sum(len(by_file[f]) for f in files)} annotator copies) on {device}")
    start=time.perf_counter()
    for n,fname in enumerate(files,start=1):
        idxs=by_file[fname]
        image,_=val_data[idxs[0]]
        with torch.no_grad():
            prob=model(image.unsqueeze(0).to(device))[0,0].float().cpu().numpy()
        gts=[gt_label_map(coco,val_data.img_ids[i]) for i in idxs]
        h,w=gts[0][0].shape
        prob=cv2.resize(prob,(w,h),interpolation=cv2.INTER_LINEAR)   #back to full resolution
        for t in thresholds:
            raw,raw_areas=components(prob>t)
            splits=[split_count(gt,n_gt,raw,len(raw_areas)-1) for gt,n_gt in gts]
            for c in closes:
                labels,areas=(raw,raw_areas) if c==0 else components(prob>t,c)
                for a in min_areas:
                    pred,n_pred=drop_small(labels,areas,a)
                    for (gt,n_gt),split in zip(gts,splits):
                        r=match(gt,n_gt,pred,n_pred)
                        r["split_raw"]=split
                        results[(t,c,a)].append(r)
        if n%20==0 or n==len(files):
            print(f"  {n}/{len(files)} images, {time.perf_counter()-start:.0f}s")

    rows=sorted(((t,c,a,summarise(r)) for (t,c,a),r in results.items()),key=lambda x:-x[3]["PQ"])
    print(f"\n{'thresh':>6} {'close':>5} {'min_area':>8} {'PQ':>6} {'SQ':>6} {'RQ':>6} {'PQ/img':>6} {'Dice':>6} "
          f"{'TP':>5} {'FP':>5} {'FN':>5} {'GT/img':>6} {'pred/img':>8} {'1-to-many':>9} {'many-to-1':>9} {'split(raw)':>10}")
    for t,c,a,s in rows:
        print(f"{t:>6.2f} {c:>5d} {a:>8d} {s['PQ']:>6.3f} {s['SQ']:>6.3f} {s['RQ']:>6.3f} {s['PQ_mean_per_image']:>6.3f} "
              f"{s['mean_dice']:>6.3f} {s['TP']:>5d} {s['FP']:>5d} {s['FN']:>5d} {s['gt_per_image']:>6.1f} "
              f"{s['pred_per_image']:>8.1f} {s['one_to_many']:>9d} {s['many_to_one']:>9d} {s['split_raw']:>10d}")
    t,c,a,s=rows[0]
    print(f"\nBest: threshold {t}, grouping {c} px, min area {a} -> PQ {s['PQ']:.3f} (SQ {s['SQ']:.3f} x RQ {s['RQ']:.3f}), "
          f"mean Dice {s['mean_dice']:.3f}, over {s['entries']} annotator copies")


if __name__=="__main__":
    main()
