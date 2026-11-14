"""
Predict solar filaments on full-disk H-alpha images and write a competition submission CSV.

Pipeline per image (same as evaluate.py, so validation scores carry over):
  grayscale image -> preprocessing as in training (none, or limb darkening + local background divided out,
  see preprocess.py; read from the checkpoint config) -> resize to the training resolution (2048 = none)
  -> model in overlapping, blended tiles (tiling.py) -> probability map at 2048x2048
  -> threshold -> connected components -> pieces grouped into one filament (--close link: only pieces whose
     ends face each other across a gap the model partly saw, see grouping.py; or a number: pieces within ~N px)
  -> drop filaments smaller than --min-area px or with mean probability below --min-conf
  -> one row per filament: filament_id = <image name>_<n>, segmentation_rle = pycocotools RLE counts

Defaults are the best validation setting of the current best model, 4c8aa13 (ResNet-34 U-Net with
full-resolution skip, limb normalisation, BCE + Dice + clDice + Lovasz loss; see README.md):
threshold 0.5, direction-aware grouping (link), min area 250 px, min confidence 0.72.
Older models were tuned with other settings (2026-11-11_22-03_f25df8d: --threshold 0.4 --close 61 --min-conf 0.92).

Usage (from the project folder):
    python deliverables/predict.py --checkpoint checkpoints/2026-11-13_06-47_4c8aa13.pt
    python deliverables/predict.py --checkpoint deliverables/weights/resnet34_unet_2048.pt   # exported weights
    python deliverables/predict.py --images path/to/jpegs --out my_submission.csv --limit 5
"""
import os
import sys

#the model code lives in the project folder, one level up
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
#same workaround as main.py: torch and cv2 each load their own OpenMP runtime on macOS
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["OMP_NUM_THREADS"]="1"

import argparse
import csv
import glob
import time

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools import mask as mask_utils

import preprocess
import tiling
from evaluate import group, close_setting, drop_small, load_model, default_checkpoint

TEST_IMAGES=os.path.join(ROOT,"MAGFiLO_1.0_Kaggle_2026","test","test_images")
#best validation setting of the current best model (also used by make_figures.py)
THRESHOLD,CLOSE,MIN_AREA,MIN_CONF=0.5,"link",250,0.72


def load_image(path, resize_to, preprocessing=None):
    """Image as the model saw it in training (same steps as dataset.Dataset), plus its full size.
    preprocessing=None: grayscale, resized (BOX) if needed, ImageNet-normalised.
    preprocessing="local": limb darkening and local background divided out at full size (preprocess.py),
    resized (area) if needed; already standardised, so no further normalisation."""
    image=Image.open(path).convert("L")
    full_size=image.size
    resize=resize_to and image.size!=(resize_to,resize_to)
    if preprocessing=="local":
        img=preprocess.local_normalise(np.array(image))
        if resize:
            img=cv2.resize(img,(resize_to,resize_to),interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img).unsqueeze(0), full_size
    if preprocessing:
        raise ValueError(f"Unknown preprocessing '{preprocessing}' in the checkpoint config")
    if resize:
        image=image.resize((resize_to,resize_to),resample=Image.BOX)
    x=TF.normalize(TF.to_tensor(image),mean=[0.485],std=[0.229])
    return x, full_size


def filaments(prob, threshold, close, min_area, min_conf):
    """Probability map -> list of boolean masks, one per predicted filament."""
    labels,areas=group(prob,threshold,close)
    pred,n,_=drop_small(labels,areas,min_area)
    if n==0:
        return []
    conf=np.bincount(pred.ravel(),weights=prob.ravel(),minlength=n+1)/np.maximum(np.bincount(pred.ravel(),minlength=n+1),1)
    return [pred==k for k in range(1,n+1) if conf[k]>=min_conf]


def rle_counts(mask):
    """pycocotools RLE 'counts' string of a 2-D boolean mask."""
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint",default=None,help="training checkpoint or exported weights; default: newest in checkpoints/")
    ap.add_argument("--images",default=TEST_IMAGES,help="folder of .jpeg images (default: the competition test set)")
    ap.add_argument("--out",default="submission.csv")
    ap.add_argument("--threshold",type=float,default=THRESHOLD)
    ap.add_argument("--close",type=close_setting,default=CLOSE,
                    help="'link' = join pieces whose ends face each other (grouping.py); a number = group pieces within ~this many pixels (0 = off)")
    ap.add_argument("--min-area",type=int,default=MIN_AREA,help="minimum filament size in pixels at 2048x2048")
    ap.add_argument("--min-conf",type=float,default=MIN_CONF,help="minimum mean probability inside a filament (0 = off)")
    ap.add_argument("--limit",type=int,default=None,help="only the first N images (quick check)")
    args=ap.parse_args()

    device=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    checkpoint=args.checkpoint or os.path.join(ROOT,default_checkpoint())
    model,config=load_model(checkpoint,device)
    resize_to=config.get("resize_to",1024)
    tile,overlap=config.get("tile_size"),config.get("tile_overlap",128)
    preprocessing=config.get("preprocess")
    paths=sorted(glob.glob(os.path.join(args.images,"*.jpeg")))[:args.limit]
    if not paths:
        sys.exit(f"No .jpeg images found in {args.images}")
    print(f"Predicting {len(paths)} images with {checkpoint} on {device} "
          f"(resolution {resize_to}, preprocessing {preprocessing or 'none'}, tiles {tile or 'off'}, overlap {overlap}; threshold {args.threshold}, "
          f"grouping {args.close}{'' if args.close=='link' else ' px'}, min area {args.min_area} px, min confidence {args.min_conf})")

    rows=0
    start=time.perf_counter()
    with open(args.out,"w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["filament_id","segmentation_rle"])
        for i,path in enumerate(paths,start=1):
            x,(w,h)=load_image(path,resize_to,preprocessing)
            with torch.no_grad():
                prob=tiling.predict_tiled(model,x.unsqueeze(0).to(device),tile,overlap)[0,0].float().cpu().numpy()
            if prob.shape!=(h,w):
                prob=cv2.resize(prob,(w,h),interpolation=cv2.INTER_LINEAR)   #back to full resolution
            name=os.path.splitext(os.path.basename(path))[0]
            for n,mask in enumerate(filaments(prob,args.threshold,args.close,args.min_area,args.min_conf),start=1):
                writer.writerow([f"{name}_{n}",rle_counts(mask)])
                rows+=1
            if i%20==0 or i==len(paths):
                print(f"  {i}/{len(paths)} images, {rows} filaments, {time.perf_counter()-start:.0f}s")
    print(f"Wrote {rows} filaments ({rows/len(paths):.1f} per image) to {args.out}")


if __name__=="__main__":
    main()
