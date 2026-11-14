"""
Regenerate the README figures in deliverables/figures/ from saved validation probability maps.

Needs the maps of the best model (4c8aa13), saved once with:
    python evaluate.py --checkpoint checkpoints/2026-11-13_06-47_4c8aa13.pt --save-probs probs/4c8aa13
Then (from the project folder):
    python deliverables/make_figures.py
Predictions use the post-processing defaults of predict.py. The example images are picked automatically by
per-annotator-copy PQ (25th / 50th / 85th percentile), so they are representative, not best cases.
"""
import os, sys, numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
sys.path.insert(0,ROOT); sys.path.insert(0,HERE)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import evaluate as ev, dataset, predict
from pycocotools.coco import COCO
from collections import defaultdict
OUT=os.path.join(HERE,"figures")
os.makedirs(OUT,exist_ok=True)
PROBS=os.path.join(ROOT,"probs","4c8aa13")
GT, PRED = "#2a78d6", "#eb6834"                 # reference palette, categorical slots 1 and 2
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size":10,"text.color":INK,"axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,
                     "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF})
coco=COCO(os.path.join(ROOT,ev.ANNOTATION_FILE)); _,va=dataset.split_by_file(coco)
by_file=defaultdict(list)
for i in va: by_file[coco.imgs[i]['file_name']].append(i)

def pred_masks(f):
    prob=cv2.imread(os.path.join(PROBS,os.path.splitext(f)[0]+".png"),0).astype(np.float32)/255
    return predict.filaments(prob,predict.THRESHOLD,predict.CLOSE,predict.MIN_AREA,predict.MIN_CONF)
def gt_masks(img_id):
    return [coco.annToMask(a).astype(bool) for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))]
def draw(ax, masks, color, scale=1.0, off=(0,0), lw=1.0):
    for m in masks:
        cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in cs:
            c=c[:,0,:].astype(float); c=np.vstack([c,c[:1]])
            ax.plot((c[:,0]-off[0])*scale,(c[:,1]-off[1])*scale,color=color,lw=lw)
def best_window(masks, size):
    acc=np.zeros((2048,2048),np.float32)
    for m in masks: acc+=m
    ii=cv2.integral(acc)
    best=(0,0,0)
    for y in range(0,2048-size+1,64):
        for x in range(0,2048-size+1,64):
            s=ii[y+size,x+size]-ii[y,x+size]-ii[y+size,x]+ii[y,x]
            if s>best[0]: best=(s,x,y)
    x,y=best[1],best[2]
    # re-centre the window on the outlines inside it, so the zoom is not mostly empty
    yy,xx=np.nonzero(acc[y:y+size,x:x+size])
    if len(xx):
        x=int(np.clip(x+xx.mean()-size/2,0,2048-size)); y=int(np.clip(y+yy.mean()-size/2,0,2048-size))
    return x,y
def image(f): return cv2.imread(os.path.join(ROOT,ev.IMAGE_DIR,f),cv2.IMREAD_GRAYSCALE)

def label_map(masks):
    lab=np.zeros((2048,2048),np.int32)
    for k,m in enumerate(masks,start=1): lab[m&(lab==0)]=k
    return lab

# ---- figure 1: three validation images (25th / 50th / 85th percentile of per-annotator-copy PQ)
scores=[]   #(PQ, file, copy index) for every annotator copy with at least one filament or prediction
for f in sorted(by_file):
    p=pred_masks(f); pl=label_map(p)
    for c,img_id in enumerate(by_file[f]):
        g,ng=ev.gt_label_map(coco,img_id)
        r=ev.match(g,ng,pl,len(p))
        d=r["tp"]+0.5*r["fp"]+0.5*r["fn"]
        if d>0: scores.append((r["sum_iou"]/d,f,c))
scores.sort()
picks=[]
for q,name in ((25,"25th percentile"),(50,"Median"),(85,"85th percentile")):
    v,f,c=scores[min(int(len(scores)*q/100),len(scores)-1)]
    n=len(by_file[f])
    picks.append((f,c,f"{name} image (image PQ {v:.2f}{f'; annotator {c+1} of {n} shown' if n>1 else ''})"))
Z=640
fig,axes=plt.subplots(3,2,figsize=(10,15.2))
for row,(f,c,title) in enumerate(picks):
    img=image(f); g=gt_masks(by_file[f][c]); p=pred_masks(f)
    small=cv2.resize(img,(1024,1024),interpolation=cv2.INTER_AREA)
    a=axes[row,0]; a.imshow(small,cmap="gray",vmin=0,vmax=255); draw(a,g,GT,0.5,lw=0.9); draw(a,p,PRED,0.5,lw=0.9)
    x,y=best_window(g+p,Z)
    a.add_patch(plt.Rectangle((x*0.5,y*0.5),Z*0.5,Z*0.5,fill=False,ec="white",lw=1,ls="--"))
    a.set_title(f"{title}\n{os.path.splitext(f)[0]}",fontsize=10,loc="left")
    b=axes[row,1]; b.imshow(img[y:y+Z,x:x+Z],cmap="gray",vmin=0,vmax=255); draw(b,g,GT,1,(x,y),lw=1.4); draw(b,p,PRED,1,(x,y),lw=1.4)
    b.set_title(f"Zoom {Z}x{Z} px (dashed box)",fontsize=10,loc="left")
    for ax in (a,b): ax.set_xticks([]); ax.set_yticks([])
    for ax,s in ((a,1024),(b,Z)): ax.set_xlim(0,s); ax.set_ylim(s,0)
fig.legend(handles=[Line2D([],[],color=GT,lw=2,label="Annotated filament (ground truth)"),
                    Line2D([],[],color=PRED,lw=2,label="Model prediction")],loc="lower center",ncol=2,frameon=False)
fig.tight_layout(rect=(0,0.025,1,1)); fig.savefig(f"{OUT}/examples.png",dpi=110); plt.close(fig)

# ---- figure 2: three annotators vs the model on the same region
f="20131016185334Ch.jpeg"; img=image(f); ids=by_file[f]; gts=[gt_masks(i) for i in ids]; p=pred_masks(f)
Z=900; x,y=best_window(sum(gts,[]),Z)
fig,axes=plt.subplots(1,4,figsize=(16,4.6))
for k,(ax,masks,color,name) in enumerate(zip(axes,gts+[p],[GT]*len(gts)+[PRED],[f"Annotator {j+1}" for j in range(len(gts))]+["Model"])):
    ax.imshow(img[y:y+Z,x:x+Z],cmap="gray",vmin=0,vmax=255); draw(ax,masks,color,1,(x,y),lw=1.3)
    ax.set_xlim(0,Z); ax.set_ylim(Z,0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name} ({len(masks)} filaments in full image)",fontsize=10,loc="left")
fig.suptitle(f"Same image ({os.path.splitext(f)[0]}, {Z}x{Z} px region), three annotators and the model",x=0.01,ha="left",fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/annotators_vs_model.png",dpi=100); plt.close(fig)

# ---- figure 3: validation PQ by step, with the annotator-agreement reference
steps=[("U-Net, 5 epochs",0.277),("+ augmentation, longer training",0.346),("+ ImageNet ResNet-34 encoder",0.395),
       ("+ 2048 resolution, cosine LR",0.399),("+ confidence filter (>= 0.92)",0.413),
       ("+ full-res skip, limb normalisation",0.405),("+ clDice + Lovasz loss (conf >= 0.72)",0.426)]
fig,ax=plt.subplots(figsize=(8,4.6))
yy=np.arange(len(steps))[::-1]
ax.barh(yy,[v for _,v in steps],height=0.55,color=GT)
for yv,(_,v) in zip(yy,steps): ax.text(v+0.004,yv,f"{v:.3f}",va="center",color=INK,fontsize=9)
ax.axvline(0.345,color=INK2,lw=1,ls="--")
ax.text(0.341,len(steps)-0.45,"annotator vs annotator: 0.345*",color=INK2,fontsize=9,ha="right",va="bottom")
ax.set_yticks(yy); ax.set_yticklabels([s for s,_ in steps]); ax.set_xlim(0,0.46); ax.set_ylim(-0.6,len(steps)-0.1)
ax.set_xlabel("Panoptic Quality on the validation set (141 images, 217 annotator copies)\n"
              "* annotators scored against each other (48 validation images with 2+ annotators)",fontsize=9)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="x",color="#e6e5e1",lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/pq_progression.png",dpi=130); plt.close(fig)
print(f"Figures written to {OUT}")
