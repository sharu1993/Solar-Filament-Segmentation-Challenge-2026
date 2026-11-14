"""
Direction-aware grouping of predicted filament pieces (an alternative to grouping by morphological closing).

The closing in evaluate.components joins any pieces within ~close_size pixels, in any direction: long
filaments with larger gaps stay split, while unrelated blobs next to a filament get attached. Here two
pieces are joined only if they look like parts of the same filament:
  1. an end of one is within max_gap pixels of an end of the other (ends = the two geodesic extremes of
     each piece, i.e. the far ends of its length, found by two breadth-first searches inside the piece);
  2. the pieces point at each other: at each of the two ends, the outward direction (from the centre of
     the last `near` pixels of the piece to the end) is within max_angle degrees of the line to the other end;
  3. the model saw something in the gap: the mean probability along the straight line between the two
     ends is at least gap_prob.
Ends closer than blob_gap pixels are joined without the direction and gap checks, and short or round
pieces (no clear direction) are only joined that way. Only the predicted pixels are kept (gaps are not
filled), as in evaluate.components. Uses numpy and OpenCV only.

Settings were chosen on the validation maps of 4c8aa13 (README.md): max_gap 100, max_angle 45,
gap_prob 0.3, blob_gap 15.
"""
from collections import deque
import numpy as np
import cv2

LINK="link"   #value of the grouping setting (--close) that selects this method


def _bfs(mask, start):
    """Geodesic distances (8-connected steps) inside mask from start; -1 outside the mask."""
    h,w=mask.shape
    dist=np.full((h,w),-1,np.int32)
    dist[start]=0
    queue=deque([start])
    while queue:
        y,x=queue.popleft()
        nd=dist[y,x]+1
        for yy in (y-1,y,y+1):
            if 0<=yy<h:
                for xx in (x-1,x,x+1):
                    if 0<=xx<w and mask[yy,xx] and dist[yy,xx]<0:
                        dist[yy,xx]=nd
                        queue.append((yy,xx))
    return dist


def piece_ends(mask, near=30, step=2):
    """The two ends of a piece (x, y) with the outward unit direction at each (None if undefined), and the
    piece's geodesic length in pixels. Works on a copy subsampled every `step` pixels, for speed."""
    ys,xs=np.nonzero(mask)
    y0,x0=ys.min(),xs.min()
    m=mask[y0:ys.max()+1,x0:xs.max()+1].astype(np.uint8)
    if step>1:
        m=cv2.resize(m,((m.shape[1]+step-1)//step,(m.shape[0]+step-1)//step),interpolation=cv2.INTER_NEAREST)
        if not m.any():
            m=np.ones((1,1),np.uint8)
    d0=_bfs(m,tuple(np.argwhere(m)[0]))
    e1=np.unravel_index(np.argmax(d0),d0.shape)    #farthest point from anywhere = one end
    d1=_bfs(m,e1)
    e2=np.unravel_index(np.argmax(d1),d1.shape)    #farthest from that end = the other end
    d2=_bfs(m,e2)
    ends=[]
    for e,dist in ((e1,d1),(e2,d2)):
        cy,cx=np.argwhere((dist>=0)&(dist<=near//step)).mean(axis=0)
        v=np.array([e[1]-cx,e[0]-cy],dtype=np.float64)
        n=np.hypot(*v)
        point=np.array([x0+e[1]*step+step/2,y0+e[0]*step+step/2])
        ends.append((point,v/n if n>0 else None))
    return ends,int(d1[e2])*step


def _joined(a, b, prob, max_gap, cos_max, gap_prob, blob_gap, min_len):
    """Should pieces a and b (each (ends, length)) be one filament? Compares their closest pair of ends."""
    dist,pa,da,pb,db,v=min(((np.hypot(*(pb-pa)),pa,da,pb,db,pb-pa) for pa,da in a[0] for pb,db in b[0]),key=lambda t:t[0])
    if dist<=blob_gap:
        return True
    if dist>max_gap or a[1]<min_len or b[1]<min_len or da is None or db is None:
        return False
    u=v/dist
    if np.dot(da,u)<cos_max or np.dot(db,-u)<cos_max:
        return False
    k=max(int(dist),2)
    xs=np.linspace(pa[0],pb[0],k).round().astype(int).clip(0,prob.shape[1]-1)
    ys=np.linspace(pa[1],pb[1],k).round().astype(int).clip(0,prob.shape[0]-1)
    return prob[ys,xs].mean()>=gap_prob


def link_components(prob, threshold, max_gap=100, max_angle=45, gap_prob=0.3, blob_gap=15, min_len=25, min_piece=30):
    """Probability map -> (label map, area of each label), like evaluate.components: connected pieces of
    prob > threshold, joined when they look like parts of the same filament (see module docstring).
    Pieces smaller than min_piece pixels are not linked (they stay on their own)."""
    binary=(prob>threshold).astype(np.uint8)
    n,labels=cv2.connectedComponents(binary,connectivity=8)
    areas=np.bincount(labels.ravel(),minlength=n)
    ids=[k for k in range(1,n) if areas[k]>=min_piece]
    pieces=[piece_ends(labels==k) for k in ids]
    parent=list(range(len(ids)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    cos_max=np.cos(np.radians(max_angle))
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            if find(i)!=find(j) and _joined(pieces[i],pieces[j],prob,max_gap,cos_max,gap_prob,blob_gap,min_len):
                parent[find(j)]=find(i)
    root=np.arange(n)
    for i,k in enumerate(ids):
        root[k]=ids[find(i)]
    merged=root[labels]
    uniq,new=np.unique(merged,return_inverse=True)   #uniq[0]==0 (background)
    new=new.reshape(labels.shape).astype(np.int32)
    return new, np.bincount(new.ravel(),minlength=len(uniq))
