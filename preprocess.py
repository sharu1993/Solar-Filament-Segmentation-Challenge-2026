"""
Per-image preprocessing for GONG H-alpha full-disk images: find the solar limb, then divide each disk
pixel by the median brightness of its ring (limb darkening) and then of its neighbourhood
(local background normalisation).

This removes limb darkening (the disk gets darker towards its edge) and other large-scale brightness
changes (haze, station/exposure differences), so a filament looks the same anywhere on the disk.
Measured on 71 training images: the background stays within +/-0.07 of the disk median from the centre
to 0.95 of the radius (-0.44 in the last 5%), and filaments are ~1.4-1.8 robust-spread units darker
than the disk in every zone. In the raw images the background is ~25% darker at 0.85-0.95 of the radius
than in the centre.
The result is standardised per image (median 0, robust spread 1) and clipped; off-disk pixels get the
lowest value. No position information is added: only brightness is evened out.
"""
import numpy as np
import cv2

CLIP=5.0            #standardised values are clipped to [-CLIP, CLIP]; off-disk = -CLIP
WINDOW=125          #neighbourhood for the local background, in pixels at 2048x2048
SMALL=512           #the background is estimated on a 512x512 copy (fast), then scaled back up
RINGS=200           #rings for the radial (limb darkening) profile; the last rings follow the steep edge


def find_limb(img):
    """Centre (x, y) and radius of the solar disk, fitted to the visible limb.
    Threshold at half the brightness of the central region, so the grey instrument ring outside
    the limb is excluded; least-squares circle through the edge pixels (typical error ~1-2 px)."""
    h,w=img.shape
    centre=img[h//2-256:h//2+256,w//2-256:w//2+256]
    b=(img>0.5*np.median(centre)).astype(np.uint8)
    b=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    p=max(contours,key=cv2.contourArea)[:,0,:].astype(np.float64)
    p=p[(p[:,0]>2)&(p[:,0]<w-3)&(p[:,1]>2)&(p[:,1]<h-3)]   #skip points on the image border
    x,y=p[:,0],p[:,1]
    cx,cy,k=np.linalg.lstsq(np.c_[2*x,2*y,np.ones_like(x)],x*x+y*y,rcond=None)[0]
    return cx,cy,np.sqrt(k+cx*cx+cy*cy)


def radial_profile(img, cx, cy, r):
    """Median brightness of the disk in thin rings around the centre, as a full-size image (sky = 1)."""
    h,w=img.shape
    yy,xx=np.mgrid[0:h,0:w]
    rr=np.hypot(xx-cx,yy-cy)/r
    disk=rr<1.0
    ring=np.minimum((rr[disk]*RINGS).astype(np.int32),RINGS-1)
    vals=img[disk].astype(np.float32)
    order=np.argsort(ring,kind="stable")
    ring,vals=ring[order],vals[order]
    edges=np.searchsorted(ring,np.arange(RINGS+1))
    prof=np.array([np.median(vals[edges[k]:edges[k+1]]) if edges[k+1]>edges[k] else np.nan for k in range(RINGS)])
    prof=np.where(np.isnan(prof),np.nanmedian(prof),prof)
    P=np.interp(np.clip(rr,0,1),(np.arange(RINGS)+0.5)/RINGS,prof).astype(np.float32)
    P[~disk]=1.0
    return P, disk


def local_normalise(img):
    """uint8 image -> float32 image of the same size: limb darkening divided out (thin rings), then each
    disk pixel divided by its local background, standardised per image and clipped to [-CLIP, CLIP];
    off-disk = -CLIP."""
    h,w=img.shape
    cx,cy,r=find_limb(img)
    P,disk=radial_profile(img,cx,cy,r)
    flat=img.astype(np.float32)/np.maximum(P,1.0)*128     #evenly lit disk, rescaled to ~128 for uint8 median
    s=SMALL/w
    yy,xx=np.mgrid[0:SMALL,0:SMALL]
    disk_small=np.hypot(xx-cx*s,yy-cy*s)<r*s
    small=cv2.resize(flat,(SMALL,SMALL),interpolation=cv2.INTER_AREA)
    #fill the sky with a smooth extension of the nearby disk, so the background near the limb is not pulled down
    sigma=WINDOW*s/2
    m=disk_small.astype(np.float32)
    smooth=cv2.GaussianBlur(small*m,(0,0),sigma)/np.maximum(cv2.GaussianBlur(m,(0,0),sigma),1e-3)
    filled=np.where(disk_small,small,smooth)
    k=int(WINDOW*s)|1
    bg=cv2.medianBlur(np.clip(filled,0,255).astype(np.uint8),k).astype(np.float32)
    bg=cv2.resize(bg,(w,h),interpolation=cv2.INTER_LINEAR)
    ratio=flat/np.maximum(bg,1.0)
    vals=ratio[disk]
    med=np.median(vals)
    spread=1.4826*np.median(np.abs(vals-med))   #robust standard deviation
    out=np.clip((ratio-med)/max(spread,1e-3),-CLIP,CLIP).astype(np.float32)
    out[~disk]=-CLIP
    return out
