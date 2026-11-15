import cv2
import numpy as np
import torch
import torch.nn.functional as F

class BCEDiceLoss(torch.nn.Module):
    """
    Binary cross-entropy + soft Dice loss for thin, rare structures (filaments are ~0.3% of pixels).

    BCE scores every pixel on its own (stable gradients, but swamped by easy background pixels).
    Dice scores the overlap between predicted and true filament area (ignores the background count,
    closely tied to IoU). Expects probabilities (the UNet ends in a sigmoid) and 0/1 targets.
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super().__init__()
        self.bce=torch.nn.BCELoss()
        self.bce_weight=bce_weight
        self.dice_weight=dice_weight
        self.smooth=smooth  #keeps Dice defined when a batch has no filament pixels at all

    def forward(self, probs, targets):
        bce=self.bce(probs, targets)

        #soft Dice over the whole batch (not per image): crops with no filament then only add their
        #false-positive area to the total, instead of each counting as a near-total miss
        intersection=(probs*targets).sum()
        dice=(2*intersection+self.smooth)/(probs.sum()+targets.sum()+self.smooth)
        dice_loss=1-dice

        return self.bce_weight*bce+self.dice_weight*dice_loss


# ---------------------------------------------------------------------------------------------------------
# clDice and Lovasz hinge, added on top of BCE + Dice by SegmentationLoss below

def _soft_erode(x):
    """Grey-level erosion with a 3x3 cross: each pixel becomes the minimum of itself and its 4 neighbours."""
    p1=-F.max_pool2d(-x,(3,1),1,(1,0))
    p2=-F.max_pool2d(-x,(1,3),1,(0,1))
    return torch.min(p1,p2)


def _soft_dilate(x):
    return F.max_pool2d(x,3,1,1)


def soft_skeleton(x, iters):
    """Differentiable centre line of a (N,1,H,W) map in [0,1] (Shit et al. 2021, clDice). Each iteration peels
    one pixel off every side, so iters must be about the half-width of the widest structure to reach its centre."""
    skel=F.relu(x-_soft_dilate(_soft_erode(x)))
    for _ in range(iters):
        x=_soft_erode(x)
        delta=F.relu(x-_soft_dilate(_soft_erode(x)))
        skel=skel+F.relu(delta-skel*delta)
    return skel


def cldice_loss(probs, targets, iters, smooth=1.0, skel_t=None):
    """1 - clDice over the whole batch. Topology precision: share of the predicted centre line inside the true
    filaments; topology sensitivity: share of the true centre line covered by the prediction. Long thin parts
    count by length, not by area, and the score barely depends on how wide each annotator drew the outline.
    skel_t: the target's soft skeleton, if already computed."""
    skel_p=soft_skeleton(probs,iters)
    if skel_t is None:
        with torch.no_grad():
            skel_t=soft_skeleton(targets,iters)
    tprec=((skel_p*targets).sum()+smooth)/(skel_p.sum()+smooth)
    tsens=((skel_t*probs).sum()+smooth)/(skel_t.sum()+smooth)
    return 1-2*tprec*tsens/(tprec+tsens)


def _lovasz_grad(gt_sorted):
    """How much 1 - IoU grows with each pixel, in the order the errors were sorted (Berman et al. 2018)."""
    gts=gt_sorted.sum()
    intersection=gts-gt_sorted.cumsum(0)
    union=gts+(1-gt_sorted).cumsum(0)          #always >= 1, also for crops without any filament
    jaccard=1-intersection/union
    jaccard[1:]=jaccard[1:]-jaccard[:-1]
    return jaccard


def lovasz_hinge(logits, targets):
    """Lovasz hinge over the whole batch (all pixels together, like the Dice term): a smooth stand-in for
    1 - IoU of the filament mask. Pixels are sorted by how wrong they are, and each error is weighted by how
    much it lowers the IoU. Needs 0/1 labels: soft targets (annotator fractions) are turned into a majority
    vote, filament where at least half of the annotators marked it."""
    logits=logits.reshape(-1)
    labels=(targets.reshape(-1)>=0.5).float()
    errors=1-logits*(2*labels-1)               #hinge margin: > 0 when the pixel is wrong or not confident enough
    errors_sorted,order=torch.sort(errors,descending=True)
    return torch.dot(F.relu(errors_sorted),_lovasz_grad(labels[order]))


def skeleton_weighted_bce(probs, targets, skel_weight, iters, skel_t=None):
    """BCE with each pixel near a filament's centre line counted 1 + skel_weight times. A gap across a filament
    then costs as much as losing skel_weight times more area, so pieces of one filament are pushed to connect.
    The centre line is the soft skeleton of the target (centre lines computed from the masks), widened by one
    pixel on each side so it is not missed by a one-pixel offset. skel_t: the target's soft skeleton, if already computed."""
    with torch.no_grad():
        if skel_t is None:
            skel_t=soft_skeleton(targets,iters)
        weight=1+skel_weight*_soft_dilate(skel_t)
    return F.binary_cross_entropy(probs,targets,weight=weight)


def instance_dice_loss(probs, targets, margin=32, smooth=1.0):
    """Mean of 1 - Dice over the annotated filaments, so each filament counts equally, as in PQ (blob loss,
    Kofler et al. 2023). A filament is a connected piece of the majority-vote target. Each one is scored in its
    own neighbourhood (the piece widened by `margin` px) with the other filaments cut out, so a prediction that
    spills into the background next to it or joins it to a neighbour lowers its score, while errors far away
    are left to the whole-image terms. Returns 0 for a batch without filaments."""
    with torch.no_grad():
        majority=(targets>=0.5).squeeze(1).to("cpu",torch.uint8).numpy()
    total,count=probs.new_zeros(()),0
    for b in range(probs.shape[0]):
        n,labels,stats,_=cv2.connectedComponentsWithStats(majority[b],connectivity=8)
        if n<=1:
            continue
        labels=torch.from_numpy(labels.astype(np.int64)).to(probs.device)
        H,W=labels.shape
        for k in range(1,n):
            x,y,w,h=stats[k,:4]
            y0,y1,x0,x1=max(y-margin,0),min(y+h+margin,H),max(x-margin,0),min(x+w+margin,W)   #box + margin
            lab=labels[y0:y1,x0:x1]
            piece=(lab==k)
            with torch.no_grad():
                #square dilation by margin, as two 1-D max-pools (same result as one square pool, much cheaper)
                near=piece[None,None].float()
                near=F.max_pool2d(near,(2*margin+1,1),1,(margin,0))
                near=F.max_pool2d(near,(1,2*margin+1),1,(0,margin))[0,0]>0
                region=(near&((lab==0)|piece)).float()       #neighbourhood minus the other filaments
            p,t=probs[b,0,y0:y1,x0:x1]*region,targets[b,0,y0:y1,x0:x1]*region
            dice=(2*(p*t).sum()+smooth)/(p.sum()+t.sum()+smooth)
            total=total+(1-dice)
            count+=1
    return total/count if count else total


class SegmentationLoss(BCEDiceLoss):
    """bce_weight*BCE + dice_weight*Dice + cldice_weight*(1 - clDice) + lovasz_weight*Lovasz hinge
    + instance_dice_weight*(1 - mean Dice per filament).
    With skel_weight > 0 the BCE term is skeleton-weighted (skeleton_weighted_bce).

    Takes probabilities like BCEDiceLoss, because validation scores the blended tile map, which has no logits.
    The Lovasz term turns them back into logits (clamped at 1e-6 from 0 and 1); pixels saturated beyond that
    get no gradient from it, the same limit BCE already has on probabilities.
    With cldice_weight=lovasz_weight=skel_weight=instance_dice_weight=0 this is exactly BCEDiceLoss."""
    def __init__(self, bce_weight=1.0, dice_weight=1.0, cldice_weight=0.5, lovasz_weight=0.5, cldice_iters=15, smooth=1.0,
                 skel_weight=0.0, instance_dice_weight=0.0, instance_margin=32):
        super().__init__(bce_weight,dice_weight,smooth)
        self.cldice_weight=cldice_weight
        self.lovasz_weight=lovasz_weight
        self.cldice_iters=cldice_iters
        self.skel_weight=skel_weight
        self.instance_dice_weight=instance_dice_weight
        self.instance_margin=instance_margin

    def forward(self, probs, targets):
        skel_t=None
        if self.skel_weight and self.cldice_weight:
            with torch.no_grad():
                skel_t=soft_skeleton(targets,self.cldice_iters)    #target centre line, shared by both terms
        if self.skel_weight:
            #same as BCEDiceLoss.forward, with the skeleton-weighted BCE in place of the plain one
            intersection=(probs*targets).sum()
            dice=(2*intersection+self.smooth)/(probs.sum()+targets.sum()+self.smooth)
            loss=(self.bce_weight*skeleton_weighted_bce(probs,targets,self.skel_weight,self.cldice_iters,skel_t)
                  +self.dice_weight*(1-dice))
        else:
            loss=super().forward(probs,targets)
        if self.cldice_weight:
            loss=loss+self.cldice_weight*cldice_loss(probs,targets,self.cldice_iters,self.smooth,skel_t)
        if self.lovasz_weight:
            p=probs.clamp(1e-6,1-1e-6)
            loss=loss+self.lovasz_weight*lovasz_hinge(torch.log(p)-torch.log1p(-p),targets)
        if self.instance_dice_weight:
            loss=loss+self.instance_dice_weight*instance_dice_loss(probs,targets,self.instance_margin,self.smooth)
        return loss
