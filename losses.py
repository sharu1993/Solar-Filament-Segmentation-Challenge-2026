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


def cldice_loss(probs, targets, iters, smooth=1.0):
    """1 - clDice over the whole batch. Topology precision: share of the predicted centre line inside the true
    filaments; topology sensitivity: share of the true centre line covered by the prediction. Long thin parts
    count by length, not by area, and the score barely depends on how wide each annotator drew the outline."""
    skel_p=soft_skeleton(probs,iters)
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
    much it lowers the IoU."""
    logits=logits.reshape(-1)
    labels=targets.reshape(-1)
    errors=1-logits*(2*labels-1)               #hinge margin: > 0 when the pixel is wrong or not confident enough
    errors_sorted,order=torch.sort(errors,descending=True)
    return torch.dot(F.relu(errors_sorted),_lovasz_grad(labels[order]))


class SegmentationLoss(BCEDiceLoss):
    """bce_weight*BCE + dice_weight*Dice + cldice_weight*(1 - clDice) + lovasz_weight*Lovasz hinge.

    Takes probabilities like BCEDiceLoss, because validation scores the blended tile map, which has no logits.
    The Lovasz term turns them back into logits (clamped at 1e-6 from 0 and 1); pixels saturated beyond that
    get no gradient from it, the same limit BCE already has on probabilities.
    With cldice_weight=lovasz_weight=0 this is exactly BCEDiceLoss."""
    def __init__(self, bce_weight=1.0, dice_weight=1.0, cldice_weight=0.5, lovasz_weight=0.5, cldice_iters=15, smooth=1.0):
        super().__init__(bce_weight,dice_weight,smooth)
        self.cldice_weight=cldice_weight
        self.lovasz_weight=lovasz_weight
        self.cldice_iters=cldice_iters

    def forward(self, probs, targets):
        loss=super().forward(probs,targets)
        if self.cldice_weight:
            loss=loss+self.cldice_weight*cldice_loss(probs,targets,self.cldice_iters,self.smooth)
        if self.lovasz_weight:
            p=probs.clamp(1e-6,1-1e-6)
            loss=loss+self.lovasz_weight*lovasz_hinge(torch.log(p)-torch.log1p(-p),targets)
        return loss
