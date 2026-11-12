import torch

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
