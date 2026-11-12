"""Tiled prediction: run the model on overlapping tiles and blend them into one probability map.

Used for validation during training and in evaluate.py, so large images (e.g. the full 2048x2048)
never go through the network in one piece. Tiles overlap by `overlap` pixels; inside each tile the
weight ramps up linearly from the edges over `overlap` pixels, so the less reliable predictions near
a tile's edge (where the network cannot see beyond the border) count less than those from the
neighbouring tile, and seams fade out smoothly. Filaments crossing a tile border are therefore
joined in the stitched map and become one connected component afterwards.
"""
import torch


def tile_starts(length, tile, overlap):
    """Start positions along one axis: every (tile - overlap) pixels, last tile flush with the end."""
    if length<=tile:
        return [0]
    starts=list(range(0,length-tile+1,tile-overlap))
    if starts[-1]!=length-tile:
        starts.append(length-tile)   #last tile shifted back to end at the border (overlaps a bit more)
    return starts


def blend_weights(tile, overlap, device=None):
    """(tile, tile) weights: 1 in the middle, falling linearly to 1/(overlap+1) at the edges (never 0)."""
    w=torch.ones(tile,device=device)
    if overlap>0:
        ramp=torch.arange(1,overlap+1,device=device,dtype=torch.float32)/(overlap+1)
        w[:overlap]=ramp
        w[-overlap:]=ramp.flip(0)
    return w[:,None]*w[None,:]


def predict_tiled(model, x, tile=512, overlap=128, batch_size=4):
    """Probability map (N=1, 1, H, W) for x (1, C, H, W). tile=None, or an image no bigger than one
    tile, means a single pass of the whole image."""
    _,_,H,W=x.shape
    if tile is None or (H<=tile and W<=tile):
        return model(x)
    if tile%32:
        raise ValueError(f"tile must be divisible by 32, got {tile}")
    if not 0<=overlap<tile:
        raise ValueError(f"overlap must be between 0 and tile-1, got {overlap}")
    th,tw=min(tile,H),min(tile,W)
    weight=blend_weights(tile,overlap,x.device)[:th,:tw]
    total=torch.zeros((1,1,H,W),device=x.device)
    norm=torch.zeros((1,1,H,W),device=x.device)
    corners=[(t,l) for t in tile_starts(H,tile,overlap) for l in tile_starts(W,tile,overlap)]
    for i in range(0,len(corners),batch_size):
        chunk=corners[i:i+batch_size]
        tiles=torch.cat([x[:,:,t:t+th,l:l+tw] for t,l in chunk])
        probs=model(tiles)
        for (t,l),p in zip(chunk,probs):
            total[0,:,t:t+th,l:l+tw]+=p*weight
            norm[0,:,t:t+th,l:l+tw]+=weight
    return total/norm
