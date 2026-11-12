"""
Export a training checkpoint as a small weights file for sharing (e.g. in the GitHub repo).

A training checkpoint also holds the optimizer and scheduler state (~295 MB for the ResNet-34 U-Net).
The export keeps only the model weights, stored in half precision (float16), plus the run config
(architecture, resolution, tiling) so evaluate.py and predict.py rebuild the same model: ~50 MB,
under GitHub's 100 MB file limit. Weights are converted back to float32 when loaded.

Usage (from the project folder):
    python deliverables/export_weights.py checkpoints/2026-11-11_22-03_f25df8d.pt deliverables/weights/resnet34_unet_2048.pt
"""
import os
import sys

import torch


def main():
    if len(sys.argv)!=3:
        sys.exit(__doc__)
    src,dst=sys.argv[1],sys.argv[2]
    state=torch.load(src,map_location="cpu",weights_only=False)
    weights={k:(v.half() if v.is_floating_point() else v) for k,v in state["model_state_dict"].items()}
    export={"model_state_dict":weights,"config":state.get("config"),
            "source_checkpoint":os.path.basename(src),"epoch":state.get("epoch"),"val_iou":state.get("val_iou")}
    os.makedirs(os.path.dirname(dst) or ".",exist_ok=True)
    torch.save(export,dst)
    print(f"{src} ({os.path.getsize(src)/1e6:.0f} MB) -> {dst} ({os.path.getsize(dst)/1e6:.0f} MB), "
          f"model {(state.get('config') or {}).get('model','unet')}, epoch {state.get('epoch',-1)+1}")


if __name__=="__main__":
    main()
