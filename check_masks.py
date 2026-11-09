"""Visual check: run dataset.Dataset (the real code) and overlay image + mask for a few samples.

Usage (from the project folder):
    python check_masks.py           # opens a window
    python check_masks.py --save    # writes mask_check_real.png instead (works without a display)
"""
import random, sys
import numpy as np
import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")  # no display needed when saving
import matplotlib.pyplot as plt
import torch
import dataset

random.seed(0); torch.manual_seed(0)

data = dataset.Dataset(
    image_dir="./MAGFiLO_1.0_Kaggle_2026/train/train_images",
    annotation_file="./MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    resize_to=1024,
    crop_size=512,
)

idxs = random.sample(range(len(data)), 6)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, i in zip(axes.flat, idxs):
    image, mask = data[i]                         # real __getitem__: resize + same random crop
    img = image[0].numpy(); m = mask[0].numpy()
    print(f"idx {i:4d} {data.img_ids[i]}: image {tuple(image.shape)}, mask {tuple(mask.shape)}, "
          f"mask values {np.unique(m)}, filament pixels {int(m.sum())}")
    ax.imshow(img, cmap="gray")
    overlay = np.zeros((*m.shape, 4)); overlay[m == 1] = [0, 1, 1, 0.45]
    ax.imshow(overlay)
    ax.set_title(f"{data.img_ids[i]}\n{int(m.sum())} filament px", fontsize=8)
    ax.axis("off")
plt.tight_layout()
if "--save" in sys.argv:
    plt.savefig("mask_check_real.png", dpi=80); print("saved mask_check_real.png")
else:
    plt.show()
