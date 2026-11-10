import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import annotations
import pdb
import unet
import torch
import train_class as TrainBox
import dataset
import losses

IMAGE_DIR="./MAGFiLO_1.0_Kaggle_2026/train/train_images"
ANNOTATION_FILE="./MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"

if __name__=="__main__":
    coco = annotations.read_coco_annotations(ANNOTATION_FILE)
    #annotations.view_annotations(coco,num_samples=-1)
    
    model = unet.UNet(in_channels=1,out_channels=1)
    print(f'Model: {model}')
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    
    #split by image file so annotator copies of the same image never end up in both train and val
    train_ids, val_ids = dataset.split_by_file(coco, val_fraction=0.2, seed=42)
    print(f"Split by image file: {len(train_ids)} train entries, {len(val_ids)} validation entries")

    train_data=dataset.Dataset(
        image_dir=IMAGE_DIR,
        annotation_file=ANNOTATION_FILE,
        resize_to=1024,
        crop_size=512,          #512x512 crops for training
        img_ids=train_ids,
        filament_crop_prob=0.8  #80% of crops placed on a filament, 20% anywhere
    )
    val_data=dataset.Dataset(
        image_dir=IMAGE_DIR,
        annotation_file=ANNOTATION_FILE,
        resize_to=1024,
        crop_size=None,         #whole 1024x1024 image for validation
        img_ids=val_ids
    )

    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    #halve the learning rate when validation loss stops improving for 2 epochs
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=2)

    train = TrainBox.TrainBox(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=losses.BCEDiceLoss(),   #BCE + Dice: Dice is not swamped by the ~99.7% background pixels
        epochs=10,
        train_data=train_data,
        val_data=val_data)

    train.prep_data_loaders()
    train.train()
