import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import subprocess
from datetime import datetime
import annotations
import pdb
import unet
import torch
import mlflow
import train_class as TrainBox
import dataset
import losses

IMAGE_DIR="./MAGFiLO_1.0_Kaggle_2026/train/train_images"
ANNOTATION_FILE="./MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
MLFLOW_TRACKING_URI="sqlite:///mlflow.db"   #mlflow.db in the project folder (run main.py from here); view with `mlflow ui`
MLFLOW_EXPERIMENT="filament-segmentation"

def git_commit():
    """Short hash of the current commit, and whether tracked files have uncommitted changes."""
    try:
        commit=subprocess.run(["git","rev-parse","--short","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
        dirty=bool(subprocess.run(["git","--no-optional-locks","status","--porcelain","--untracked-files=no"],capture_output=True,text=True,check=True).stdout.strip())
        return commit, dirty
    except Exception:
        return "unknown", False

if __name__=="__main__":
    #all settings in one place: used below AND logged to MLflow, so each run records how it was made
    config=dict(
        resize_to=1024,
        crop_size=512,                #512x512 crops for training
        filament_crop_prob=0.8,       #80% of crops placed on a filament, 20% anywhere
        val_fraction=0.2,
        split_seed=42,
        lr=0.001,
        lr_decay_factor=0.5,          #halve the learning rate...
        lr_decay_patience=2,          #...when validation loss stops improving for 2 epochs
        epochs=30,                    #maximum; early stopping usually ends sooner
        early_stopping_patience=5,    #stop if validation IoU has not improved for 5 epochs
        num_workers=4,                #prepare batches in 4 background processes
    )

    coco = annotations.read_coco_annotations(ANNOTATION_FILE)
    #annotations.view_annotations(coco,num_samples=-1)
    
    model = unet.UNet(in_channels=1,out_channels=1)
    print(f'Model: {model}')
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    
    #split by image file so annotator copies of the same image never end up in both train and val
    train_ids, val_ids = dataset.split_by_file(coco, val_fraction=config["val_fraction"], seed=config["split_seed"])
    print(f"Split by image file: {len(train_ids)} train entries, {len(val_ids)} validation entries")

    train_data=dataset.Dataset(
        image_dir=IMAGE_DIR,
        annotation_file=ANNOTATION_FILE,
        resize_to=config["resize_to"],
        crop_size=config["crop_size"],
        img_ids=train_ids,
        filament_crop_prob=config["filament_crop_prob"]
    )
    val_data=dataset.Dataset(
        image_dir=IMAGE_DIR,
        annotation_file=ANNOTATION_FILE,
        resize_to=config["resize_to"],
        crop_size=None,         #whole image for validation
        img_ids=val_ids
    )

    optimizer=torch.optim.Adam(model.parameters(),lr=config["lr"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",
        factor=config["lr_decay_factor"],patience=config["lr_decay_patience"])

    #one checkpoint per run, named like the MLflow run (date + commit), so earlier models are never overwritten
    commit, dirty = git_commit()
    run_name=f"{datetime.now():%Y-%m-%d_%H-%M}_{commit}"
    checkpoint_path=os.path.join("checkpoints",f"{run_name}.pt")

    train = TrainBox.TrainBox(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=losses.BCEDiceLoss(),   #BCE + Dice: Dice is not swamped by the ~99.7% background pixels
        epochs=config["epochs"],
        early_stopping_patience=config["early_stopping_patience"],
        num_workers=config["num_workers"],
        checkpoint_path=checkpoint_path,
        train_data=train_data,
        val_data=val_data)

    train.prep_data_loaders()

    #MLflow: fixed location, named experiment, one run per training. The `with` block ends the run
    #properly even on a crash or Ctrl-C (marked FAILED / KILLED instead of left "running").
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)
        mlflow.log_param("model_parameters", sum(p.numel() for p in model.parameters()))
        mlflow.set_tags({"git_commit":commit, "git_uncommitted_changes":dirty})
        train.train()
