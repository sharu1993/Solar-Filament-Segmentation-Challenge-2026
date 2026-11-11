import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import subprocess
from datetime import datetime
import annotations
import pdb
import resnet_unet
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
        lr_min=1e-6,                  #never decay the learning rate below this
        augment=True,                 #random flips/rotations + brightness/contrast jitter on training crops
        epochs=100,                   #long run (~21 h on the M1 at ~12.5 min/epoch)
        early_stopping_patience=15,   #stop only if validation IoU has not improved for 15 epochs
        num_workers=4,                #prepare batches in 4 background processes
        model="resnet34_unet",        #"unet" (from scratch) or "resnet34_unet" (ImageNet-pretrained encoder)
        pretrained=True,              #load ImageNet weights into the encoder (resnet34_unet only)
    )

    #--resume checkpoints/<run>_last.pt (or an older best checkpoint <run>.pt): continue that run
    parser=argparse.ArgumentParser()
    parser.add_argument("--resume",default=None,help="checkpoint to continue training from")
    args=parser.parse_args()
    resume_state=None
    if args.resume:
        resume_state=torch.load(args.resume,map_location="cpu",weights_only=False)
        if resume_state.get("config") is not None:
            config=resume_state["config"]   #same settings as the interrupted run
        else:
            config["model"]="unet"   #checkpoints without a saved config predate the ResNet option
            print("Checkpoint has no saved config: using the config above with the plain U-Net (must match the original run).")

    coco = annotations.read_coco_annotations(ANNOTATION_FILE)
    #annotations.view_annotations(coco,num_samples=-1)
    
    model = resnet_unet.build_model(config.get("model","unet"), pretrained=config.get("pretrained",False) and resume_state is None)
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
        filament_crop_prob=config["filament_crop_prob"],
        augment=config["augment"]
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
        factor=config["lr_decay_factor"],patience=config["lr_decay_patience"],min_lr=config["lr_min"])

    #one checkpoint per run, named like the MLflow run (date + commit), so earlier models are never overwritten
    commit, dirty = git_commit()
    if resume_state is not None:
        #same run name and best-model file as the interrupted run
        checkpoint_path=resume_state.get("best_checkpoint_path") or args.resume
        run_name=os.path.splitext(os.path.basename(checkpoint_path))[0]
    else:
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
        run_config=config,
        train_data=train_data,
        val_data=val_data)

    train.prep_data_loaders()
    if resume_state is not None:
        train.load_state(resume_state)

    #MLflow: fixed location, named experiment, one run per training. The `with` block ends the run
    #properly even on a crash or Ctrl-C (marked FAILED / KILLED instead of left "running").
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    if resume_state is not None:
        #reopen the interrupted MLflow run (looked up by name for checkpoints that predate run ids)
        run_id=resume_state.get("mlflow_run_id")
        if run_id is None:
            found=mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{run_name}'",output_format="list")
            run_id=found[0].info.run_id if found else None
        with mlflow.start_run(run_id=run_id,run_name=None if run_id else run_name):
            mlflow.set_tag("resumed_at_commit",commit)
            train.train()
    else:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(config)
            mlflow.log_param("model_parameters", sum(p.numel() for p in model.parameters()))
            mlflow.set_tags({"git_commit":commit, "git_uncommitted_changes":dirty})
            train.train()
