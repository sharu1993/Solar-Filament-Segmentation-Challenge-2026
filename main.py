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

if __name__=="__main__":
    coco = annotations.read_coco_annotations("MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    #annotations.view_annotations(coco,num_samples=-1)
    
    model = unet.UNet(in_channels=1,out_channels=1)
    print(f'Model: {model}')
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    
    device=torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")

    model.to(device)

    data=dataset.Dataset(
        image_dir="./MAGFiLO_1.0_Kaggle_2026/train/train_images",
        annotation_file="./MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        img_size=256
    )

    train = TrainBox.TrainBox(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(),lr=0.001),
        criterion=torch.nn.BCELoss(),
        epochs=10,
        data=data)

    train.prep_data_loaders()
    train.train()
