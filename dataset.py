import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from torchvision import transforms

import pdb

class Dataset(Dataset):
    def __init__(self, image_dir, annotation_file, img_size=256):
        self.image_dir=image_dir
        self.coco=COCO(annotation_file)
        self.transforms=transforms
        self.img_size=img_size
        self.img_ids=self.coco.getImgIds()

        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # Resize to square
            transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0,1]
            transforms.Normalize(mean=[0.485], std=[0.229])  # Optional: normalize for pretrained models
        ])
        
        # Define transforms for mask (different from image!)
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), 
                             interpolation=transforms.InterpolationMode.NEAREST),  # Nearest for masks!
            transforms.ToTensor(),  # Converts to [C, H, W]
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self,idx):
        img_id=self.img_ids[idx]
        filename = img_id.split("-",1)[-1]
        #print(f"filename: {filename}")
        #get image data
        image=Image.open(f'{self.image_dir}/{filename}.jpeg').convert("L") #convert to grayscale
        #image=image[np.newaxis,...] #add channel dimension
        #print(f"Image path: {self.image_dir}/{filename}.jpeg")

        #get binary mask data
        mask=self.create_mask(img_id)

        image = self.image_transform(image)  # Shape: [1, 512, 512]
        mask = Image.fromarray(mask) 
        mask = self.mask_transform(mask)    # Shape: [1, 512, 512]

        #print(f"image got\n")
        #print(image.shape)
        return image,mask

    def create_mask(self,img_id):
        ann_ids=self.coco.getAnnIds(imgIds=[img_id])
        anns=self.coco.loadAnns(ann_ids)
        mask=np.zeros((self.img_size,self.img_size),dtype=np.uint8)

        for annotation in anns:
            rle=maskUtils.frPyObjects(annotation['segmentation'],self.img_size,self.img_size)
            m=maskUtils.decode(rle).squeeze(-1)
            mask=np.maximum(mask,m)
        return mask
