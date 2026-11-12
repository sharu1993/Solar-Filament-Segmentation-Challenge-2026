import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2
import preprocess

import pdb

def split_by_file(coco, val_fraction=0.2, seed=42):
    """Split image ids into train/val so that all annotator copies of one image file land on the same side."""
    all_ids=coco.getImgIds()
    files=sorted({coco.imgs[i]['file_name'] for i in all_ids})
    random.Random(seed).shuffle(files)
    val_files=set(files[:round(len(files)*val_fraction)])
    train_ids=[i for i in all_ids if coco.imgs[i]['file_name'] not in val_files]
    val_ids=[i for i in all_ids if coco.imgs[i]['file_name'] in val_files]
    return train_ids, val_ids

def augment_geometric(image, mask):
    """Same random flip / 90-degree rotation for image and mask (8 possible orientations)."""
    if random.random()<0.5:
        image, mask = image.flip(-1), mask.flip(-1)      #left-right
    if random.random()<0.5:
        image, mask = image.flip(-2), mask.flip(-2)      #up-down
    k=random.randrange(4)
    return image.rot90(k,dims=(-2,-1)), mask.rot90(k,dims=(-2,-1))

def augment_standardised(image, shift=0.2, contrast=0.2):
    """Brightness/contrast jitter for a preprocessed (standardised) image: random contrast scaling and
    shift, clipped to the preprocessing range (the mask is unchanged)."""
    c=1+random.uniform(-contrast,contrast)
    b=random.uniform(-shift,shift)
    return (image*c+b).clamp(-preprocess.CLIP,preprocess.CLIP)

def augment_photometric(image, brightness=0.1, contrast=0.2):
    """Random brightness shift and contrast scaling of a 0..1 image (the mask is unchanged)."""
    c=1+random.uniform(-contrast,contrast)
    b=random.uniform(-brightness,brightness)
    mean=image.mean()
    return ((image-mean)*c+mean+b).clamp(0,1)

class Dataset(Dataset):
    def __init__(self, image_dir, annotation_file, resize_to=1024, crop_size=512, img_ids=None, filament_crop_prob=0.8, augment=False,
                 preprocess=None):
        """
        resize_to: side length the full 2048x2048 image AND mask are shrunk to (2048 -> 1024 = 2x;
                   2048 = keep the original resolution).
        crop_size: side of the random crop taken (same window) from image and mask. None = no crop.
        img_ids: which image ids to use (e.g. from split_by_file). None = all.
        filament_crop_prob: chance that a crop is placed on a filament; otherwise it is placed anywhere.
        augment: random flips / 90-degree rotations (image and mask together) and brightness/contrast
                 jitter (image only). Use for training only.
        preprocess: None = raw image with the ImageNet normalisation (earlier runs);
                    "local" = limb darkening and local background divided out, standardised per image
                    (preprocess.local_normalise).
        """
        self.image_dir=image_dir
        self.coco=COCO(annotation_file)
        self.resize_to=resize_to
        self.crop_size=crop_size
        self.filament_crop_prob=filament_crop_prob
        self.augment=augment
        if preprocess not in (None,"local"):
            raise ValueError(f"Unknown preprocess '{preprocess}'")
        self.preprocess=preprocess
        self.img_ids=list(img_ids) if img_ids is not None else self.coco.getImgIds()


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self,idx):
        img_id=self.img_ids[idx]
        filename = img_id.split("-",1)[-1]
        #get image data
        image=Image.open(f'{self.image_dir}/{filename}.jpeg').convert("L") #convert to grayscale

        #get binary mask data at the image's full resolution (2048x2048)
        mask=self.create_mask(img_id)
        mask=Image.fromarray(mask*255)

        # 0) optional preprocessing at full resolution (before any resize): float image, ~0 on the quiet disk
        if self.preprocess=="local":
            image=preprocess.local_normalise(np.array(image))

        # 1) shrink image and mask by the same factor
        #    BOX = area averaging (best for downscaling); mask is averaged then thresholded back to 0/1
        size=(self.resize_to, self.resize_to)
        if mask.size!=size:   #at the original resolution there is nothing to resize
            if self.preprocess:
                image=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
            else:
                image=image.resize(size, resample=Image.BOX)
            mask=mask.resize(size, resample=Image.BOX)
        if self.preprocess:
            image=torch.from_numpy(image).unsqueeze(0)          # [1, H, W], already standardised

        # 2) take the SAME window from image and mask (usually placed on a filament, see choose_crop)
        if self.crop_size is not None:
            top, left = self.choose_crop(img_id)
            image=TF.crop(image, top, left, self.crop_size, self.crop_size)
            mask=TF.crop(mask, top, left, self.crop_size, self.crop_size)

        if not self.preprocess:
            image = TF.to_tensor(image)                                        # [1, crop, crop], values 0..1
        mask = torch.from_numpy((np.array(mask) > 127).astype(np.float32))    # back to 0/1
        mask = mask.unsqueeze(0)                                               # Shape: [1, crop, crop]
        if self.augment:
            image, mask = augment_geometric(image, mask)
            image = augment_standardised(image) if self.preprocess else augment_photometric(image)
        if not self.preprocess:
            image = TF.normalize(image, mean=[0.485], std=[0.229])             # same normalisation as before
        return image,mask

    def choose_crop(self,img_id):
        """Top-left corner (top, left) of the crop window, in resized (resize_to) coordinates."""
        size=self.resize_to
        c=self.crop_size
        anns=self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
        if anns and random.random()<self.filament_crop_prob:
            # pick one filament uniformly (small filaments are chosen as often as big ones),
            # then one point on its spine, which always lies on the filament
            ann=random.choice(anns)
            spine=np.array(ann['spine'],dtype=np.float64).reshape(-1,2)
            x,y=spine[random.randrange(len(spine))]
            info=self.coco.imgs[img_id]
            x=x*size/int(info['width'])
            y=y*size/int(info['height'])
            # place the window so that point lands at a random spot inside it (not always the centre)
            left=int(x)-random.randrange(c)
            top=int(y)-random.randrange(c)
            left=min(max(left,0),size-c)
            top=min(max(top,0),size-c)
            return top,left
        # otherwise: a window anywhere, so the model also sees background, the limb and off-disk sky
        return random.randrange(size-c+1), random.randrange(size-c+1)

    def create_mask(self,img_id):
        """Binary mask at the image's true size, with every filament of this annotator copy drawn in."""
        img_info=self.coco.imgs[img_id]
        height, width = int(img_info['height']), int(img_info['width'])
        mask=np.zeros((height,width),dtype=np.uint8)

        ann_ids=self.coco.getAnnIds(imgIds=[img_id])
        anns=self.coco.loadAnns(ann_ids)
        for annotation in anns:
            m=self.coco.annToMask(annotation)  # polygon -> (height, width) 0/1 array, using the image's real size
            mask=np.maximum(mask,m)
        return mask
