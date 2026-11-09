import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from torchvision import transforms
import torchvision.transforms.functional as TF

import pdb

class Dataset(Dataset):
    def __init__(self, image_dir, annotation_file, resize_to=1024, crop_size=512):
        """
        resize_to: side length the full 2048x2048 image AND mask are shrunk to (2048 -> 1024 = 2x).
        crop_size: side of the random crop taken (same window) from image and mask. None = no crop.
        """
        self.image_dir=image_dir
        self.coco=COCO(annotation_file)
        self.resize_to=resize_to
        self.crop_size=crop_size
        self.img_ids=self.coco.getImgIds()

        # Applied after resize + crop
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0,1]
            transforms.Normalize(mean=[0.485], std=[0.229])  # Optional: normalize for pretrained models
        ])

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

        # 1) shrink image and mask by the same factor
        #    BOX = area averaging (best for downscaling); mask is averaged then thresholded back to 0/1
        size=(self.resize_to, self.resize_to)
        image=image.resize(size, resample=Image.BOX)
        mask=mask.resize(size, resample=Image.BOX)

        # 2) take the SAME random window from image and mask
        if self.crop_size is not None:
            top, left, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
            image=TF.crop(image, top, left, h, w)
            mask=TF.crop(mask, top, left, h, w)

        image = self.image_transform(image)                                    # Shape: [1, crop, crop]
        mask = torch.from_numpy((np.array(mask) > 127).astype(np.float32))    # back to 0/1
        mask = mask.unsqueeze(0)                                               # Shape: [1, crop, crop]
        return image,mask

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
