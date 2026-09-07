import cv2 as cv
from pycocotools.coco import COCO
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pdb

def read_coco_annotations(json_file):
    coco = COCO(json_file)

    #Print basic info
    print(f"Number of images: {len(coco.imgs)}")
    print(f"Number of annotations: {len(coco.anns)}")
    print(f"Number of categories: {len(coco.cats)}")
    return coco

def view_annotations(coco, img_id=None, num_samples=1):
    if img_id is None:
        img_ids=coco.getImgIds() #Get all image IDs
    else:
        img_ids=[img_id]

    #load iamges
    for id in img_ids[:num_samples]:

        #Get image data
        filename = id.split("-",1)[-1]
        img_path=f'./MAGFiLO_1.0_Kaggle_2026/train/train_images/{filename}.jpeg'
        image=plt.imread(img_path)

        #Display image
        fig,ax = plt.subplots()#plt.figure(figsize=(12,10))
        ax.imshow(image)

        #get annotation data
        ann_id=coco.getAnnIds(imgIds=[id]) #convert single id to a list, will not work otherwise
        anns=coco.loadAnns(ann_id)

        #bounding boxes of annotations
        for ann in anns:
            bbox=ann['bbox']
            x,y,w,h=bbox
            rect=patches.Rectangle((x,y),w,h,fill=False,color='red')
            ax.add_patch(rect)

        plt.show()
        