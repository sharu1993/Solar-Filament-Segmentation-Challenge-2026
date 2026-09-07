import annotations
import pdb

if __name__=="__main__":
    coco = annotations.read_coco_annotations("MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    pdb.set_trace()
    annotations.view_annotations(coco,num_samples=-1)
    pdb.set_trace()