import sys
import gc
# sys computes data structure sizes for memory overflow handling

import os
import glob

import cv2
import numpy as np

from random import shuffle

from torch.utils.data import Dataset

from . import composite_labels, dataset_splits

from ..augment import augment_fn

#import tracemalloc

class CocoSegmentationDataset(Dataset):
    
    def __init__(self, coco_images, coco_txt, img_shape, min_segment_positivity_ratio=0.05, 
                    organs=None, sample_dataset=True, ann_format="xyxy", augment_flag = True):
        
        #TODO ann_format

        global composite_labels
        
        assert ann_format in ["xywh", "xyxy"] and len(coco_images) == len(coco_txt)
        
        if sample_dataset:
            coco_images, coco_txt = coco_images[:60], coco_txt[:60]
        
        self.organs = composite_organs if organs is None else organs
        self.img_shape = img_shape
        self.min_segment_positivity_ratio = min_segment_positivity_ratio
        self.image_paths = coco_images
        self.polygons = []
        
        deletion_indices = []
        for index, (image_path, objects_file) in enumerate(zip(coco_images, coco_txt)):
            
            try:
                image = cv2.imread(image_path)
                img_original_shape = image.shape
            except Exception:
                deletion_indices.append(index)
                continue

            with open(objects_file, 'r') as f:
                obj = [x.strip() for x in f.readlines()]
            
            num_objects = int(obj[0])
            h, w = [int(x) for x in obj[2].split(' ')]
        
            image_polys = []
            #TODO: Account for multiple polygons per object: while loop conditioned on # variable polygons
            for idx in range(4, len(obj), 4):
                organ = obj[idx]
                organ.replace(" ", "_")
                
                if not self.organs is None and not organ in self.organs:
                    continue

                if not organ in composite_labels:
                    if not organs is None:
                        if not organ in organs:
                            continue
                        if organ == "original_image":
                            continue
                    composite_labels.append(organ)

                area_of_poly = float(obj[idx+1])
                if area_of_poly == 0:
                    continue
               
                poly_indices = [int(float(x)) for x in obj[idx+2].split(' ')]
                polygon = [(poly_indices[i], poly_indices[i+1]) 
                                for i in range(0, len(poly_indices)-1, 2)]
                size_ratios = np.array([img_shape / img_original_shape[1], img_shape / img_original_shape[0]]) 
                image_polys.append({organ: np.array(polygon * size_ratios).astype(np.int32)})
            
            if len(image_polys)==0:
                deletion_indices.append(index)
            self.polygons.append(image_polys)

        for del_idx in reversed(deletion_indices):
            del self.image_paths[del_idx]
            del self.polygons[del_idx]
        
        assert len(self.image_paths) == len(self.polygons)
        
        self.set_augment_flag(augment_flag)

    def set_augment_flag(self, flag):
        self.augment_flag = flag

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image_path, polygons = self.image_paths[idx], self.polygons[idx]
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.img_shape, self.img_shape))

        num_segments = len(composite_labels) if self.organs is None else len(self.organs)
        segment_array = np.zeros((self.img_shape, self.img_shape, num_segments)) 
        
        organ_indices = set(range(len(self.organs)))
        for poly in polygons:
            organ, polygon = list(poly.keys())[0], list(poly.values())[0]
            
            organ_index = self.organs.index(organ)
            
            try:
                organ_indices.remove(organ_index)
            except Exception:
                
                assert organ_index != -1, \
                        "Organ %s cannot be found in %s" % (organ, ','.join(self.organs))
                
                segment_array[:, :, organ_index] = -1
                continue

            seg = segment_array[:, :, organ_index].astype(np.uint8) 

            cv2.fillPoly(seg, [polygon], 255) 
            
            if seg.sum() < (self.min_segment_positivity_ratio * self.img_shape * self.img_shape):
                seg.fill(-1)

            segment_array[:, :, organ_index] = seg 
        
        for empty_index in organ_indices:
            segment_array[:, :, empty_index] = np.ones_like(seg) * -1

        if self.augment_flag:
            image, segment_array = augment_fn(image, segment_array)
        
        return image.transpose((2,0,1)).astype(np.float32) /255.0, segment_array.transpose((2,0,1)).astype(np.float32) / 255.0, image_path

def get_alvaradolab_data(dtype, path, folder_path, img_shape, min_segment_positivity_ratio, sample_dataset=True, organs=None, bbox_dir=None, augment_flag = True):
    
    #tracemalloc.start()
    assert "segmentation/composite" in dtype

    images = glob.glob(os.path.join(folder_path, path, '*.jpg'))
    labels = [x.replace(".jpg", ".txt") for x in images]
    
    removable_indices = []
    for idx, (img, label) in enumerate(zip(images, labels)):
        if not (os.path.exists(img) and os.path.exists(label)):
            removable_indices.append(idx)

    for idx in reversed(removable_indices):
        del images[idx]
        del labels[idx]
    
    dataset = CocoSegmentationDataset(images, labels, img_shape, min_segment_positivity_ratio, sample_dataset=sample_dataset, organs=organs, augment_flag=augment_flag)
    print ("Using %d labeled images from dataset: %s!" % (len(dataset), "COCO Segmentation: %s" % path))
    
    """
    print("index: ", index, [x / 1e6 for x in tracemalloc.get_traced_memory()], "MB")
    Resize best case: reduction of 1 MB
    index:  1000 [5446.751448, 5451.705668] MB
    index:  2000 [10887.057792, 10887.674912] MB
    index:  2534 [13792.174833, 13792.456553] MB
    Killed
    """

    #tracemalloc.stop()

    return dataset

if __name__ == "__main__":
    
    dataset = get_alvaradolab_data(
                    dtype=["segmentation/composite"], 
                    path="Cichlid Picture Collection REVISED (UPDATED)/Annotated Photos/", 
                    folder_path="/home/hans/data", 
                    img_shape=256, 
                    min_segment_positivity_ratio=0.05, 
                    sample_dataset=True, 
                    organs=["whole_body", "head"], 
                    bbox_dir=None, 
                    augment_flag = True)

    for img, seg, fname in dataset:
        img = img.transpose((1,2,0))*255 
        cv2.imshow("f", img.astype(np.uint8))
        cv2.imshow("g", (seg[0]*255).astype(np.uint8))
        cv2.imshow("h", (seg[1]*255).astype(np.uint8))
        print (fname)
        cv2.waitKey()

