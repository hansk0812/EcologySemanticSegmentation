import json 
import os
import glob
import traceback

import rawpy
import cv2
import numpy as np

from torch.utils.data import Dataset

from . import composite_labels

from ..augment import augment_fn
#from ..utils.bbox_masks_problem import remove_islands_in_segment_gt

def imread(file_path):
    
    if ".arw" not in file_path.lower():
        return cv2.imread(file_path)
    else:
        img = rawpy.imread(file_path) 
        img = img.postprocess() 
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 

class SegmentationDataset(Dataset):
    
    composite_labels = composite_labels
    def __init__(self, segmentation_data, img_shape, min_segment_positivity_ratio, sample_dataset = True, organs=None, augment_flag=False): 

#        if sample_dataset:
#            segmentation_data = {key: segmentation_data[key] for key in list(segmentation_data)[:60]}
        
        # Ensure all files contribute to data wrt organs
        if organs is None:
            test_organs = self.composite_labels
        else:
            test_organs = organs

        removable_keys = []
        for key in segmentation_data:
            
            ctx = 0
            for organ in test_organs:
                try:
                    imread(segmentation_data[key]["segments"][organ])
                    ctx += 1
                except Exception:
                    continue
            
            if ctx == 0:
                removable_keys.append(key)
        
        for key in removable_keys:
            del segmentation_data[key]

        self.segmentation_data = segmentation_data
        self.segmentation_keys = list(segmentation_data.keys())
        self.img_shape = img_shape 
        self.min_segment_positivity_ratio = min_segment_positivity_ratio
        self.organs = organs
        
        if not organs is None:
            self.label_indices = [self.composite_labels.index(organ) for organ in organs]
        else:
            self.label_indices = list(range(len(self.composite_labels)))

        self.set_augment_flag(augment_flag)

    def __len__(self):
        return len(self.segmentation_keys)
    
    def set_augment_flag(self, flag):
        self.augment_flag = flag
    
    def use_bbox_for_mask(self, img, mask, bbox_file="output.json"):
        
        with open(bbox_file, 'r') as f:
            bboxes_dict = json.load(f)
        
        mask = cv2.resize(mask, (256,256)) 

    def __getitem__(self, idx):
        
        data_key = self.segmentation_keys[idx]
        
        image_path, segments_paths = self.segmentation_data[data_key]["image"], self.segmentation_data[data_key]["segments"]
        image = imread(image_path)
        image = cv2.resize(image, (self.img_shape, self.img_shape))
        
        num_segments = len(self.composite_labels) if self.organs is None else len(self.organs)
        segment_array = np.zeros((self.img_shape, self.img_shape, num_segments)) 
        
        for label_index, organ_index in enumerate(self.label_indices):
            
            try:
                organ = self.composite_labels[organ_index]

                try:
                    segment = imread(segments_paths[organ])
                except Exception:
                    segment_array[:, :, label_index] = np.ones((self.img_shape, self.img_shape)) * -1 
                    continue

                segment = cv2.resize(segment, (self.img_shape, self.img_shape))

                segment = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
                
                segment = cv2.bitwise_not(segment)
                segment[segment>0] = 255
                
                #segment = remove_islands_in_segment_gt(segment)
                
                # Decide using this image: Machine learning training set (copy)/photos 1.30.2019/original image/f132C.png
                #SEGMENT_THRESHOLD = 225
                #segment[segment > SEGMENT_THRESHOLD] = 0
                #segment[segment != 0] = 255
                
                area_of_segment = segment.sum() / 255.0
                
                if area_of_segment * 255 < (self.min_segment_positivity_ratio * self.img_shape * self.img_shape):
                    #TODO: Ignore labels
                    segment.fill(0) # (-1)
                
                segment_array[:, :, label_index] = segment 
            
            except Exception:
                traceback.print_exc()
                segment_array[:, :, label_index] = np.ones((self.img_shape, self.img_shape)) * -1 
        
        if self.augment_flag:
            image, segment_array = augment_fn(image, segment_array)
        
        return image.transpose((2,0,1)).astype(np.float32) / 255.0, segment_array.transpose((2,0,1)).astype(np.float32) / 255.0, image_path

def get_ml_training_set_data(dtype, path, folder_path, img_shape, min_segment_positivity_ratio, 
                                sample_dataset=True, organs=None, bbox_dir=None ,augment_flag = True):
    
    #TODO: 9 missing images from dataset!

    global composite_labels 

    assert dtype == "segmentation/composite"
    
    folders = [x for x in glob.glob(os.path.join(folder_path, path, "*")) \
                if os.path.isdir(x)]
    
    if not bbox_dir is None:
        folders.append(os.path.join(folder_path, bbox_dir))
        folders = list(reversed(folders))

    data = {}
    for directory in folders:
        
        dir_folders = glob.glob(os.path.join(directory, "*"))
        
        images = glob.glob(os.path.join(directory, 'original image/*'))

        if sample_dataset:
            images = images[:20]

        for image_path in images:
            fname = "/".join(image_path.split('/')[-3:])
            search_key = '.'.join(fname.split('/')[-1].split('.')[:-1])
            data_index = os.path.join(directory.split('/')[-1], search_key)
            
            segments_path = glob.glob(os.path.join(directory, "*", search_key + "*"))
            segments = [x.split('/')[-2] for x in segments_path]
            segments.remove("original image")
            
            if not os.path.exists(image_path):
                #TODO print (image_path)
                continue

            segment_paths = {}
            for organ in segments:
                ann_paths = glob.glob(os.path.join(directory, organ, search_key + "*")) 
                
                organ = organ.replace(" ", "_")
                if not organ in composite_labels and organ != "original_image":
                    composite_labels.append(organ)

                if len(ann_paths) == 1:
                    if os.path.exists(ann_paths[0]):
                        if organs is None or (not organs is None and organ in organs):
                            segment_paths[organ] = ann_paths[0]
            
            if len(segment_paths) > 0:

                try:
                    img = cv2.imread(image_path)
                    assert not img is None

                    data[data_index] = {"image": image_path, \
                                        "segments": segment_paths}

                except Exception:
                    pass

    dataset = SegmentationDataset(data, img_shape, min_segment_positivity_ratio, sample_dataset=sample_dataset, organs=organs, augment_flag=augment_flag)
    print ("Using %d labeled images from dataset: %s!" % (len(dataset), "Segmentation dataset: %s" % path))
    
    return dataset

if __name__ == "__main__":
    
    data_dir = "/home/hans/data/"
    organs = ["whole_body", "head"]
    dset = get_ml_training_set_data(dtype="segmentation/composite", path="Machine learning training set/", folder_path=data_dir, 
            img_shape=256, min_segment_positivity_ratio=0.05, sample_dataset=True, organs=organs)#, bbox_dir="bbox_dependent_images_path")

    for img, seg, fpath in dset:
        
        print (seg[1].min(), seg[1].max())
        for idx in range(len(organs)):
            if -1 in seg[idx]:
                print ('Omitting %s annotation for file %s' % (organs[idx], fpath))

        cv2.imshow('f', img.transpose((1,2,0)))
        cv2.imshow('g', seg[0])
        cv2.imshow('i', seg[1])
        cv2.waitKey()

        print (img.shape, seg.shape, fpath)
