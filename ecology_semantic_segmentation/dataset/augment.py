import numpy as np

import cv2
import albumentations as A
from albumentations.augmentations.transforms import (
        FancyPCA, 
        HueSaturationValue, 
        CLAHE, 
        ColorJitter, 
        Emboss, 
        RandomFog, 
        ChannelShuffle, 
        RandomToneCurve,
        ToGray, 
        UnsharpMask)

from albumentations.augmentations.geometric.functional import rotate
from albumentations.augmentations.blur.transforms import GaussianBlur, Defocus, ZoomBlur

from albumentations.augmentations.domain_adaptation import FDA

def augment_fn(image, masks, img_size=256):
    
    target_image = np.random.randint(0, 256, image.shape, dtype=np.uint8)
    transforms = A.Compose([
                        A.OneOf([
                               Defocus(radius = (3, 3), alias_blur = (0.1, 0.1), p = 1),
                               GaussianBlur(blur_limit = (3,3), sigma_limit = (0.2,0.2), p = 1),
                               ZoomBlur(max_factor = 1.11, step_factor = (0.01, 0.02), p = 1),
                               RandomFog(fog_coef_lower = 0.3, fog_coef_upper = 1, alpha_coef = 0.08, p = 0.4),
                           ], p=0.4),
                        A.OneOf([
                                ColorJitter(hue=0.4, brightness=0.4, contrast=0.4, saturation=0.4, p=0.3),
                                A.RandomBrightnessContrast(p=0.5),
                                A.RandomGamma(p=0.5),
                                Emboss(alpha=(0.3, 0.6), strength=(0.3, 0.7), p=0.3),
                            ], p=0.4),
                         A.RandomResizedCrop(img_size, img_size, p=0.3),
                         #A.CropAndPad(percent=[30, 20]), # makes it too slow
                         #FDA([target_image], p=0.2, read_fn=lambda x: x), # makes it too slow
#                        A.OneOf([
#                            A.RandomCrop(width=128, height=128, p=0.5),
#                            A.RandomCrop(width=64, height=64, p=0.5),
#                            ], p=0.5),
# Batch Size 1 for random sizes training
                       A.HorizontalFlip(p=0.5),
                       FancyPCA(p=0.3, alpha=0.35),
                       ChannelShuffle(p=0.5),
                       ToGray(p=0.3),
                       # this forces edge detection to create false positives if trained for long
                       #UnsharpMask(blur_limit=(3, 7), \
                       #        sigma_limit=0.0, alpha=(0.2, 0.5), threshold=10, always_apply=True, p=0.1) 
                       ], p=0.7)

    transform = transforms(image=image, mask=masks)
    image, masks = transform["image"], np.array(transform["mask"])
    
    if np.random.rand() < 0.4:
        color_transform = HueSaturationValue(p = 1, hue_shift_limit = [-60, 60], sat_shift_limit = [-60, 60], val_shift_limit = [-30, 30])
        image = color_transform(image=image)["image"]
    
    if np.random.rand() < 0.7:
        clahe_transform = CLAHE(p=1, clip_limit = [1, 4.0], tile_grid_size= [8, 8])  
        image = clahe_transform(image=image)["image"]
    
    if np.random.rand() < 0.4:
        image, masks = Arotate(image, masks, p=1)
    
    if np.random.rand() < 0.5:
        brightness_transform = RandomToneCurve(scale=0.25, p=0.5)
        image = brightness_transform(image=image)["image"]

    del target_image

    return image, masks

def Arotate(image, masks, degree=None, p=0.5):

    if degree is None:
        degree = np.random.randint(90)
        if np.random.rand() <= 0.2:
            degree = 0
    else:
        assert 0 < degree < 90, "Invalid degree ranges for rotate augmentation!"

    if np.random.rand() <= p: 
        image = rotate(image, degree)
        masks = rotate(masks, degree)

    return image, masks

if __name__ == "__main__":

    from .fish import fish_train_dataset
    import time

    for data in fish_train_dataset:
        img, segment, fname = data
        print (img.shape, segment.shape, img.min(), img.max(), segment.min(), segment.max(), fname)
        img, segment = (img.transpose((1,2,0))*255).astype(np.uint8), \
                       segment.transpose((1,2,0)).astype(np.uint8)*255
        print ("saving f.png")
        cv2.imwrite('f.png', img)
        #cv2.imwrite('fm.png', segment[:,:,0])
        
        augment_fn = UnsharpMask (blur_limit=(3, 7), sigma_limit=0.0, alpha=(0.2, 0.5), threshold=10, always_apply=True, p=1)
        aug = augment_fn(image=img)
        img = aug["image"]
        cv2.imwrite('unsharpmask.png', img)
        exit()

        img, segment = augment_fn(img, segment)
        print ("saving g.png")
        cv2.imwrite('g.png', img)
        cv2.imwrite('gm.png', segment[:,:,0])

        time.sleep(4)
