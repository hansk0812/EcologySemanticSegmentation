# AlvaradoLabSegmentation
#### Prerequisites:
1. Please email `hk2729@nyu.edu` for access to the dataset.
2. Download the models from the Google Drive link provided below (optional).
3. Download the required packages using `pip install -r requirements.txt`
4. Run the training script using the instructions provided below. If you followed step 2, you may also run the testing script below.

#### Dataset Tools:

Use Python script to separate cropped segmentation parts and use it for semantic segmentation labels
`python -m ecology_semantic_segmentation.dataset.utils` 

After finding too many such cases (~150 manually fixed examples), I have FINALLY decided to write a program to solve the problem!
`python -m ecology_semantic_segmentation.dataset.bbox_masks_problem`

#### U-Net based Semantic Segmentation

Using U-Net models from `segmentation_models_pytorch`
`pip install segmentation_models_pytorch `

#### Datasets:
  ML Training Set - AlvaradoLab annotated data - composite segmentation
  Cichlid Collection - AlvaradoLab annotated data - composite segmentation
  SUIM - semantic segmentation with >1 fish per image
  Deep Fish - fish_tray_images - Accurately labeled sharp masks - Semantic segmentation with >1 fish per image

##### Available Pre-trained Backbones:
  Resnet34
  Resnet50
  DeepLabv3Plus

**Link to trained models**: https://drive.google.com/drive/folders/1jrUJtpxR8WvUf3td9nWXMIEdPmsgK_gp?usp=share_link

#### Training script:

Use `SAMPLE=1`, `SAMPLE=0`, `IMG_SIZE=256`, `ORGANS="whole_body"` flags to control training and debugging training code
Using batch sizes as multiples of 9 gives most efficient use of GPU space

`ORGANS=whole_body,ventral_side,dorsal_side python -m ecology_semantic_segmentation.train_multiclass --batch_size 54`
`python -m ecology_semantic_segmentation.train --batch_size 54`


Learning Rate Scheduling for Adam: 0.0003 to start, re-define for every checkpoint resume manually using script parameter 

#### Testing Script:

Uses Dice score for accuracy
`ORGANS=<comma separated list of organs> python -m ecology_semantic_segmentation.test_multiclass --models <MODELDIR> --single_model <EPOCH_NUM>`

An example script would be:  
`ORGANS=whole_body,, python -m ecology_semantic_segmentation.test_multiclass --models ./deeplabv3plus --single_model 500`

This script will segment the whole body, but not the ventral side or dorsal side. The script will use a model stored in the `deeplabv3plus` directory, and use the model from epoch 500.


### 4-Connected lines from Photoshop annotation + 8-Connected lines from cv2 Polygon annotation = Anti-Aliased edges in CNN model result 
Reasons:
1. high-resolution
2. Supersampling
3. Intensity decisions based on object overlap
4. Line intensity assignment differences between straight and diagonal lines

https://www.geeksforgeeks.org/antialiasing/

### TODO: DEEPSUPERVISION CODE!


### TODO: Implement feature for model to accept video as input
We do not need to train a new model which takes video as input. Rather, we will take video, parse it into individual frames and pass that to the model. After the model is done processing the images or segmenting them, we will take those images and reassemble them into a video.

The general pipeline is as follows:
1. Parse video into frames.
2. Pass frames to the model to be segmented.
3. Aggregate frames back into a video.

Now there are two ways we could achieve this:
1. Create the feature that works with the current project pipeline (i.e. reading from the json file for project configuration and using the preexisting training script to kind of "inject" the frames into the model). What I mean by this is that there is code already that passes images to the model and saves the output. We would integrate the new feature to work with this preexisting code.
2. The second method is to create a standalone file/project that just loads the pretrained model file and asks it to segment the images.

I (one of the Spring 2024 interns) am not well versed enough to say which is the correct option, though I feel the first option sounds easier. It's just reading a lot of old code and understanding how it works. 
