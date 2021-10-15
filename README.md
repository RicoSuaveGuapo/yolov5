## Action items
1. Dataloader
2. Model
3. Inference
4. Metric


## 1. Dataloader

### Issue
Need to check the bbox size (increase or not) and mask
### Note
* The data is from `/root/Defect_detection/defect_data/green_crop`.
* I will turn off mosaic augmentation in dataloader save the effort
* One image with one mask, does not matter the instance counts.
* Remeber to use the same json files and data.yaml, they are decouple!
* In `/root/Defect_detection/defect_data/toyolov5_format.py`, there is function for splitting dataset
* I have turned off all the data argumentation
* letterbox will not change mask, if img size is 640x640

### Code Snippet
python train.py --enable_seg True --json_dir /root/Defect_detection/defect_data/green_crop/annotations

### Target
Transform the mask data to (0/1) tensor

### Process
* Mask output in `__getitiem__`

### Logic
First use sth like `json2yolov5` in `toyolov5_format.py` under `../Defect_detection/defect_data`, to get
1. image data
2. detection data
Let the mask data be read in the function in `dataset.py`

## 2. Model and Training

## 3. Inference

## 4. Metric
1. FPS
2. common use mask mAP
    - [x] `val.py` uses the function `process_batch()` (line 212) to compute bbox IoU and `correct` which is collected in the list `stats` later, and then uses the function `utils.ap_per_class()` (line 236) to compute AP.
    - [ ] need to modify `process_batch()` to compute mask IoU instead of bbox IoU. Specifically, replace `box_iou()` in `process_batch()` with a function that can compute mask IoU.